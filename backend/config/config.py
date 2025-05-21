import os
import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, Extra, computed_field, ConfigDict
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger("backend.config") # Используем специфичный логгер

class CommonSettings(BaseModel):
    model_base_path: str = "backend/ml_service/models"
    log_dir: str = "logs"
    cors_origins: List[Union[HttpUrl, str]] = Field(default_factory=lambda: [str(HttpUrl("http://localhost:5173"))])
    llm_model_name: str = "qwen3:latest"
    ollama_base_url: HttpUrl = Field(default_factory=lambda: HttpUrl("http://localhost:11434"))

class APISettings(BaseModel):
    title: str = "Anomaly Detection API"
    description: str = "API for anomaly detection and e-commerce data management."
    version: str = "0.1.0"

class DetectorConfigSchema(BaseModel):
    type: str
    model_filename: str
    weight: Optional[float] = 1.0
    class Config:
        extra = Extra.allow # Разрешает дополнительные поля, не описанные явно

class MultilevelDetectorConfigSchema(BaseModel):
    transaction_level: List[DetectorConfigSchema]
    behavior_level: List[DetectorConfigSchema]
    time_series_level: List[DetectorConfigSchema]
    combination_weights: Dict[str, float]
    transaction_level_combination_method: Optional[str] = "weighted_average"
    behavior_level_combination_method: Optional[str] = "weighted_average"
    time_series_level_combination_method: Optional[str] = "weighted_average"

class MLServiceSettings(BaseModel):
    multilevel_detector_default_config: MultilevelDetectorConfigSchema

class AppSettings(BaseModel):
    DATABASE_URL: str
    ENVIRONMENT: str = Field(default="development") # Будет переопределено .env -> YAML
    APP_BASE_DIR: str # Абсолютный путь к корню проекта

    common: CommonSettings
    api: APISettings
    ml_service: MLServiceSettings

    @field_validator('DATABASE_URL')
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DATABASE_URL не может быть пустым")
        return v

    @computed_field
    @property
    def DATABASE_URL_SAFE_LOGGING(self) -> str:
        """Возвращает DATABASE_URL с замаскированным паролем для безопасного логирования."""
        try:
            parsed_url = urlparse(self.DATABASE_URL)
            if parsed_url.password:
                new_netloc = parsed_url.hostname or ""
                if parsed_url.username:
                    new_netloc = f"{parsed_url.username}:********@{new_netloc}"
                else:
                    new_netloc = f"********@{new_netloc}"
                
                if parsed_url.port:
                    new_netloc = f"{new_netloc}:{parsed_url.port}"
                safe_url_parts = list(parsed_url)
                safe_url_parts[1] = new_netloc # netloc - это второй элемент (индекс 1)
                return urlunparse(tuple(safe_url_parts))
            else:
                return self.DATABASE_URL
        except Exception: # В случае ошибки парсинга или другой проблемы
            logger.warning(f"Не удалось безопасно замаскировать DATABASE_URL: {self.DATABASE_URL}", exc_info=True)
            return "DATABASE_URL_MASKING_ERROR" # Или более общее сообщение

# --- Глобальный экземпляр настроек ---
_settings_instance: Optional[AppSettings] = None

# --- Определение путей ---
# Директория, где находится этот файл (config.py), т.е. backend/config/
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Директория backend/
_BACKEND_DIR_FROM_CONFIG = os.path.dirname(_CURRENT_FILE_DIR)
# Корень проекта (например, ANOMALENS3.0/)
_PROJECT_ROOT_RESOLVED = os.path.dirname(_BACKEND_DIR_FROM_CONFIG)

DEFAULT_YAML_CONFIG_PATH = os.path.join(_CURRENT_FILE_DIR, "config.yaml")

# --- Вспомогательные функции для load_settings ---

def _load_env_variables(project_root_path: str) -> None:
    """Загружает переменные окружения из .env файла в корне проекта."""
    dotenv_path = os.path.join(project_root_path, '.env')
    loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
    if loaded_env:
        logger.info(f"Переменные окружения загружены из: {dotenv_path}")
    else:
        logger.warning(f"Файл .env не найден или не удалось загрузить по пути: {dotenv_path}")
        if not os.path.exists(dotenv_path):
            logger.warning(f"Подтверждение: Файл {dotenv_path} не существует.")

def _load_yaml_data(config_path_to_load: str) -> dict:
    """Загружает данные из YAML файла конфигурации."""
    yaml_data = {}
    if os.path.exists(config_path_to_load):
        try:
            with open(config_path_to_load, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из YAML: {config_path_to_load}")
        except Exception as e:
            logger.error(f"Ошибка загрузки YAML ({config_path_to_load}): {e}", exc_info=True)
            raise RuntimeError(f"Не удалось загрузить YAML: {e}")
    else:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: YAML файл не найден: {config_path_to_load}")
        raise FileNotFoundError(f"YAML файл не найден: {config_path_to_load}")
    return yaml_data

def _build_common_settings(yaml_common_data: dict) -> CommonSettings:
    """Создает экземпляр CommonSettings на основе данных из YAML."""
    common_settings_pydantic_defaults = CommonSettings.model_fields
    
    # Получаем cors_origins из YAML или из default_factory
    raw_cors_origins = yaml_common_data.get("cors_origins", common_settings_pydantic_defaults["cors_origins"].default_factory())

    return CommonSettings(
        model_base_path=yaml_common_data.get("model_base_path", common_settings_pydantic_defaults["model_base_path"].default),
        log_dir=yaml_common_data.get("log_dir", common_settings_pydantic_defaults["log_dir"].default),
        cors_origins=raw_cors_origins,
        llm_model_name=yaml_common_data.get("llm_model_name", common_settings_pydantic_defaults["llm_model_name"].default),
        ollama_base_url=yaml_common_data.get("ollama_base_url", common_settings_pydantic_defaults["ollama_base_url"].default_factory())
    )

def _resolve_and_create_paths(settings: AppSettings, project_root_path: str) -> None:
    """Преобразует пути в абсолютные и создает необходимые директории."""
    abs_model_base_path = settings.common.model_base_path
    if not os.path.isabs(abs_model_base_path):
        abs_model_base_path = os.path.join(project_root_path, abs_model_base_path)
    os.makedirs(abs_model_base_path, exist_ok=True)
    settings.common.model_base_path = abs_model_base_path
    logger.info(f"Абсолютный путь для моделей: {settings.common.model_base_path}")

    abs_log_dir = settings.common.log_dir
    if not os.path.isabs(abs_log_dir):
        abs_log_dir = os.path.join(project_root_path, abs_log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)
    settings.common.log_dir = abs_log_dir
    logger.info(f"Абсолютный путь для логов: {settings.common.log_dir}")

def load_settings(yaml_config_path: Optional[str] = None) -> AppSettings:
    global _settings_instance
    if _settings_instance is not None:
        return _settings_instance

    _load_env_variables(_PROJECT_ROOT_RESOLVED)

    config_path_to_load = yaml_config_path if yaml_config_path is not None else DEFAULT_YAML_CONFIG_PATH
    yaml_data = _load_yaml_data(config_path_to_load)
    
    yaml_common_data = yaml_data.get("common", {})

    common_config = _build_common_settings(yaml_common_data)

    db_url_from_env = os.getenv("DATABASE_URL")
    
    app_env_from_os = os.getenv("ENVIRONMENT")
    app_env_from_yaml_common = yaml_common_data.get("environment") # Читаем из YAML common для ENVIRONMENT
    determined_environment = app_env_from_os or app_env_from_yaml_common

    settings_data = {
        "DATABASE_URL": db_url_from_env,
        "APP_BASE_DIR": _PROJECT_ROOT_RESOLVED,
        "common": common_config,
        "api": APISettings(**yaml_data.get("api", {})),
        "ml_service": MLServiceSettings(**yaml_data.get("ml_service", {})),
    }
    if determined_environment is not None:
        settings_data["ENVIRONMENT"] = determined_environment
    
    try:
        current_settings_instance = AppSettings(**settings_data)
    except Exception as e:
        logger.error(f"Ошибка при инициализации настроек Pydantic: {e}", exc_info=True)
        raise

    _resolve_and_create_paths(current_settings_instance, _PROJECT_ROOT_RESOLVED)

    _settings_instance = current_settings_instance # Присваиваем глобальному экземпляру
    logger.info(f"Настройки успешно загружены и провалидированы. Окружение: {_settings_instance.ENVIRONMENT}")
    return _settings_instance

def get_settings() -> AppSettings:
    if _settings_instance is None:
        logger.critical("Критическая ошибка: Экземпляр настроек не был инициализирован. Вызовите load_settings() при старте приложения.")
        try:
            return load_settings()
        except Exception as e:
            logger.critical(f"Критическая ошибка: Аварийная загрузка настроек также не удалась: {e}", exc_info=True)
            raise RuntimeError("Не удалось инициализировать настройки приложения.")
    return _settings_instance