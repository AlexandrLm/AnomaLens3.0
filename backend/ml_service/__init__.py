"""
ml_service - Сервис обнаружения аномалий для AnomaLens 3.0
===========================================================
Этот пакет содержит компоненты для обнаружения аномалий в данных электронной коммерции,
включая различные алгоритмы и ансамблевый подход для повышения точности.
"""

import logging
import sys
import os
from datetime import datetime

# Попытка импортировать настройки. Если не удается (например, при unit-тестировании
# этого модуля в изоляции), будут использованы значения по умолчанию.
try:
    from backend.config.config import get_settings
    settings = get_settings()
    DEFAULT_LOG_DIR = settings.common.log_dir
    # Уровень логирования из настроек, если есть, иначе DEBUG
    DEFAULT_LOG_LEVEL_STR = settings.ENVIRONMENT
    if DEFAULT_LOG_LEVEL_STR.lower() == "production":
        DEFAULT_LOG_LEVEL = logging.INFO
    else:
        DEFAULT_LOG_LEVEL = logging.DEBUG # Для development и других
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ (ml_service/__init__): Не удалось импортировать настройки. Используются пути/уровни логирования по умолчанию.")
    DEFAULT_LOG_DIR = "logs" # Дефолт, если настройки не доступны
    DEFAULT_LOG_LEVEL = logging.DEBUG

# Конфигурация логирования для всего модуля ml_service
def setup_logging(log_level=DEFAULT_LOG_LEVEL, log_to_file=True, log_dir=DEFAULT_LOG_DIR):
    """
    Настраивает логирование для всего модуля ml_service.
    
    Args:
        log_level: Уровень логирования (INFO, DEBUG и т.д.)
        log_to_file: Включает логирование в файл
        log_dir: Директория для файлов логов
    """
    # Создаем корневой логгер для ml_service
    logger = logging.getLogger("backend.ml_service")
    logger.setLevel(log_level)
    logger.propagate = False  # Не пропускать логи вверх по иерархии
    
    # Очищаем существующие обработчики, если они есть
    if logger.handlers:
        logger.handlers.clear()
    
    # Создаем форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Добавляем обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Добавляем обработчик для записи в файл, если нужно
    if log_to_file:
        # Создаем директорию для логов, если её нет
        os.makedirs(log_dir, exist_ok=True)
        
        # Создаем имя файла лога с текущей датой
        log_file = os.path.join(
            log_dir, 
            f"ml_service_{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        
        # Создаем обработчик для файла
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Логируем информацию о настройке
    logger.info(f"Логирование настроено с уровнем {logging.getLevelName(log_level)}")
    if log_to_file:
        logger.info(f"Логи записываются в файл: {log_file}")
    
    return logger

# Настраиваем логирование при импорте пакета
logger = setup_logging()

# Версия пакета
__version__ = "1.0.0"

# Экспортируем основные классы и функции для удобства импорта
from .detector import AnomalyDetector, StatisticalDetector, IsolationForestDetector, AutoencoderDetector
from .vae_detector import VAEDetector
from .graph_detector import GraphAnomalyDetector
from .detector_factory import DetectorFactory
from .common import load_data_from_db, engineer_features
from .metrics import compute_binary_metrics
from .monitoring import ModelMonitor, create_monitors_for_ensemble 