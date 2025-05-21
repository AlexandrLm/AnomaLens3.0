import logging
from sqlalchemy import create_engine, Engine # Добавляем Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional

# ----- Конфигурация подключения к базе данных -----

def get_engine(database_url: str) -> Engine: # Функция для создания engine
    """Создает движок SQLAlchemy на основе DATABASE_URL."""
    engine_args = {}
    if database_url.startswith("sqlite"):
        engine_args["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **engine_args)

# Создаем базовый класс для ORM-моделей
Base = declarative_base()

# ----- Инициализация глобальных переменных engine и SessionLocal ---
engine: Engine | None = None
SessionLocal: Optional[sessionmaker] = None

def initialize_database_session(app_settings): # Передаем настройки
    """Инициализирует engine и SessionLocal на основе загруженных настроек."""
    global engine, SessionLocal
    
    if engine is None: # Инициализируем только один раз
        engine = get_engine(app_settings.DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info(f"SQLAlchemy Engine и SessionLocal инициализированы для URL: {app_settings.DATABASE_URL_SAFE_LOGGING}") # Логируем безопасно
    return engine, SessionLocal

# ----- Вспомогательные функции -----
def create_db_and_tables(db_engine: Engine): # Принимает engine
    """Создает все таблицы в базе данных на основе моделей SQLAlchemy."""
    try:
        Base.metadata.create_all(bind=db_engine)
        logger.info("Таблицы в базе данных успешно созданы (если их не было).")
    except Exception as e:
        logger.error(f"Ошибка при создании таблиц: {e}", exc_info=True)
        raise # Перевыбрасываем ошибку, чтобы приложение могло ее обработать

def get_db():
    """Функция-генератор для получения сессии базы данных."""
    if SessionLocal is None:
        # Этого не должно происходить, если initialize_database_session вызван при старте
        logger.error("SessionLocal не инициализирован. Вызовите initialize_database_session() при старте приложения.")
        raise RuntimeError("SessionLocal не инициализирован.")
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logger = logging.getLogger(__name__)