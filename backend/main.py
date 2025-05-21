"""
main.py - Основной файл FastAPI приложения
==========================================
Этот модуль содержит настройку приложения FastAPI,
подключение CORS, маршрутизацию и инициализацию базы данных.
"""

# --- ПРИНУДИТЕЛЬНАЯ УСТАНОВКА УРОВНЯ ЛОГИРОВАНИЯ ДЛЯ ML_SERVICE (В САМОМ НАЧАЛЕ) ---
import logging
# Настройка базового логирования на случай критических ошибок при старте
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ROOT - %(message)s')

# Загрузка конфигурации приложения - должна быть одной из первых операций
try:
    from backend.config.config import load_settings, AppSettings
    settings: AppSettings = load_settings()
    logging.info("Конфигурация приложения успешно загружена.")
except Exception as e:
    logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить конфигурацию приложения: {e}", exc_info=True)
    exit(1) # Выход из приложения, если конфигурация не загружена


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Импортируем роутеры API и модели/схемы
from backend.api import anomalies, orders, settings as api_settings_router, geolocation, multilevel
from backend.api.utils import create_generic_router
from backend import models
from backend import schemas as root_schemas # Переименуем для ясности

# --- Импорт для TaskManager ---
from backend.services.task_manager import router as task_manager_router
# ------------------------------

# --- SQLAlchemy Engine и SessionLocal будут инициализированы в lifespan ---
from backend.database import initialize_database_session, create_db_and_tables
# ----------------------------------------------------

# =============================================================================
# Lifespan для управления ресурсами приложения
# =============================================================================
@asynccontextmanager
async def lifespan(app_lifespan: FastAPI):
    """
    Асинхронный context manager для управления жизненным циклом приложения.
    Инициализирует базу данных при запуске.
    """
    logging.info("Lifespan: Запуск приложения...")
    # Инициализация БД
    # Используем 'settings' из глобальной области видимости, загруженные ранее
    engine, _ = initialize_database_session(settings)
    if engine:
        create_db_and_tables(engine)
        logging.info("Lifespan: База данных успешно инициализирована.")
    else:
        # initialize_database_session должна бы выбросить исключение, если engine создать не удалось.
        # Если она возвращает None без исключения, это проблема в самой функции.
        logging.critical("Lifespan: SQLAlchemy Engine не был инициализирован (engine is None). Завершение работы.")
        # В реальном приложении, если initialize_database_session не выбрасывает исключение,
        # то здесь нужно либо выбросить его, либо корректно завершить работу.
        raise RuntimeError("Lifespan: Не удалось инициализировать SQLAlchemy Engine.")

    yield

    logging.info("Lifespan: Завершение работы приложения...")

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    lifespan=lifespan # <--- Регистрируем lifespan
)

# Настройка CORS
cors_origins_str = []
for origin_item in settings.common.cors_origins:
    cors_origins_str.append(str(origin_item))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_str,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.info(f"CORS настроен для следующих origins: {cors_origins_str}")

try:
    from backend.ml_service import logger as ml_logger_instance # Импортируем настроенный логгер
    if ml_logger_instance:
        ml_logger_instance.info(
            f"ML Service Logger активен. Уровень: {logging.getLevelName(ml_logger_instance.level)}. "
            f"Директория логов ml_service: {settings.common.log_dir}"
        )
    else:
        logging.warning("Экземпляр логгера ml_service не был получен.")
except ImportError:
    logging.error("Не удалось импортировать логгер ml_service. Логирование для этого модуля может быть не настроено.")
except Exception as e:
    logging.error(f"Ошибка доступа к логгеру ml_service в main.py: {e}", exc_info=True)


@app.get("/", tags=["Root"])
async def read_root():
    """
    Корневой эндпоинт для проверки работоспособности API.
    Возвращает приветственное сообщение и текущее окружение.
    """
    return {"message": f"Welcome to the {settings.api.title}", "environment": settings.ENVIRONMENT}

# Подключение роутеров API
app.include_router(anomalies.router, prefix="/api/anomalies")
app.include_router(orders.router, prefix="/api/orders", tags=["Orders"])
app.include_router(api_settings_router.router, prefix="/api/settings", tags=["Settings"]) # Используем переименованный импорт
app.include_router(geolocation.router, prefix="/api/geolocation", tags=["Geolocation"])
app.include_router(multilevel.router, prefix="/api/multilevel", tags=["Multilevel Anomaly Detection"])

# --- Подключение роутера для TaskManager ---
app.include_router(task_manager_router, prefix="/api/tasks", tags=["Tasks Management"])
# -----------------------------------------

# Подключение CRUD роутеров для основных сущностей
app.include_router(
    create_generic_router(models.Product, root_schemas.Product, "/api/products", ["Products"])
)
app.include_router(
    create_generic_router(models.Customer, root_schemas.Customer, "/api/customers", ["Customers"])
)
app.include_router(
    create_generic_router(models.Seller, root_schemas.Seller, "/api/sellers", ["Sellers"])
)
app.include_router(
    create_generic_router(models.OrderReview, root_schemas.OrderReview, "/api/reviews", ["Reviews"])
)
app.include_router(
    create_generic_router(
        models.ProductCategoryNameTranslation,
        root_schemas.ProductCategoryNameTranslation,
        "/api/translations",
        ["Translations"]
    )
)

# Команда для запуска (пример):
# uvicorn backend.main:app --reload --port 8001 --host 0.0.0.0