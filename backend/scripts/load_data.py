import os
import pandas as pd
import logging
import time
from sqlalchemy.exc import SQLAlchemyError

# Импортируем ТОЛЬКО Base и initialize_database_session.
# Engine получать будем из возвращаемого значения initialize_database_session.
from ..database import Base, initialize_database_session
from .. import models # Импортируем все модели

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к директории с данными
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

# Сопоставление имен файлов CSV (остается без изменений)
CSV_MODEL_MAP = {
    "olist_customers_dataset": (models.Customer, 'customer_id'),
    "olist_geolocation_dataset": (models.Geolocation, ('geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng')),
    "olist_orders_dataset": (models.Order, 'order_id'),
    "olist_order_items_dataset": (models.OrderItem, ('order_id', 'order_item_id')),
    "olist_order_payments_dataset": (models.OrderPayment, ('order_id', 'payment_sequential')),
    "olist_order_reviews_dataset": (models.OrderReview, 'review_id'),
    "olist_products_dataset": (models.Product, 'product_id'),
    "olist_sellers_dataset": (models.Seller, 'seller_id'),
    "product_category_name_translation": (models.ProductCategoryNameTranslation, 'product_category_name')
}

# Колонки с датами (остается без изменений)
DATE_COLUMNS_MAP = {
    "olist_orders_dataset": [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date'
    ],
    "olist_order_items_dataset": ['shipping_limit_date'],
    "olist_order_reviews_dataset": ['review_creation_date', 'review_answer_timestamp']
}

# Порядок загрузки (остается без изменений)
LOAD_ORDER = [
    "product_category_name_translation", "olist_customers_dataset", "olist_sellers_dataset",
    "olist_products_dataset", "olist_orders_dataset", "olist_order_items_dataset",
    "olist_order_payments_dataset", "olist_order_reviews_dataset", "olist_geolocation_dataset"
]

# Функция теперь принимает db_engine как аргумент
def create_tables(db_engine):
    logging.info("Создание таблиц (если не существуют)...")
    try:
        Base.metadata.create_all(bind=db_engine) # Используем переданный db_engine
        logging.info("Таблицы успешно созданы или уже существуют.")
    except SQLAlchemyError as e:
        logging.error(f"Ошибка при создании таблиц: {e}")
        raise

# Функция теперь принимает db_engine как аргумент
def load_csv_to_db(filename_no_ext, db_engine):
    if filename_no_ext not in CSV_MODEL_MAP:
        logging.warning(f"Пропуск файла {filename_no_ext}.csv: нет сопоставления с моделью.")
        return

    model, _ = CSV_MODEL_MAP[filename_no_ext]
    table_name = model.__tablename__
    csv_path = os.path.join(DATA_DIR, f"{filename_no_ext}.csv")

    if not os.path.exists(csv_path):
        logging.warning(f"Пропуск файла {filename_no_ext}: CSV файл не найден по пути {csv_path}")
        return

    logging.info(f"Загрузка данных из {filename_no_ext}.csv в таблицу {table_name}...")
    start_time = time.time()

    try:
        parse_dates_cols = DATE_COLUMNS_MAP.get(filename_no_ext, False)
        # Используем переданный db_engine
        with db_engine.connect() as connection:
            chunk_size = 10000
            first_chunk = True
            for chunk_df in pd.read_csv(csv_path,
                                        chunksize=chunk_size,
                                        parse_dates=parse_dates_cols,
                                        infer_datetime_format=True if parse_dates_cols else False):
                # Эта строка удалит существующую таблицу и создаст ее заново при первой загрузке чанка,
                # затем будет добавлять данные. Если вы хотите добавлять данные к существующим,
                # измените 'replace' на 'append' для всех чанков.
                write_mode = 'replace' if first_chunk else 'append'
                chunk_df.to_sql(name=table_name,
                                con=connection,
                                if_exists=write_mode,
                                index=False,
                                chunksize=1000)
                first_chunk = False
                logging.info(f"  Записан чанк размером {len(chunk_df)} строк в {table_name}")

        end_time = time.time()
        logging.info(f"Таблица {table_name} успешно загружена за {end_time - start_time:.2f} секунд.")

    except FileNotFoundError:
        logging.error(f"Ошибка: Файл {csv_path} не найден.")
    except pd.errors.EmptyDataError:
        logging.warning(f"Предупреждение: Файл {csv_path} пуст.")
    except SQLAlchemyError as e:
        logging.error(f"Ошибка SQLAlchemy при загрузке {table_name}: {e}")
    except Exception as e:
        logging.error(f"Неожиданная ошибка при загрузке {table_name} из {filename_no_ext}.csv: {e}")
        import traceback
        traceback.print_exc()

def main():
    logging.info("Запуск скрипта загрузки данных Olist...")

    class ScriptAppSettings:
        # Используем "database.db" в текущей директории (корне проекта) по умолчанию.
        # Если скрипт запускается из C:\Users\alex\Desktop\AnomaLens3.0,
        # то "sqlite:///./database.db" будет указывать на C:\Users\alex\Desktop\AnomaLens3.0\database.db
        # Позволяем переопределить через переменную окружения DATABASE_URL (если она установлена для всего проекта).
        default_db_filename = "database.db"
        DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///./{default_db_filename}")
        DATABASE_URL_SAFE_LOGGING = DATABASE_URL # Для логирования

    app_settings = ScriptAppSettings()

    # Вызываем initialize_database_session и получаем инициализированный engine.
    local_db_engine, _ = initialize_database_session(app_settings)

    if local_db_engine is None:
        logging.error("Критическая ошибка: SQLAlchemy engine не был инициализирован функцией initialize_database_session. "
                      "Проверьте логи от initialize_database_session и корректность DATABASE_URL.")
        return

    # Этот лог покажет, какой URL используется для подключения к БД
    logging.info(f"Скрипт будет использовать SQLAlchemy Engine для URL: {local_db_engine.url}")

    # create_tables создаст таблицы, если их еще нет. Если они есть, ничего не сделает.
    create_tables(local_db_engine)

    logging.info(f"Директория с данными: {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        logging.error(f"Директория с данными не найдена: {DATA_DIR}")
        return

    total_start_time = time.time()

    for filename in LOAD_ORDER:
        load_csv_to_db(filename, local_db_engine) # Передаем local_db_engine

    total_end_time = time.time()
    logging.info(f"Загрузка всех данных завершена за {total_end_time - total_start_time:.2f} секунд.")

if __name__ == "__main__":
    main()

# python -m backend.scripts.load_data