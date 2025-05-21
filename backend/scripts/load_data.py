import os
import pandas as pd
import logging
import time
from sqlalchemy.exc import SQLAlchemyError
from ..database import engine, Base, SessionLocal # Импорты из родительской директории
from .. import models # Импортируем все модели

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к директории с данными относительно ЭТОГО скрипта
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

# Сопоставление имен файлов CSV с моделями SQLAlchemy и их первичными ключами
# (имена файлов без расширения .csv)
# Первичные ключи нужны для правильной обработки replace/append и потенциальной дедупликации
# Для таблиц с составными ключами указываем кортеж
CSV_MODEL_MAP = {
    "olist_customers_dataset": (models.Customer, 'customer_id'),
    "olist_geolocation_dataset": (models.Geolocation, ('geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng')), # Составной PK для примера
    "olist_orders_dataset": (models.Order, 'order_id'),
    "olist_order_items_dataset": (models.OrderItem, ('order_id', 'order_item_id')),
    "olist_order_payments_dataset": (models.OrderPayment, ('order_id', 'payment_sequential')),
    "olist_order_reviews_dataset": (models.OrderReview, 'review_id'),
    "olist_products_dataset": (models.Product, 'product_id'),
    "olist_sellers_dataset": (models.Seller, 'seller_id'),
    "product_category_name_translation": (models.ProductCategoryNameTranslation, 'product_category_name')
}

# Колонки с датами для парсинга в Pandas
# Ключи - имена файлов CSV
DATE_COLUMNS_MAP = {
    "olist_orders_dataset": [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ],
    "olist_order_items_dataset": [
        'shipping_limit_date'
    ],
    "olist_order_reviews_dataset": [
        'review_creation_date',
        'review_answer_timestamp'
    ]
}

# Порядок загрузки важен из-за внешних ключей
# Сначала таблицы без FK или те, на которые ссылаются другие
LOAD_ORDER = [
    "product_category_name_translation",
    "olist_customers_dataset",
    "olist_sellers_dataset",
    "olist_products_dataset",
    "olist_orders_dataset", # Зависит от customers
    "olist_order_items_dataset", # Зависит от orders, products, sellers
    "olist_order_payments_dataset", # Зависит от orders
    "olist_order_reviews_dataset", # Зависит от orders
    "olist_geolocation_dataset" # Независимая, но большая, оставим напоследок
]

def create_tables():
    logging.info("Создание таблиц (если не существуют)...")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Таблицы успешно созданы или уже существуют.")
    except SQLAlchemyError as e:
        logging.error(f"Ошибка при создании таблиц: {e}")
        raise

def load_csv_to_db(filename_no_ext):
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
        # Получаем список колонок с датами для парсинга
        parse_dates_cols = DATE_COLUMNS_MAP.get(filename_no_ext, False) # False если нет дат

        with engine.connect() as connection:
            chunk_size = 10000 # Размер чанка для больших файлов
            first_chunk = True
            # Используем parse_dates вместо dtype для дат
            for chunk_df in pd.read_csv(csv_path,
                                        chunksize=chunk_size,
                                        parse_dates=parse_dates_cols,
                                        infer_datetime_format=True if parse_dates_cols else False): # infer_datetime_format только если есть даты

                # Дополнительная проверка/обработка NaT после парсинга, если нужно
                # (Например, SQLAlchemy может не любить NaT в non-nullable колонках)
                # if parse_dates_cols:
                #     for col in parse_dates_cols:
                #         if col in chunk_df.columns:
                #             # Заменить NaT на None, если колонка nullable в модели
                #             # if model.__table__.columns[col].nullable:
                #             #     chunk_df[col] = chunk_df[col].replace({pd.NaT: None})
                #             pass # Пока оставим как есть, SQLite должен справиться с NaT

                write_mode = 'replace' if first_chunk else 'append'

                chunk_df.to_sql(name=table_name,
                                con=connection,
                                if_exists=write_mode,
                                index=False,
                                chunksize=1000
                               )
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
    create_tables()

    logging.info(f"Директория с данными: {DATA_DIR}")

    total_start_time = time.time()

    for filename in LOAD_ORDER:
        load_csv_to_db(filename)

    total_end_time = time.time()
    logging.info(f"Загрузка всех данных завершена за {total_end_time - total_start_time:.2f} секунд.")

if __name__ == "__main__":
    main() 