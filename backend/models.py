"""
models.py - Модели SQLAlchemy для работы с базой данных
======================================================
Этот модуль содержит определения всех моделей данных, используемых в приложении.
Модели соответствуют таблицам в базе данных и описывают их структуру.
"""

from sqlalchemy import (Column, Integer, String, Float, DateTime, ForeignKey,
                        PrimaryKeyConstraint)
from sqlalchemy.orm import relationship
from .database import Base # Импортируем Base из database.py

# Важно: Имена таблиц здесь даны в единственном числе (по конвенции или для простоты),
# но можно использовать и множественное, если удобнее.

# ------------------------------------------------------------------------
# Основные модели (сущности) для электронной коммерции
# ------------------------------------------------------------------------

class Customer(Base):
    """
    Модель покупателя.
    Содержит основную информацию о клиентах и их местоположении.
    """
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True, index=True)
    customer_unique_id = Column(String, index=True)  # Уникальный идентификатор клиента
    customer_zip_code_prefix = Column(Integer)       # Почтовый индекс
    customer_city = Column(String)                   # Город
    customer_state = Column(String(2))               # Штат/регион (2-символьный код)

    orders = relationship("Order", back_populates="customer")

class Geolocation(Base):
    """
    Модель геолокации.
    Хранит координаты и информацию о местоположении для почтовых индексов.
    """
    __tablename__ = "geolocation"
    # Составной первичный ключ для уникальной идентификации геолокации
    __table_args__ = (PrimaryKeyConstraint('geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'),)

    geolocation_zip_code_prefix = Column(Integer, index=True)
    geolocation_lat = Column(Float)                     # Широта
    geolocation_lng = Column(Float)                     # Долгота
    geolocation_city = Column(String)                   # Город
    geolocation_state = Column(String(2))               # Штат/регион (2-символьный код)

class Order(Base):
    """
    Модель заказа.
    Содержит информацию о заказе, его статусе и временных метках.
    """
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), index=True)
    order_status = Column(String)                             # Статус заказа
    order_purchase_timestamp = Column(DateTime)               # Дата и время покупки
    order_approved_at = Column(DateTime, nullable=True)       # Дата и время одобрения
    order_delivered_carrier_date = Column(DateTime, nullable=True)  # Дата передачи перевозчику
    order_delivered_customer_date = Column(DateTime, nullable=True) # Дата доставки клиенту
    order_estimated_delivery_date = Column(DateTime)          # Ожидаемая дата доставки

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
    payments = relationship("OrderPayment", back_populates="order")
    reviews = relationship("OrderReview", back_populates="order")

class OrderItem(Base):
    """
    Модель товарной позиции в заказе.
    Связывает заказ с конкретными товарами и продавцами.
    """
    __tablename__ = "order_items"
    # Составной первичный ключ: order_id + order_item_id
    __table_args__ = (PrimaryKeyConstraint('order_id', 'order_item_id'),)
    
    order_id = Column(String, ForeignKey("orders.order_id"), primary_key=True)  # Часть составного ключа
    order_item_id = Column(Integer, primary_key=True)         # Порядковый номер товара в заказе
    product_id = Column(String, ForeignKey("products.product_id"), index=True)
    seller_id = Column(String, ForeignKey("sellers.seller_id"), index=True)
    shipping_limit_date = Column(DateTime)                    # Крайний срок отправки
    price = Column(Float)                                     # Цена
    freight_value = Column(Float)                             # Стоимость доставки

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")
    seller = relationship("Seller", back_populates="order_items")

class OrderPayment(Base):
    """
    Модель платежа по заказу.
    Содержит информацию о способе оплаты и сумме.
    """
    __tablename__ = "order_payments"
    __table_args__ = (PrimaryKeyConstraint('order_id', 'payment_sequential'),)

    order_id = Column(String, ForeignKey("orders.order_id"), index=True)
    payment_sequential = Column(Integer)                     # Порядковый номер платежа
    payment_type = Column(String)                            # Тип платежа
    payment_installments = Column(Integer)                   # Количество платежей/взносов
    payment_value = Column(Float)                            # Сумма платежа

    order = relationship("Order", back_populates="payments")

class OrderReview(Base):
    """
    Модель отзыва о заказе.
    Хранит информацию об оценке и комментариях покупателя.
    """
    __tablename__ = "order_reviews"

    review_id = Column(String, primary_key=True, index=True)
    order_id = Column(String, ForeignKey("orders.order_id"), index=True)
    review_score = Column(Integer)                           # Оценка (обычно 1-5)
    review_comment_title = Column(String, nullable=True)     # Заголовок отзыва
    review_comment_message = Column(String, nullable=True)   # Текст отзыва
    review_creation_date = Column(DateTime)                  # Дата создания отзыва
    review_answer_timestamp = Column(DateTime)               # Дата ответа на отзыв

    order = relationship("Order", back_populates="reviews")

class Product(Base):
    """
    Модель товара.
    Содержит характеристики товара и его категорию.
    """
    __tablename__ = "products"

    product_id = Column(String, primary_key=True, index=True)
    product_category_name = Column(
        String, 
        ForeignKey("product_category_name_translation.product_category_name"), 
        nullable=True, 
        index=True
    )
    product_name_lenght = Column(Integer, nullable=True)     # Длина названия товара
    product_description_lenght = Column(Integer, nullable=True)  # Длина описания товара
    product_photos_qty = Column(Integer, nullable=True)      # Количество фотографий
    product_weight_g = Column(Integer, nullable=True)        # Вес в граммах
    product_length_cm = Column(Integer, nullable=True)       # Длина в см
    product_height_cm = Column(Integer, nullable=True)       # Высота в см
    product_width_cm = Column(Integer, nullable=True)        # Ширина в см

    order_items = relationship("OrderItem", back_populates="product")
    category_translation = relationship("ProductCategoryNameTranslation", back_populates="products")

class Seller(Base):
    """
    Модель продавца.
    Содержит основную информацию о продавцах и их местоположении.
    """
    __tablename__ = "sellers"

    seller_id = Column(String, primary_key=True, index=True)
    seller_zip_code_prefix = Column(Integer)                 # Почтовый индекс
    seller_city = Column(String)                             # Город
    seller_state = Column(String(2))                         # Штат/регион (2-символьный код)

    order_items = relationship("OrderItem", back_populates="seller")

class ProductCategoryNameTranslation(Base):
    """
    Модель перевода названий категорий товаров.
    Обеспечивает многоязычную поддержку категорий.
    """
    __tablename__ = "product_category_name_translation"

    product_category_name = Column(String, primary_key=True, index=True)  # Оригинальное название категории
    product_category_name_english = Column(String)                        # Название на английском

    products = relationship("Product", back_populates="category_translation")

# ------------------------------------------------------------------------
# Модели для аналитики и настроек приложения
# ------------------------------------------------------------------------

class Anomaly(Base):
    """
    Модель обнаруженной аномалии.
    Хранит результаты работы детекторов аномалий для товарных позиций.
    """
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)        # Уникальный ID аномалии
    # Ссылки на OrderItem без внешнего ключа
    order_id = Column(String, index=True)                     # ID заказа
    order_item_id = Column(Integer, index=True)               # ID товарной позиции
    
    detection_date = Column(DateTime, index=True)             # Дата обнаружения аномалии
    anomaly_score = Column(Float, nullable=True)              # Оценка аномальности
    detector_type = Column(String, index=True)                # Тип детектора аномалий
    details = Column(String, nullable=True)                   # Дополнительные детали в формате JSON

class Setting(Base):
    """
    Модель настройки приложения.
    Хранит пары ключ-значение для конфигурации системы.
    """
    __tablename__ = "settings"

    key = Column(String, primary_key=True, index=True)       # Ключ настройки
    value = Column(String)                                   # Значение настройки (строка)