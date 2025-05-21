"""
schemas.py - Определение Pydantic-схем для валидации данных
===========================================================
Этот модуль содержит схемы для валидации входных/выходных данных API.
Схемы разделены на категории и соответствуют моделям базы данных.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field
from typing import List, Optional, TypeVar, Generic, Dict, Any
from datetime import datetime
import json

# =============================================================================
# Основные схемы для работы с бизнес-сущностями
# =============================================================================

# ----- Схемы для Customer (Покупатель) -----

class CustomerBase(BaseModel):
    """Базовая схема для покупателя."""
    customer_id: str
    customer_unique_id: str
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str

class Customer(CustomerBase):
    """Схема для чтения данных покупателя из БД."""
    model_config = ConfigDict(from_attributes=True)


# ----- Схемы для Geolocation (Геолокация) -----

class GeolocationBase(BaseModel):
    """Базовая схема для геолокации."""
    geolocation_zip_code_prefix: int
    geolocation_lat: float
    geolocation_lng: float
    geolocation_city: str
    geolocation_state: str

class GeolocationCreate(GeolocationBase):
    """Схема для создания новой геолокации."""
    pass

class Geolocation(GeolocationBase):
    """Схема для чтения данных геолокации из БД."""
    id: int
    model_config = ConfigDict(from_attributes=True)


# ----- Схемы для Product (Товар) и связанных сущностей -----

class ProductCategoryNameTranslationBase(BaseModel):
    """Базовая схема для перевода названия категории товара."""
    product_category_name: str
    product_category_name_english: str

class ProductCategoryNameTranslation(ProductCategoryNameTranslationBase):
    """Схема для чтения данных перевода категории из БД."""
    model_config = ConfigDict(from_attributes=True)

class ProductBase(BaseModel):
    """Базовая схема для товара."""
    product_id: str
    product_category_name: Optional[str] = None
    product_name_lenght: Optional[int] = None
    product_description_lenght: Optional[int] = None
    product_photos_qty: Optional[int] = None
    product_weight_g: Optional[int] = None
    product_length_cm: Optional[int] = None
    product_height_cm: Optional[int] = None
    product_width_cm: Optional[int] = None

class Product(ProductBase):
    """Схема для чтения данных товара из БД с возможной информацией о категории."""
    # Включаем переведенную категорию при запросе продукта
    category_translation: Optional[ProductCategoryNameTranslation] = None
    model_config = ConfigDict(from_attributes=True)


# ----- Схемы для Seller (Продавец) -----

class SellerBase(BaseModel):
    """Базовая схема для продавца."""
    seller_id: str
    seller_zip_code_prefix: int
    seller_city: str
    seller_state: str

class Seller(SellerBase):
    """Схема для чтения данных продавца из БД."""
    model_config = ConfigDict(from_attributes=True)


# ----- Схемы для Order (Заказ) и связанных сущностей -----

class OrderItemBase(BaseModel):
    """Базовая схема для товарной позиции заказа."""
    order_id: str
    order_item_id: int
    product_id: str
    seller_id: str
    shipping_limit_date: datetime
    price: float
    freight_value: float

class OrderItemCreate(OrderItemBase):
    """Схема для создания новой товарной позиции."""
    pass

class OrderItem(OrderItemBase):
    """Схема для чтения данных товарной позиции из БД."""
    model_config = ConfigDict(from_attributes=True)


class OrderPaymentBase(BaseModel):
    """Базовая схема для платежа по заказу."""
    order_id: str
    payment_sequential: int
    payment_type: str
    payment_installments: int
    payment_value: float

class OrderPayment(OrderPaymentBase):
    """Схема для чтения данных платежа из БД."""
    model_config = ConfigDict(from_attributes=True)


class OrderReviewBase(BaseModel):
    """Базовая схема для отзыва о заказе."""
    review_id: str
    order_id: str
    review_score: int
    review_comment_title: Optional[str] = None
    review_comment_message: Optional[str] = None
    review_creation_date: datetime
    review_answer_timestamp: datetime

class OrderReview(OrderReviewBase):
    """Схема для чтения данных отзыва из БД."""
    model_config = ConfigDict(from_attributes=True)


class OrderBase(BaseModel):
    """Базовая схема для заказа."""
    order_id: str
    customer_id: str
    order_status: str
    order_purchase_timestamp: datetime
    order_approved_at: Optional[datetime] = None
    order_delivered_carrier_date: Optional[datetime] = None
    order_delivered_customer_date: Optional[datetime] = None
    order_estimated_delivery_date: datetime

class OrderCreate(OrderBase):
    """Схема для создания нового заказа."""
    pass

class Order(OrderBase):
    """
    Схема для чтения данных заказа из БД, 
    включая связанные объекты (покупатель, товары, платежи, отзывы).
    """
    customer: Optional[Customer] = None
    items: List[OrderItem] = []
    payments: List[OrderPayment] = []
    reviews: List[OrderReview] = []
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Схемы для аналитических функций и обнаружения аномалий
# =============================================================================

class AnomalyBase(BaseModel):
    """Базовая схема для обнаруженной аномалии."""
    order_item_id: int
    order_id: str
    detection_date: datetime
    anomaly_score: Optional[float] = None
    detector_type: str
    details: Optional[str] = None  # JSON-строка с деталями аномалии

class AnomalyCreate(AnomalyBase):
    """Схема для создания новой записи об аномалии."""
    pass

class Anomaly(AnomalyBase):
    """
    Схема для чтения данных аномалии из БД.
    Включает автоматическое преобразование JSON-строки details в словарь.
    """
    id: int
    
    @computed_field
    @property
    def details_dict(self) -> Optional[Dict[str, Any]]:
        if isinstance(self.details, str):
            try:
                return json.loads(self.details)
            except json.JSONDecodeError:
                # Можно логировать ошибку здесь, если есть доступ к логгеру
                return {"error": "Failed to parse details JSON"}
        elif isinstance(self.details, dict): # Если details уже словарь
            return self.details
        return None
        
    model_config = ConfigDict(from_attributes=True)


class AnomalyDetectionResult(BaseModel):
    """Схема для ответа API после запуска задачи обнаружения аномалий."""
    task_id: str          # Идентификатор фоновой задачи
    status: str           # Статус выполнения (например, "pending", "completed")
    message: str          # Информационное сообщение


class DetectionRequest(BaseModel):
    """Схема для запроса на обнаружение аномалий."""
    detectors: List[str] = ['statistical', 'isolation_forest']  # Используемые детекторы
    start_date: Optional[datetime] = None                       # Начальная дата периода
    end_date: Optional[datetime] = None                         # Конечная дата периода
    entity_type: Optional[str] = None                           # Тип сущности для анализа


# =============================================================================
# Схемы для настроек приложения
# =============================================================================

class SettingBase(BaseModel):
    """Базовая схема для настройки приложения."""
    key: str
    value: str

class SettingCreate(SettingBase):
    """Схема для создания новой настройки."""
    pass

class SettingUpdate(BaseModel):
    """Схема для обновления значения настройки."""
    value: str

class Setting(SettingBase):
    """Схема для чтения настройки из БД."""
    model_config = ConfigDict(from_attributes=True)


class SettingsBase(BaseModel):
    """Расширенная схема настройки с дополнительной информацией."""
    setting_key: str = Field(..., max_length=100)
    setting_value: str
    description: Optional[str] = None

class SettingsCreate(SettingsBase):
    """Схема для создания расширенной настройки."""
    pass

class Settings(SettingsBase):
    """Схема для чтения расширенной настройки из БД."""
    id: int
    last_updated: datetime
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Служебные схемы для API и фоновых задач
# =============================================================================

ItemSchema = TypeVar('ItemSchema')

class PaginatedResponse(BaseModel, Generic[ItemSchema]):
    """
    Обобщенная схема для пагинированного ответа API.
    Может использоваться с любым типом элементов.
    """
    total: int                # Общее количество элементов
    items: List[ItemSchema]   # Список элементов на текущей странице


class TaskStatusResult(BaseModel):
    """Схема для получения статуса выполнения фоновой задачи."""
    status: str                       # Статус задачи
    start_time: datetime              # Время начала выполнения
    end_time: Optional[datetime] = None  # Время завершения (если завершена)
    details: str                      # Детали выполнения
    result: Optional[Dict[str, Any]] = None  # Результат выполнения 