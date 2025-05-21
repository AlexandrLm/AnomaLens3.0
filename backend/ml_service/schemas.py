"""
schemas.py - Pydantic-схемы для ML-сервиса обнаружения аномалий
==============================================================
Этот модуль содержит определения Pydantic-схем, используемых в API ML-сервиса
и для внутреннего взаимодействия между компонентами системы.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, TypeVar, Generic, Dict, Any
import datetime

# =============================================================================
# Основные схемы для аномалий
# =============================================================================

class AnomalyBase(BaseModel):
    """Базовая схема для информации об аномалии."""
    order_id: str 
    order_item_id: Optional[int] = None
    detector_type: str
    anomaly_score: Optional[float] = None
    details: Optional[str] = None
    detection_date: datetime.datetime

class Anomaly(AnomalyBase):
    """Схема для чтения данных аномалии из БД."""
    id: int
    model_config = ConfigDict(from_attributes=True)

class AnomalyCreate(AnomalyBase):
    """Схема для создания новой записи об аномалии."""
    pass
    
class AnomalyDetectionResult(BaseModel):
    """Схема для результата запуска задачи обнаружения аномалий."""
    task_id: str
    status: str
    message: str

# =============================================================================
# Вспомогательные схемы для сущностей e-commerce
# =============================================================================

class CustomerBase(BaseModel):
    """Схема для информации о покупателе."""
    customer_id: str
    customer_unique_id: str
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str

class Customer(CustomerBase):
    """Схема для чтения данных покупателя из БД."""
    model_config = ConfigDict(from_attributes=True)

class SellerBase(BaseModel):
    """Схема для информации о продавце."""
    seller_id: str
    seller_zip_code_prefix: int
    seller_city: str
    seller_state: str

class Seller(SellerBase):
    """Схема для чтения данных продавца из БД."""
    model_config = ConfigDict(from_attributes=True)

class ProductBase(BaseModel):
    """Схема для информации о товаре."""
    product_id: str
    product_category_name: Optional[str] = None
    product_name_lenght: Optional[float] = None # Сохраняем оригинальные имена из БД
    product_description_lenght: Optional[float] = None
    product_photos_qty: Optional[float] = None
    product_weight_g: Optional[float] = None
    product_length_cm: Optional[float] = None
    product_height_cm: Optional[float] = None
    product_width_cm: Optional[float] = None

class Product(ProductBase):
    """Схема для чтения данных товара из БД."""
    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Схемы для заказов и связанных сущностей
# =============================================================================

class OrderItemBase(BaseModel):
    """Схема для информации о товарной позиции в заказе."""
    price: float
    freight_value: float

class OrderItemWithProductSeller(OrderItemBase):
    """Расширенная схема товарной позиции со связанными товаром и продавцом."""
    order_item_id: int
    product: Product
    seller: Seller
    model_config = ConfigDict(from_attributes=True)

class OrderPaymentBase(BaseModel):
    """Схема для информации о платеже по заказу."""
    payment_sequential: int
    payment_type: str
    payment_installments: int
    payment_value: float

class OrderPayment(OrderPaymentBase):
    """Схема для чтения данных платежа из БД."""
    model_config = ConfigDict(from_attributes=True)

class OrderReviewBase(BaseModel):
    """Схема для информации об отзыве о заказе."""
    review_score: int
    review_comment_title: Optional[str] = None
    review_comment_message: Optional[str] = None
    review_creation_date: datetime.datetime
    review_answer_timestamp: datetime.datetime

class OrderReview(OrderReviewBase):
    """Схема для чтения данных отзыва из БД."""
    model_config = ConfigDict(from_attributes=True)

class OrderBase(BaseModel):
    """Схема для информации о заказе."""
    order_id: str
    order_status: str
    order_purchase_timestamp: datetime.datetime
    order_approved_at: Optional[datetime.datetime] = None
    order_delivered_carrier_date: Optional[datetime.datetime] = None
    order_delivered_customer_date: Optional[datetime.datetime] = None
    order_estimated_delivery_date: datetime.datetime

class OrderWithDetails(OrderBase):
    """Расширенная схема заказа со всеми связанными данными."""
    customer: Customer
    items: List[OrderItemWithProductSeller]
    payments: List[OrderPayment]
    reviews: List[OrderReview]
    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Контекстные схемы для аномалий
# =============================================================================

class AnomalyContextResponse(BaseModel):
    """Схема для информации о контексте аномалии (заказ и его детали)."""
    order_details: OrderWithDetails

# =============================================================================
# Общие схемы для пагинации и геолокации
# =============================================================================

DataType = TypeVar('DataType')

class PaginatedResponse(BaseModel, Generic[DataType]):
    """Обобщенная схема для пагинированных ответов API."""
    total: int
    items: List[DataType]
    
class Geolocation(BaseModel):
    """Схема для информации о геолокации."""
    geolocation_zip_code_prefix: int = Field(..., primary_key=True)
    geolocation_lat: float
    geolocation_lng: float
    geolocation_city: str = Field(..., primary_key=True)
    geolocation_state: str = Field(..., primary_key=True)
    model_config = ConfigDict(from_attributes=True)