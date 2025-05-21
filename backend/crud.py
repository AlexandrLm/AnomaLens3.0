"""
crud.py - Функции для работы с базой данных
===========================================
Этот модуль содержит функции для выполнения операций CRUD
(Create, Read, Update, Delete) над моделями SQLAlchemy.
"""

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, desc, delete
from typing import List, Optional, Type, Dict, Any, Union
from datetime import datetime
import json

from . import models, schemas

# =============================================================================
# Функции для работы с заказами и связанными данными
# =============================================================================

def get_orders_with_details(
    db: Session, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None, 
    skip: int = 0, 
    limit: Optional[int] = None
) -> List[models.Order]:
    """
    Получает список заказов с предзагрузкой связанных данных.
    
    Args:
        db: Сессия SQLAlchemy для работы с БД
        start_date: Начальная дата для фильтрации (включительно)
        end_date: Конечная дата для фильтрации (не включительно)
        skip: Количество записей для пропуска (для пагинации)
        limit: Максимальное количество записей для возврата
        
    Returns:
        Список объектов Order со всеми связанными данными
    """
    query = (
        select(models.Order)
        .options(
            joinedload(models.Order.items).options(
                joinedload(models.OrderItem.product).options(
                    joinedload(models.Product.category_translation)
                ),
                joinedload(models.OrderItem.seller)
            ),
            joinedload(models.Order.payments),
            joinedload(models.Order.reviews),
            joinedload(models.Order.customer)
        )
        .order_by(desc(models.Order.order_purchase_timestamp))
    )

    if start_date:
        query = query.where(models.Order.order_purchase_timestamp >= start_date)
    if end_date:
        query = query.where(models.Order.order_purchase_timestamp < end_date)
    
    # Применяем пагинацию только если limit указан
    if limit is not None: # Изменено для корректной обработки отсутствия limit
        query = query.offset(skip).limit(limit)
    else:
        query = query.offset(skip) # Если limit не указан, offset все еще может быть применен

    result = db.execute(query)
    return result.unique().scalars().all() 


def get_order_by_id_with_details(db: Session, order_id: str) -> Optional[models.Order]:
    """
    Получает один заказ по его ID с предзагрузкой всех связанных данных.
    
    Args:
        db: Сессия SQLAlchemy для работы с БД
        order_id: Идентификатор заказа
        
    Returns:
        Объект Order со всеми связанными данными или None, если заказ не найден
    """
    query = (
        select(models.Order)
        .where(models.Order.order_id == order_id)  # Фильтруем по ID заказа
        .options(
            # Предзагружаем товарные позиции и их связанные объекты
            joinedload(models.Order.items).options(
                joinedload(models.OrderItem.product).options(
                    joinedload(models.Product.category_translation)
                ),
                joinedload(models.OrderItem.seller)
            ),
            # Предзагружаем остальные связанные объекты
            joinedload(models.Order.payments),
            joinedload(models.Order.reviews),
            joinedload(models.Order.customer)
        )
    )
    
    result = db.execute(query)
    # Используем .unique() для избежания дублирования объектов из-за JOIN
    # и .first() для получения одного объекта или None
    return result.unique().scalars().first()


def get_order_count(
    db: Session, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> int:
    """
    Получает общее количество заказов с возможностью фильтрации по дате.
    
    Args:
        db: Сессия SQLAlchemy для работы с БД
        start_date: Начальная дата для фильтрации (включительно)
        end_date: Конечная дата для фильтрации (не включительно)
        
    Returns:
        Количество заказов, соответствующих фильтрам
    """
    query = select(func.count(models.Order.order_id))
    
    if start_date:
        query = query.where(models.Order.order_purchase_timestamp >= start_date)
    if end_date:
        query = query.where(models.Order.order_purchase_timestamp < end_date)
    
    count = db.execute(query).scalar_one_or_none() or 0
    return count

# =============================================================================
# Общие функции CRUD для любых моделей
# =============================================================================

def get_items(db: Session, model: Type[Any], skip: int = 0, limit: int = 100) -> List[Any]:
    """
    Обобщенная функция для получения списка записей из любой таблицы.
    """
    result = db.execute(select(model).offset(skip).limit(limit))
    return result.scalars().all()


def get_item_count(db: Session, model: Type[Any]) -> int:
    """
    Обобщенная функция для получения количества записей в таблице.
    """
    count = db.execute(select(func.count()).select_from(model)).scalar_one_or_none() or 0
    return count


def get_item_by_id(db: Session, model: Type[Any], item_id: Union[str, int]) -> Optional[Any]:
    """
    Обобщенная функция для получения одной записи по ID.
    Работает с моделями, у которых первичный ключ - одна колонка.
    """
    pk_column = model.__mapper__.primary_key[0]
    result = db.execute(select(model).where(pk_column == item_id))
    return result.scalars().first()


# =============================================================================
# Функции для получения связанных с заказом данных (специфичные, могут быть полезны)
# =============================================================================

# Удаляем get_items_for_order, get_payments_for_order, get_reviews_for_order, get_order_items_for_period
# т.к. они не используются и данные загружаются через relationships в get_orders_with_details/get_order_by_id_with_details

# =============================================================================
# Функции CRUD для Аномалий
# =============================================================================

def create_anomaly(db: Session, anomaly: schemas.AnomalyCreate) -> models.Anomaly:
    """
    Создает новую запись об аномалии в БД.
    """
    details_str = None
    if anomaly.details is not None:
        if isinstance(anomaly.details, dict) or isinstance(anomaly.details, list): # Добавил list для гибкости
            details_str = json.dumps(anomaly.details)
        elif isinstance(anomaly.details, str):
            details_str = anomaly.details
        else:
            details_str = str(anomaly.details)
    
    anomaly_data = anomaly.model_dump(exclude={'details'})
    anomaly_data['details'] = details_str
    
    db_anomaly = models.Anomaly(**anomaly_data)
    db.add(db_anomaly)
    db.commit()
    db.refresh(db_anomaly)
    
    return db_anomaly


def get_anomalies(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    detector_type: Optional[str] = None
) -> List[models.Anomaly]:
    """
    Получает список обнаруженных аномалий с применением различных фильтров.
    """
    query = select(models.Anomaly)
    
    if start_date:
        query = query.where(models.Anomaly.detection_date >= start_date)
    if end_date:
        query = query.where(models.Anomaly.detection_date < end_date)
    
    if min_score is not None:
        query = query.where(models.Anomaly.anomaly_score.isnot(None))
        query = query.where(models.Anomaly.anomaly_score >= min_score)
    if max_score is not None:
        # Не нужно isnot(None), если min_score уже установлен
        # Если только max_score, то isnot(None) здесь нужен
        if min_score is None:
             query = query.where(models.Anomaly.anomaly_score.isnot(None))
        query = query.where(models.Anomaly.anomaly_score <= max_score)
    
    if detector_type:
        query = query.where(models.Anomaly.detector_type == detector_type)
    
    query = query.order_by(desc(models.Anomaly.detection_date)).offset(skip).limit(limit)
    
    result = db.execute(query)
    return result.scalars().all()


def get_anomaly(db: Session, anomaly_id: int) -> Optional[models.Anomaly]:
    """
    Получает одну аномалию по ее ID.
    (Использует старый стиль SQLAlchemy, можно обновить)
    """
    # Обновление до нового стиля:
    return db.execute(select(models.Anomaly).where(models.Anomaly.id == anomaly_id)).scalars().first()
    # Старый стиль:
    # return db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()


def delete_anomaly(db: Session, anomaly_id: int) -> Optional[models.Anomaly]:
    """Удаляет одну аномалию по ее ID."""
    # db_anomaly = db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()
    # Новый стиль:
    stmt = select(models.Anomaly).where(models.Anomaly.id == anomaly_id)
    db_anomaly = db.execute(stmt).scalars().first()
    
    if db_anomaly:
        db.delete(db_anomaly)
        db.commit()
    return db_anomaly


def delete_all_anomalies(db: Session) -> int:
    """
    Удаляет ВСЕ аномалии из таблицы Anomaly.
    Возвращает количество удаленных записей.
    ВНИМАНИЕ: Эта операция необратима!
    """
    # Старый стиль (для справки):
    # num_rows_deleted = db.query(models.Anomaly).delete()
    # db.commit()
    # return num_rows_deleted
    
    # Новый стиль SQLAlchemy 2.0:
    stmt = delete(models.Anomaly)
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


# =============================================================================
# Функции CRUD для Настроек
# =============================================================================

def create_setting(db: Session, setting: schemas.SettingCreate) -> models.Setting:
    """
    Создает новую настройку.
    """
    db_setting = models.Setting(**setting.model_dump())
    db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting


def get_settings(db: Session, skip: int = 0, limit: int = 100) -> List[models.Setting]:
    """
    Получает список всех настроек с пагинацией.
    """
    result = db.execute(select(models.Setting).offset(skip).limit(limit))
    return result.scalars().all()


def get_setting_by_key(db: Session, key: str) -> Optional[models.Setting]:
    """
    Получает настройку по ключу.
    """
    result = db.execute(select(models.Setting).where(models.Setting.key == key))
    return result.scalars().first()


def update_setting(db: Session, key: str, setting_update: schemas.SettingUpdate) -> Optional[models.Setting]:
    """
    Обновляет значение настройки по ключу.
    """
    db_setting = get_setting_by_key(db, key)
    if db_setting is None:
        return None
    
    db_setting.value = setting_update.value
    db.commit()
    db.refresh(db_setting)
    return db_setting


def delete_setting(db: Session, key: str) -> Optional[models.Setting]:
    """
    Удаляет настройку по ключу.
    """
    db_setting = get_setting_by_key(db, key)
    if db_setting is None:
        return None
    
    db.delete(db_setting)
    db.commit()
    return db_setting