"""
common.py - Общие функции и утилиты для ML-сервиса
==================================================
Модуль содержит общие функции, используемые различными детекторами аномалий,
включая загрузку данных из БД и инженерию признаков.
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging

# Настраиваем логирование
logger = logging.getLogger(__name__)

# Импортируем модели
# Исправляем импорт: используем .. для перехода на уровень выше
from .. import models

# =============================================================================
# Константы для инженерии признаков
# =============================================================================

# Списки признаков по типам (используются для документации и могут применяться в коде)
CATEGORICAL_FEATURES = ['product_category_name_english', 'seller_id', 'payment_type']
NUMERICAL_FEATURES = ['price', 'freight_value', 'price_deviation_from_category_mean']

# =============================================================================
# Функции для валидации данных
# =============================================================================

def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Проверяет наличие необходимых колонок в DataFrame.
    
    Args:
        data: DataFrame для проверки
        required_columns: Список необходимых колонок
        
    Returns:
        True если все колонки присутствуют, иначе False
        
    Raises:
        ValueError: Если отсутствуют обязательные колонки
    """
    if data.empty:
        raise ValueError("Пустой DataFrame")
        
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
    
    return True

# =============================================================================
# Функции для загрузки данных
# =============================================================================

def load_data_from_db(
    db: Session, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None, 
    load_associations: bool = False
) -> pd.DataFrame:
    """
    Загружает данные о товарных позициях из базы данных с возможностью присоединения связанных таблиц.
    
    Args:
        db: Сессия SQLAlchemy для работы с БД
        start_date: Начальная дата периода для фильтрации (включительно)
        end_date: Конечная дата периода для фильтрации (не включительно)
        load_associations: Флаг для загрузки связанных таблиц (Order, Product, Customer и т.д.)
        
    Returns:
        DataFrame с данными о товарных позициях и возможными связанными данными
    """
    logger.info(f"Загрузка данных из БД за период: {start_date} - {end_date}"
          f"{' (с ассоциациями)' if load_associations else ''}")
    
    # ----- Определяем базовый набор полей из OrderItem -----
    select_fields = [
        models.OrderItem.order_item_id,
        models.OrderItem.order_id,
        models.OrderItem.product_id,
        models.OrderItem.seller_id,
        models.OrderItem.price,
        models.OrderItem.freight_value,
    ]
    
    # ----- Формируем запрос в зависимости от флага load_associations -----
    if load_associations:
        # Добавляем поля из связанных таблиц
        select_fields.extend([
            models.Order.order_purchase_timestamp,
            models.Order.customer_id,
            models.Product.product_category_name,
            models.Product.product_name_lenght,
            models.Product.product_description_lenght,
            models.Product.product_photos_qty,
            models.Product.product_weight_g,
            models.Product.product_length_cm,
            models.Product.product_height_cm,
            models.Product.product_width_cm,
            models.ProductCategoryNameTranslation.product_category_name_english,
            models.Customer.customer_state,
            models.Seller.seller_state,
            models.OrderPayment.payment_type,
            models.OrderPayment.payment_installments,
            models.OrderPayment.payment_value,
            models.OrderReview.review_score 
        ])
        
        # Создаем запрос с необходимыми JOIN
        query_final = select(*select_fields).select_from(models.OrderItem)\
            .join(models.Order, models.OrderItem.order_id == models.Order.order_id)\
            .join(models.Customer, models.Order.customer_id == models.Customer.customer_id)\
            .join(models.Product, models.OrderItem.product_id == models.Product.product_id)\
            .outerjoin(models.ProductCategoryNameTranslation,
                     models.Product.product_category_name == models.ProductCategoryNameTranslation.product_category_name)\
            .join(models.Seller, models.OrderItem.seller_id == models.Seller.seller_id)\
            .outerjoin(models.OrderPayment, models.Order.order_id == models.OrderPayment.order_id)\
            .outerjoin(models.OrderReview, models.Order.order_id == models.OrderReview.order_id)
    else:
        # Если ассоциации не нужны, делаем минимальный JOIN только для дат
        select_fields.append(models.Order.order_purchase_timestamp)
        query_final = select(*select_fields).select_from(models.OrderItem)\
            .join(models.Order, models.OrderItem.order_id == models.Order.order_id)
    
    # ----- Применяем фильтры по дате -----
    if start_date:
        query_final = query_final.where(models.Order.order_purchase_timestamp >= start_date)
    if end_date:
        query_final = query_final.where(models.Order.order_purchase_timestamp < end_date)

    # ----- Выполняем запрос и обрабатываем возможные ошибки -----
    try:
        df = pd.read_sql(query_final, db.bind)
        logger.info(f"Загружено {len(df)} записей OrderItem.")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из БД: {repr(e)}")
        logger.debug(f"Подробности ошибки: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame в случае ошибки
            
    # Проверяем результат
    if df.empty:
        logger.warning("Загружен пустой DataFrame.")
        
    return df

# =============================================================================
# Функции инженерии признаков
# =============================================================================

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные признаки к DataFrame с данными OrderItem.
    
    Текущие признаки:
    - price_deviation_from_category_mean: Отклонение цены от средней по категории
    
    Args:
        data: DataFrame с исходными данными
        
    Returns:
        DataFrame с добавленными признаками
    """
    # ----- Проверка входных данных -----
    if data.empty:
        logger.warning("engineer_features: Входной DataFrame пуст.")
        return data
        
    required_cols = ['price', 'product_category_name_english']
    
    try:
        validate_data(data, required_cols)
    except ValueError as e:
        logger.warning(f"engineer_features: {e}. Вычисление признаков невозможно.")
        # Добавляем пустые колонки для совместимости с последующими шагами
        for col in ['price_deviation_from_category_mean']:
            if col not in data.columns:
                data[col] = np.nan
        return data

    # ----- Создаем копию для обработки -----
    features_df = data.copy()
    category_col = 'product_category_name_english'
    price_col = 'price'
    freight_col = 'freight_value' # Добавим переменную для удобства
    
    # ----- 1. Рассчитываем средние цены по категориям -----
    # Используем только не-NaN значения для расчета среднего
    category_avg_price = features_df.dropna(subset=[category_col, price_col])\
                              .groupby(category_col)[price_col].mean()
                                  
    if category_avg_price.empty:
        logger.warning("engineer_features: Не удалось рассчитать средние цены по категориям "
              "(нет данных или категорий).")
        features_df['price_deviation_from_category_mean'] = np.nan
        # return features_df # Не выходим, пробуем вычислить другие признаки
    else:
        # ----- 2. Присоединяем средние цены к основному DataFrame -----
        features_df = pd.merge(
            features_df,
            category_avg_price.rename('category_avg_price'),
            left_on=category_col,
            right_index=True,
            how='left'
        )
        # ----- 3. Вычисляем отклонение цены от средней по категории -----
        features_df['price_deviation_from_category_mean'] = \
            features_df[price_col] / features_df['category_avg_price']
        # Заполняем NaN, если средняя цена была NaN (например, для новых категорий)
        features_df['price_deviation_from_category_mean'] = features_df['price_deviation_from_category_mean'].fillna(1.0)
        # Убираем временную колонку
        features_df.drop(columns=['category_avg_price'], inplace=True)
        logger.info("Добавлен признак: 'price_deviation_from_category_mean'")

    # ----- 4. Вычисляем отношение стоимости доставки к цене товара ----- 
    if price_col in features_df.columns and freight_col in features_df.columns:
        # Используем np.divide для безопасного деления на ноль (результат будет inf)
        features_df['freight_to_price_ratio'] = np.divide(
            features_df[freight_col],
            features_df[price_col]
        )
        # Заменяем бесконечность (если цена была 0) на NaN
        features_df['freight_to_price_ratio'] = features_df['freight_to_price_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info("Добавлен признак: 'freight_to_price_ratio'")
    else:
        logger.warning(f"engineer_features: Не удалось вычислить 'freight_to_price_ratio'. Отсутствуют колонки '{price_col}' или '{freight_col}'.")
        if 'freight_to_price_ratio' not in features_df.columns:
             features_df['freight_to_price_ratio'] = np.nan

    # ----- Добавление других признаков при необходимости -----
    # ...
    
    return features_df


# Конец файла, удаляем закомментированные неиспользуемые функции

# Удаляем неиспользуемую функцию get_category_avg_prices, если она была просто заглушкой.
# Также удаляем закомментированный код engineer_order_item_features и calculate_category_stats, если он будет найден.
# (Если эти функции используются или реализованы, этот комментарий и удаление ниже не будут применены к ним)

# --- Пример удаления, если бы они были в конце файла и не использовались: ---
# @lru_cache(maxsize=32)
# def get_category_avg_prices(category_data_key: str) -> pd.Series:
#     pass

# def engineer_order_item_features(order_items_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
#     # ... (код был бы здесь) ...
#     pass

# def calculate_category_stats(products_df: pd.DataFrame) -> pd.DataFrame:
#     # ... (код был бы здесь) ...
#     pass 