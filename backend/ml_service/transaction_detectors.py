"""
transaction_detectors.py - Детекторы аномалий для транзакционного уровня
=================================================================
Модуль содержит реализации детекторов для обнаружения аномалий
в отдельных транзакциях (необычные цены, стоимость доставки).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
import joblib
import os

from .detector import AnomalyDetector, StatisticalDetector, IsolationForestDetector, AutoencoderDetector
from .vae_detector import VAEDetector

# Настраиваем логирование
logger = logging.getLogger(__name__)

class PriceFreightRatioDetector(StatisticalDetector):
    """
    Детектор, основанный на соотношении цены товара и стоимости доставки.
    
    Выявляет аномальные транзакции, где соотношение стоимости доставки к цене
    товара слишком высокое (может указывать на мошенничество или ошибки ввода).
    """
    
    def __init__(self, threshold: float = 3.0, min_price: float = 1.0, model_name: str = "price_freight_ratio"):
        """
        Инициализирует детектор соотношения цены и доставки.
        
        Args:
            threshold: Пороговое значение Z-score для определения аномалий
            min_price: Минимальная цена товара для учета (чтобы избежать деления на очень маленькие числа)
            model_name: Уникальное имя модели
        """
        super().__init__(feature="freight_to_price_ratio", threshold=threshold, model_name=model_name)
        self.min_price = min_price
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует данные, добавляя признак соотношения доставки к цене.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame с добавленным признаком
        """
        df = data.copy()
        
        # Проверяем наличие необходимых столбцов
        if 'price' not in df.columns or 'freight_value' not in df.columns:
            raise ValueError("Для работы PriceFreightRatioDetector необходимы столбцы 'price' и 'freight_value'")
        
        # Ограничиваем минимальную цену для избежания деления на очень маленькие числа
        effective_price = np.maximum(df['price'], self.min_price)
        
        # Вычисляем соотношение стоимости доставки к цене
        df['freight_to_price_ratio'] = df['freight_value'] / effective_price
        
        # Обрабатываем бесконечные значения
        df['freight_to_price_ratio'] = df['freight_to_price_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Заполняем отсутствующие значения медианой
        df['freight_to_price_ratio'] = df['freight_to_price_ratio'].fillna(df['freight_to_price_ratio'].median())
        
        return df

    def train(self, data: pd.DataFrame):
        """
        Обучение детектора: вызов preprocess и обучение базового StatisticalDetector.
        """
        # Сначала добавляем признак freight_to_price_ratio
        processed_data = self.preprocess(data)
        # Затем обучаем базовый детектор на этом признаке
        super().train(processed_data)

class CategoryPriceOutlierDetector(AnomalyDetector):
    """
    Детектор, выявляющий аномальные цены относительно средних цен в категории.
    
    Обнаруживает товары, цены которых значительно отличаются от средних цен
    в их категории, что может указывать на ошибки ценообразования или мошенничество.
    """
    
    def __init__(self, 
                 category_col: str, 
                 price_col: str, 
                 threshold: float = 2.5, 
                 min_samples_per_category: int = 5, # Переименовано для соответствия YAML
                 model_name: str = "category_price_outlier"):
        """
        Инициализирует детектор аномальных цен в категориях.
        
        Args:
            category_col: Название столбца с категориями товаров.
            price_col: Название столбца с ценами товаров.
            threshold: Пороговое значение отклонения (в стандартных отклонениях) для определения аномалий.
            min_samples_per_category: Минимальное количество товаров в категории для расчета статистик.
            model_name: Уникальное имя модели.
        """
        super().__init__(model_name=model_name)
        self.category_col = category_col
        self.price_col = price_col
        self.threshold = threshold
        self.min_samples_per_category = min_samples_per_category # Сохраняем параметр
        self.category_stats: Dict[str, Dict[str, float]] = {} # Типизация для ясности
        logger.info(f"CategoryPriceOutlierDetector '{model_name}' инициализирован с параметрами: "
                    f"category_col='{category_col}', price_col='{price_col}', threshold={threshold}, "
                    f"min_samples_per_category={min_samples_per_category}")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        if self.price_col not in df.columns or self.category_col not in df.columns:
            raise ValueError(f"Для работы CategoryPriceOutlierDetector необходимы столбцы '{self.price_col}' и '{self.category_col}'")
        
        df[self.category_col] = df[self.category_col].fillna('unknown')
        return df
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: вычисление статистик по категориям.
        
        Args:
            data: DataFrame с обучающими данными
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        df = self.preprocess(data)
        self.category_stats = {}
        category_groups = df.groupby(self.category_col)
        
        for category, group in category_groups:
            if len(group) < self.min_samples_per_category: # Используем сохраненный параметр
                continue
            
            mean = group[self.price_col].mean()
            std = group[self.price_col].std()
            if std < 0.001: std = 0.001
                
            self.category_stats[category] = {
                'mean': mean, 'std': std,
                'min': group[self.price_col].min(), 'max': group[self.price_col].max(),
                'count': len(group)
            }
        
        all_scores = []
        for category, group in category_groups:
            if category not in self.category_stats: continue
            stats = self.category_stats[category]
            z_scores = np.abs((group[self.price_col] - stats['mean']) / stats['std'])
            all_scores.extend(z_scores.tolist())
            
        self.fit_normalizer(np.array(all_scores))
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: {len(self.category_stats)} категорий")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в новых данных.
        Args:
            data: DataFrame с тестовыми данными
        Returns:
            DataFrame с оригинальными данными и добавленными скорами аномалий
        """
        if not self.is_trained:
            raise RuntimeError(f"Детектор {self.model_name} не обучен")
        df = self.preprocess(data)
        anomaly_scores = np.zeros(len(df))
        
        for category, stats in self.category_stats.items():
            mask = (df[self.category_col] == category)
            if not any(mask): continue
            z_scores = np.abs((df.loc[mask, self.price_col] - stats['mean']) / stats['std'])
            anomaly_scores[mask] = z_scores
        
        unknown_categories_mask = ~df[self.category_col].isin(self.category_stats.keys())
        if any(unknown_categories_mask):
            if self.max_score_ is not None and np.isfinite(self.max_score_):
                anomaly_scores[unknown_categories_mask] = self.max_score_ + 1.0 
            else:
                anomaly_scores[unknown_categories_mask] = self.threshold + 2.0 
            logger.info(f"Для {sum(unknown_categories_mask)} товаров с неизвестными категориями присвоен высокий скор аномальности.")
        
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = anomaly_scores > self.threshold
        
        # Нормализуем скоры и добавляем их в DataFrame
        if hasattr(self, 'min_score_') and hasattr(self, 'max_score_') and self.min_score_ is not None and self.max_score_ is not None:
            df['anomaly_score_normalized'] = self.normalize_score(df['anomaly_score'].values)
        else:
            df['anomaly_score_normalized'] = df['anomaly_score']  # Если нормализатор не обучен, оставляем как есть
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} транзакций")
        return df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает дополнительные атрибуты для сохранения."""
        return {
            'category_stats': self.category_stats,
            'threshold': self.threshold,
            'min_samples_per_category': self.min_samples_per_category,
            'category_col': self.category_col,
            'price_col': self.price_col
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]):
        """Загружает дополнительные атрибуты."""
        self.category_stats = loaded_data.get('category_stats', {})
        self.threshold = loaded_data.get('threshold', 2.5)
        self.min_samples_per_category = loaded_data.get('min_samples_per_category', 5)
        self.category_col = loaded_data.get('category_col', 'product_category_name') # Значение по умолчанию, если нет в файле
        self.price_col = loaded_data.get('price_col', 'price') # Значение по умолчанию
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детали для объяснения аномалий на основе статистик цен по категориям.
        
        Args:
            data_for_explanation_raw: DataFrame с "сырыми" данными для объяснения.
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None, если возникла ошибка.
        """
        if not self.is_trained or not self.category_stats:
            logger.warning(f"({self.model_name}) Детектор не обучен или отсутствуют статистики по категориям. Невозможно предоставить объяснения.")
            return None
        
        try:
            df = self.preprocess(data_for_explanation_raw.copy())
            if df.empty or self.price_col not in df.columns or self.category_col not in df.columns:
                logger.warning(f"({self.model_name}) После предобработки данные пусты или отсутствуют необходимые столбцы.")
                return None
            
            explanations = []
            for index, row in df.iterrows():
                category = row[self.category_col]
                price = row[self.price_col]
                
                explanation_info = {
                    "category": category,
                    "price": float(price) if pd.notna(price) else None,
                }
                
                if category in self.category_stats:
                    stats = self.category_stats[category]
                    z_score = abs(price - stats['mean']) / stats['std'] if stats['std'] > 0 else 0.0
                    
                    explanation_info.update({
                        "category_mean_price": float(stats['mean']),
                        "category_std_price": float(stats['std']),
                        "category_min_price": float(stats['min']),
                        "category_max_price": float(stats['max']),
                        "category_count": int(stats['count']),
                        "z_score": float(z_score) if not np.isnan(z_score) else None,
                        "threshold": float(self.threshold),
                        "explanation": f"Цена {price:.2f} отличается от средней цены в категории '{category}' ({stats['mean']:.2f}) на {z_score:.2f} стандартных отклонений (порог: {self.threshold})."
                    })
                else:
                    explanation_info["explanation"] = f"Категория '{category}' не встречалась в обучающих данных, статистики отсутствуют."
                
                explanations.append({"detector_specific_info": explanation_info})
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None

class MultiFeatureIsolationForestDetector(IsolationForestDetector):
    """
    Многопризнаковый детектор на основе Isolation Forest для транзакционного уровня.
    
    Использует несколько признаков для обнаружения аномальных транзакций,
    включая цену, стоимость доставки, вес и другие доступные числовые признаки.
    """
    
    def __init__(self, 
                 features: List[str] = None, 
                 n_estimators: int = 100, 
                 contamination: str = 'auto', 
                 random_state: int = 42,
                 model_name: str = "transaction_isolation_forest"):
        """
        Инициализирует многопризнаковый детектор на основе Isolation Forest.
        
        Args:
            features: Список используемых признаков (если None, используются стандартные)
            n_estimators: Количество деревьев в ансамбле
            contamination: Ожидаемая доля выбросов в данных
            random_state: Seed для генератора случайных чисел
            model_name: Уникальное имя модели
        """
        # Если признаки не указаны, используем стандартный набор
        if features is None:
            features = ['price', 'freight_value', 'product_weight_g', 'product_photos_qty']
            
        super().__init__(features=features, n_estimators=n_estimators, 
                        contamination=contamination, random_state=random_state,
                        model_name=model_name)
        
    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Расширенная предобработка данных с добавлением производных признаков.
        
        Args:
            data: Исходный DataFrame с данными
            fit_scaler: Флаг, указывающий, нужно ли обучить скейлер
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        # Добавляем производные признаки, если есть необходимые данные
        try:
            if 'price' in df.columns and 'freight_value' in df.columns:
                # Соотношение доставки к цене
                df['freight_to_price_ratio'] = df['freight_value'] / df['price'].replace(0, 0.01)
                
            if 'product_weight_g' in df.columns and 'price' in df.columns:
                # Цена за грамм (для выявления аномально дорогих/дешевых товаров)
                df['price_per_gram'] = df['price'] / df['product_weight_g'].replace(0, 0.01)
                
                # Добавляем в список признаков, если их там еще нет
                if 'freight_to_price_ratio' not in self.features and 'freight_to_price_ratio' in df.columns:
                    self.features.append('freight_to_price_ratio')
                    
                if 'price_per_gram' not in self.features and 'price_per_gram' in df.columns:
                    self.features.append('price_per_gram')
                    
            # Обрабатываем бесконечные значения и NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Для каждого признака заполняем NaN медианой
            for feature in self.features:
                if feature in df.columns and df[feature].isna().any():
                    df[feature] = df[feature].fillna(df[feature].median())
        except Exception as e:
            logger.warning(f"Ошибка при создании производных признаков: {e}")
        
        # Вызываем стандартную предобработку родительского класса
        return super().preprocess(df, fit_scaler)

class TransactionVAEDetector(VAEDetector):
    """
    Детектор аномалий на основе вариационного автокодировщика (VAE) для транзакционных данных.
    
    Этот детектор использует VAE для изучения нормального распределения признаков транзакций.
    Аномалии идентифицируются как точки данных с высокой ошибкой реконструкции.
    """
    
    def __init__(self, 
                 features: List[str],
                 encoding_dim: int = 8, 
                 hidden_dim1: int = 32,
                 hidden_dim2: int = 16, 
                 epochs: int = 20,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 kld_weight: float = 0.5, 
                 dropout_rate: float = 0.1, 
                 shap_background_samples: int = 100, 
                 model_name: str = "transaction_vae"):
        
        if not features:
            raise ValueError("Параметр 'features' является обязательным для TransactionVAEDetector.")
            
        # Определяем ПОЛНЫЙ список признаков, которые будет использовать этот детектор,
        # включая те, что будут сгенерированы в его preprocess.
        # Это важно, чтобы базовый VAEDetector.preprocess знал, какие колонки ожидать.
        actual_features_to_use = list(features) # Копируем исходные
        if 'price' in actual_features_to_use and 'freight_value' in actual_features_to_use:
            if 'freight_to_price_ratio' not in actual_features_to_use:
                actual_features_to_use.append('freight_to_price_ratio')
        if ('product_length_cm' in actual_features_to_use and 
            'product_height_cm' in actual_features_to_use and 
            'product_width_cm' in actual_features_to_use):
            if 'product_volume' not in actual_features_to_use:
                actual_features_to_use.append('product_volume')
        
        # Сохраняем исходные features из конфига, если они нужны для чего-то специфичного
        # в TransactionVAEDetector до того, как они смешаются с генерируемыми
        self.config_features = list(features) 

        # Сохраняем специфичные для TransactionVAE параметры, если они не используются напрямую базовым VAEDetector
        # или если у них другие имена/логика.
        self.hidden_dim2_transaction = hidden_dim2 
        # dropout_rate уже передается в super и сохраняется там как self.dropout_rate
        # self.dropout_rate_transaction = dropout_rate 

        super().__init__(
            features=actual_features_to_use, # <--- ПЕРЕДАЕМ ПОЛНЫЙ СПИСОК
            latent_dim=encoding_dim,  
            hidden_dim=hidden_dim1,   # hidden_dim1 используется как основной hidden_dim для VAE
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            kld_weight=kld_weight,
            dropout_rate=dropout_rate, # Передаем dropout_rate из TransactionVAEDetector
            shap_background_samples=shap_background_samples,
            model_name=model_name
        )
        
        logger.info(
            f"({self.model_name}) TransactionVAEDetector инициализирован. "
            f"Конфигурационные признаки: {self.config_features}, "
            f"Реально используемые признаки (с генерируемыми): {self.features}, "
            f"encoding_dim(latent_dim): {self.latent_dim}, hidden_dim1(hidden_dim): {self.hidden_dim}, "
            f"hidden_dim2: {self.hidden_dim2_transaction}, epochs: {self.epochs}, batch_size: {self.batch_size}, "
            f"lr: {self.learning_rate}, kld_w: {self.kld_weight}, dropout: {self.dropout_rate}"
        )
        # self._reset_state() # Вызов _reset_state() уже есть в VAEDetector.__init__ -> AnomalyDetector.__init__ -> VAEDetector._reset_state()

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> Optional[pd.DataFrame]:
        """
        Предобработка данных для TransactionVAEDetector.
        Создает производные признаки 'freight_to_price_ratio' и 'product_volume',
        обрабатывает пропуски и вызывает preprocess базового класса.
        """
        df = data.copy()
        try:
            # Генерация производных признаков
            # Эти признаки УЖЕ должны быть в self.features благодаря __init__
            if 'price' in df.columns and 'freight_value' in df.columns:
                df['freight_to_price_ratio'] = df['freight_value'] / df['price'].replace(0, 1e-6) # Избегаем деления на ноль
            
            if ('product_length_cm' in df.columns and 
                'product_height_cm' in df.columns and 
                'product_width_cm' in df.columns):
                df['product_volume'] = (df['product_length_cm'] * 
                                        df['product_height_cm'] * 
                                        df['product_width_cm'])
            
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Заполняем NaN медианой ТОЛЬКО для тех признаков, которые реально будут использоваться (self.features)
            # self.features уже должен содержать полный список из __init__
            for feature in self.features:
                if feature in df.columns:
                    if df[feature].isna().any():
                        df[feature] = df[feature].fillna(df[feature].median())
                else:
                    # Если генерируемый признак так и не появился (например, не было product_length_cm)
                    # или если признак из конфига отсутствует в исходных данных
                    # VAEDetector.preprocess это отловит и выдаст ошибку, что является корректным поведением.
                    logger.warning(
                        f"({self.model_name}) В TransactionVAEDetector.preprocess: "
                        f"Ожидаемый признак '{feature}' отсутствует в DataFrame перед вызовом super().preprocess. "
                        f"Колонки в df: {list(df.columns)}"
                    )
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при создании производных признаков или заполнении NaN в TransactionVAEDetector.preprocess: {e}", exc_info=True)
            # Важно не возвращать None здесь, чтобы ошибка была поймана в VAEDetector.preprocess при проверке колонок
            # return None # Не возвращаем None, чтобы super().preprocess мог проверить колонки

        # Вызов preprocess базового класса (VAEDetector)
        # VAEDetector.preprocess ожидает, что все self.features будут в df.
        return super().preprocess(df, fit_scaler) 