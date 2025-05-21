"""
behavior_detectors.py - Детекторы аномалий для поведенческого уровня
=================================================================
Модуль содержит реализации детекторов для обнаружения аномалий
в поведении продавцов и покупателей (нестабильное ценообразование и т.д.)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .detector import AnomalyDetector, IsolationForestDetector

# Настраиваем логирование
logger = logging.getLogger(__name__)

class SellerPricingBehaviorDetector(AnomalyDetector):
    """
    Детектор аномального поведения продавцов в ценообразовании.
    
    Выявляет продавцов с необычным ценообразованием: резкие скачки цен,
    нестабильные цены на товары одной категории, аномальные соотношения
    цены и доставки и т.д.
    """
    
    def __init__(self, 
                 volatility_threshold: float = 0.5, 
                 range_threshold: float = 5.0,
                 freight_ratio_threshold: float = 1.0,
                 min_transactions: int = 5,
                 model_name: str = "seller_pricing_behavior"):
        """
        Инициализирует детектор аномального ценообразования продавцов.
        
        Args:
            volatility_threshold: Пороговое значение волатильности цен (std/mean)
            range_threshold: Пороговое значение диапазона цен (max/min)
            freight_ratio_threshold: Порог для соотношения доставки к цене
            min_transactions: Минимальное количество транзакций для анализа
            model_name: Уникальное имя модели
        """
        super().__init__(model_name=model_name)
        self.volatility_threshold = volatility_threshold
        self.range_threshold = range_threshold
        self.freight_ratio_threshold = freight_ratio_threshold
        self.min_transactions = min_transactions
        
        # Статистики по всем продавцам
        self.global_stats = {}
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        # Проверяем, содержит ли DataFrame агрегированные данные по продавцам
        # или нужно их агрегировать
        if 'seller_id' in df.columns and 'mean_price' not in df.columns:
            # Это сырые данные, которые нужно агрегировать
            logger.info("Агрегация данных по продавцам для анализа поведения...")
            
            required_columns = ['seller_id', 'price', 'freight_value']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Для работы SellerPricingBehaviorDetector необходимы столбцы: {required_columns}")
                
            # Агрегируем данные по продавцам
            seller_agg = df.groupby('seller_id').agg(
                mean_price=('price', 'mean'),
                std_price=('price', 'std'),
                count=('price', 'count'),
                min_price=('price', 'min'),
                max_price=('price', 'max'),
                mean_freight=('freight_value', 'mean'),
                std_freight=('freight_value', 'std')
            ).reset_index()
            
            # Вычисляем дополнительные признаки для поведенческой аналитики
            seller_agg['price_volatility'] = seller_agg['std_price'] / seller_agg['mean_price'].replace(0, np.nan)
            seller_agg['price_range_ratio'] = seller_agg['max_price'] / seller_agg['min_price'].replace(0, np.nan)
            seller_agg['freight_to_price_ratio'] = seller_agg['mean_freight'] / seller_agg['mean_price'].replace(0, np.nan)
            
            # Заполняем NaN и бесконечности
            seller_agg = seller_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Отфильтровываем продавцов с малым количеством транзакций
            seller_agg = seller_agg[seller_agg['count'] >= self.min_transactions]
            
            return seller_agg
        else:
            # Данные уже агрегированы, просто проверяем наличие нужных столбцов
            required_columns = ['seller_id', 'price_volatility', 'price_range_ratio', 'freight_to_price_ratio']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Отсутствуют некоторые столбцы из: {required_columns}")
            
            return df
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: вычисление глобальных статистик по продавцам.
        
        Args:
            data: DataFrame с обучающими данными
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        
        # Предобработка данных
        df = self.preprocess(data)
        
        if df.empty:
            raise ValueError("После предобработки данных получен пустой DataFrame")
        
        # Вычисляем глобальные статистики
        self.global_stats = {
            'volatility': {
                'mean': df['price_volatility'].mean(),
                'std': df['price_volatility'].std(),
                'median': df['price_volatility'].median(),
                'q75': df['price_volatility'].quantile(0.75),
                'q95': df['price_volatility'].quantile(0.95)
            },
            'range_ratio': {
                'mean': df['price_range_ratio'].mean(),
                'std': df['price_range_ratio'].std(),
                'median': df['price_range_ratio'].median(),
                'q75': df['price_range_ratio'].quantile(0.75),
                'q95': df['price_range_ratio'].quantile(0.95)
            },
            'freight_ratio': {
                'mean': df['freight_to_price_ratio'].mean(),
                'std': df['freight_to_price_ratio'].std(),
                'median': df['freight_to_price_ratio'].median(),
                'q75': df['freight_to_price_ratio'].quantile(0.75),
                'q95': df['freight_to_price_ratio'].quantile(0.95)
            }
        }
        
        # Собираем скоры для обучения нормализатора
        scores = self._calculate_anomaly_scores(df)
        
        # Обучаем нормализатор скоров
        self.fit_normalizer(scores)
        
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: проанализировано {len(df)} продавцов")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в поведении продавцов.
        
        Args:
            data: DataFrame с данными для анализа
            
        Returns:
            DataFrame с оригинальными данными и добавленными скорами аномалий
        """
        if not self.is_trained:
            raise RuntimeError(f"Детектор {self.model_name} не обучен")
            
        # Предобработка данных
        df = self.preprocess(data)
        
        if df.empty:
            logger.warning("После предобработки данных получен пустой DataFrame")
            return df
        
        # Вычисляем скоры аномалий
        anomaly_scores = self._calculate_anomaly_scores(df)
        
        # Добавляем скоры в результат
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = False  # По умолчанию не аномалия
        
        # Определяем аномалии с использованием порогов
        if 'price_volatility' in df.columns:
            volatility_anomalies = df['price_volatility'] > self.global_stats['volatility']['q95']
            df.loc[volatility_anomalies, 'is_anomaly'] = True
            
        if 'price_range_ratio' in df.columns:
            range_anomalies = df['price_range_ratio'] > self.global_stats['range_ratio']['q95']
            df.loc[range_anomalies, 'is_anomaly'] = True
            
        if 'freight_to_price_ratio' in df.columns:
            freight_anomalies = df['freight_to_price_ratio'] > self.global_stats['freight_ratio']['q95']
            df.loc[freight_anomalies, 'is_anomaly'] = True
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} продавцов")
        
        return df
    
    def _calculate_anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет скоры аномалий для продавцов.
        
        Args:
            data: DataFrame с агрегированными данными по продавцам
            
        Returns:
            numpy.ndarray со скорами аномалий
        """
        df = data.copy()
        
        # Инициализируем скоры
        anomaly_scores = np.zeros(len(df))
        
        # Считаем отклонения по каждому признаку
        if 'price_volatility' in df.columns and 'volatility' in self.global_stats:
            volatility_dev = (df['price_volatility'] - self.global_stats['volatility']['median']) / max(self.global_stats['volatility']['std'], 0.001)
            volatility_scores = np.maximum(0, volatility_dev)
            anomaly_scores = np.maximum(anomaly_scores, volatility_scores)
            
        if 'price_range_ratio' in df.columns and 'range_ratio' in self.global_stats:
            range_dev = (df['price_range_ratio'] - self.global_stats['range_ratio']['median']) / max(self.global_stats['range_ratio']['std'], 0.001)
            range_scores = np.maximum(0, range_dev)
            anomaly_scores = np.maximum(anomaly_scores, range_scores)
            
        if 'freight_to_price_ratio' in df.columns and 'freight_ratio' in self.global_stats:
            freight_dev = (df['freight_to_price_ratio'] - self.global_stats['freight_ratio']['median']) / max(self.global_stats['freight_ratio']['std'], 0.001)
            freight_scores = np.maximum(0, freight_dev)
            anomaly_scores = np.maximum(anomaly_scores, freight_scores)
        
        return anomaly_scores

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты SellerPricingBehaviorDetector для сохранения."""
        # min_score_, max_score_ будут сохранены базовым классом из self.
        # is_trained не сохраняем, он определяется при загрузке.
        # self.global_stats сохраняем как 'model_state', чтобы базовая логика его подхватила.
        return {
            'model_state': self.global_stats, 
            'volatility_threshold': self.volatility_threshold,
            'range_threshold': self.range_threshold,
            'freight_ratio_threshold': self.freight_ratio_threshold,
            'min_transactions': self.min_transactions
            # scaler не используется этим детектором, так что он будет None и корректно обработается
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты SellerPricingBehaviorDetector."""
        # self.model (здесь это self.global_stats), self.min_score_, self.max_score_ 
        # уже должны быть установлены (или попытка сделана) базовым load_model.
        
        if self.model is not None: 
            self.global_stats = self.model
        else:
            self.global_stats = loaded_data.get('global_stats', {}) 
            if not self.global_stats:
                 logger.warning(f"global_stats не найдены при загрузке {self.model_name}")

        self.volatility_threshold = loaded_data.get('volatility_threshold', self.volatility_threshold)
        self.range_threshold = loaded_data.get('range_threshold', self.range_threshold)
        self.freight_ratio_threshold = loaded_data.get('freight_ratio_threshold', self.freight_ratio_threshold)
        self.min_transactions = loaded_data.get('min_transactions', self.min_transactions)
        
        # Определяем is_trained
        valid_global_stats = isinstance(self.global_stats, dict) and bool(self.global_stats)
        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_global_stats and normalizer_loaded:
            self.is_trained = True
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_global_stats:
                missing_parts_log.append("global_stats")
            if not normalizer_loaded:
                missing_parts_log.append("параметры нормализации")
            
            if missing_parts_log: # Логируем только если есть что сообщить
                logger.warning(f"{self.model_name} не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")
            # Если оба False, но missing_parts_log пуст (маловероятно при текущей логике),
            # можно добавить еще одно общее предупреждение, но пока оставим так.

        self._custom_is_trained_logic_applied_in_load_additional = True
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детальные объяснения для обнаруженных аномалий в ценовом поведении продавцов.
        
        Args:
            data_for_explanation_raw: DataFrame с данными продавцов (агрегированные данные)
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None, если возникла ошибка
        """
        if not self.is_trained:
            logger.warning(f"({self.model_name}) Детектор не обучен. Невозможно предоставить объяснения.")
            return None
            
        try:
            explanations = []
            
            for _, row in data_for_explanation_raw.iterrows():
                # Проверяем наличие необходимых колонок
                required_cols = ['seller_id', 'mean_price', 'std_price', 'price_volatility', 
                                'price_range_ratio', 'freight_to_price_ratio', 'count']
                missing_cols = [col for col in required_cols if col not in row.index]
                
                if missing_cols:
                    logger.warning(f"({self.model_name}) Отсутствуют необходимые колонки для объяснения: {missing_cols}")
                    explanations.append({"detector_specific_info": {"error": f"Missing columns: {missing_cols}"}})
                    continue
                
                # Проверяем минимальное количество транзакций
                if row['count'] < self.min_transactions:
                    explanations.append({"detector_specific_info": {
                        "warning": f"Недостаточно транзакций для надежного анализа ({int(row['count'])} < {self.min_transactions})"
                    }})
                    continue
                
                # Собираем факторы аномальности
                anomaly_factors = []
                
                # Проверяем волатильность цен
                if row['price_volatility'] > self.volatility_threshold:
                    volatility_factor = {
                        "factor": "price_volatility",
                        "value": float(row['price_volatility']),
                        "threshold": float(self.volatility_threshold),
                        "description": f"Высокая волатильность цен ({row['price_volatility']:.2f} > {self.volatility_threshold:.2f})"
                    }
                    anomaly_factors.append(volatility_factor)
                
                # Проверяем диапазон цен
                if row['price_range_ratio'] > self.range_threshold:
                    range_factor = {
                        "factor": "price_range_ratio",
                        "value": float(row['price_range_ratio']),
                        "threshold": float(self.range_threshold),
                        "description": f"Широкий диапазон цен (max/min = {row['price_range_ratio']:.2f} > {self.range_threshold:.2f})"
                    }
                    anomaly_factors.append(range_factor)
                
                # Проверяем соотношение стоимости доставки к цене
                if row['freight_to_price_ratio'] > self.freight_ratio_threshold:
                    freight_factor = {
                        "factor": "freight_to_price_ratio",
                        "value": float(row['freight_to_price_ratio']),
                        "threshold": float(self.freight_ratio_threshold),
                        "description": f"Высокое отношение стоимости доставки к цене ({row['freight_to_price_ratio']:.2f} > {self.freight_ratio_threshold:.2f})"
                    }
                    anomaly_factors.append(freight_factor)
                
                # Создаем человекочитаемое объяснение
                text_explanation = "Выявлены следующие аномалии в ценовом поведении продавца:"
                for factor in anomaly_factors:
                    text_explanation += f"\n- {factor['description']}"
                
                if not anomaly_factors:
                    text_explanation = "Конкретных факторов аномальности в ценовом поведении не выявлено, но общий паттерн ценообразования нетипичен."
                
                # Формируем итоговое объяснение
                explanation = {
                    "detector_specific_info": {
                        "seller_id": str(row.get('seller_id', 'N/A')),
                        "anomaly_factors": anomaly_factors,
                        "text_explanation": text_explanation,
                        "raw_metrics": {
                            "mean_price": float(row['mean_price']),
                            "std_price": float(row['std_price']),
                            "price_volatility": float(row['price_volatility']),
                            "price_range_ratio": float(row['price_range_ratio']),
                            "freight_to_price_ratio": float(row['freight_to_price_ratio']),
                            "transaction_count": int(row['count'])
                        }
                    }
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None

class SellerCategoryMixDetector(AnomalyDetector):
    """
    Детектор аномального поведения продавцов на основе разнообразия категорий товаров.
    
    Обнаруживает продавцов с необычным разнообразием категорий товаров,
    что может указывать на мошенническую активность или другие аномалии.
    """
    
    def __init__(self, diversity_threshold: float = 0.8, min_transactions: int = 5,
                model_name: str = "seller_category_mix"):
        """
        Инициализирует детектор разнообразия категорий продавцов.
        
        Args:
            diversity_threshold: Пороговое значение для индекса разнообразия категорий
            min_transactions: Минимальное количество транзакций для анализа
            model_name: Уникальное имя модели
        """
        super().__init__(model_name=model_name)
        self.diversity_threshold = diversity_threshold
        self.min_transactions = min_transactions
        
        # Глобальные статистики
        self.global_stats_categories = {}
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет наличие необходимых агрегированных столбцов.
        
        Args:
            data: DataFrame с предположительно агрегированными данными
            
        Returns:
            DataFrame, если столбцы есть
            
        Raises:
            ValueError: Если столбцы отсутствуют
        """
        # Теперь этот метод просто проверяет наличие УЖЕ АГРЕГИРОВАННЫХ столбцов,
        # которые должен подготовить вызывающий код (например, MultilevelDetectorService)
        required_columns = ['seller_id', 'category_diversity', 'normalized_diversity']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Для SellerCategoryMixDetector необходимы агрегированные столбцы: {missing}")
        
        return data # Возвращаем как есть, если все хорошо
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: вычисление глобальных статистик по разнообразию категорий.
        Работает с УЖЕ АГРЕГИРОВАННЫМИ данными.
        
        Args:
            data: DataFrame с агрегированными данными по продавцам
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        
        # Проверяем наличие нужных столбцов (больше не агрегируем здесь)
        df = self.preprocess(data)
        
        if df.empty:
            raise ValueError("Получен пустой DataFrame для обучения")
        
        # Вычисляем глобальные статистики (среднее, стд, квантили) по готовым колонкам
        self.global_stats_categories = {
            'diversity': {
                'mean': df['category_diversity'].mean(),
                'std': df['category_diversity'].std(),
                'median': df['category_diversity'].median(),
                'q75': df['category_diversity'].quantile(0.75),
                'q95': df['category_diversity'].quantile(0.95)
            },
            'normalized_diversity': {
                'mean': df['normalized_diversity'].mean(),
                'std': df['normalized_diversity'].std(),
                'median': df['normalized_diversity'].median(),
                'q75': df['normalized_diversity'].quantile(0.75),
                'q95': df['normalized_diversity'].quantile(0.95)
            }
        }
        
        # Вычисляем скоры аномалий для обучения нормализатора
        # Используем Z-образные скоры по обоим показателям разнообразия
        diversity_dev = (df['category_diversity'] - self.global_stats_categories['diversity']['median']) / \
                        max(self.global_stats_categories['diversity']['std'], 0.001)
        norm_diversity_dev = (df['normalized_diversity'] - self.global_stats_categories['normalized_diversity']['median']) / \
                            max(self.global_stats_categories['normalized_diversity']['std'], 0.001)
        
        # Берем максимальное значение из двух скоров
        anomaly_scores = np.maximum(diversity_dev.fillna(0), norm_diversity_dev.fillna(0))
        
        # Обучаем нормализатор скоров
        self.fit_normalizer(anomaly_scores)
        
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: проанализировано {len(df)} продавцов")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в разнообразии категорий продавцов.
        Работает с УЖЕ АГРЕГИРОВАННЫМИ данными.
        
        Args:
            data: DataFrame с агрегированными данными по продавцам
            
        Returns:
            DataFrame с оригинальными данными и добавленными скорами аномалий
        """
        if not self.is_trained:
            raise RuntimeError(f"Детектор {self.model_name} не обучен")
            
        # Проверяем наличие агрегированных столбцов
        df = self.preprocess(data)
        
        if df.empty:
            logger.warning("Получен пустой DataFrame для детекции")
            return df
        
        # Вычисляем скоры аномалий на основе сохраненных статистик
        diversity_dev = (df['category_diversity'] - self.global_stats_categories['diversity']['median']) / \
                        max(self.global_stats_categories['diversity']['std'], 0.001)
        norm_diversity_dev = (df['normalized_diversity'] - self.global_stats_categories['normalized_diversity']['median']) / \
                            max(self.global_stats_categories['normalized_diversity']['std'], 0.001)
        
        # Берем максимальное значение из двух скоров
        anomaly_scores = np.maximum(diversity_dev.fillna(0), norm_diversity_dev.fillna(0))
        
        # Добавляем скоры в результат
        df['anomaly_score'] = anomaly_scores
        
        # Нормализуем скоры
        normalized_scores = self.normalize_score(df['anomaly_score'].values)
        df['normalized_score'] = normalized_scores
        
        # Определяем аномалии на основе 95-го перцентиля обучающей выборки
        df['is_anomaly'] = (df['category_diversity'] > self.global_stats_categories['diversity']['q95']) | \
                           (df['normalized_diversity'] > self.global_stats_categories['normalized_diversity']['q95'])
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} продавцов")
        
        return df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты SellerCategoryMixDetector для сохранения."""
        # min_score_, max_score_ сохраняются базовым классом.
        # self.global_stats_categories сохраняем как 'model_state'.
        # Параметры конструктора (diversity_threshold, min_transactions) сохраним отдельно,
        # т.к. они не являются частью 'model_state' в данном случае.
        return {
            'model_state': self.global_stats_categories, 
            'diversity_threshold': self.diversity_threshold,
            'min_transactions': self.min_transactions
            # 'params' как отдельный словарь можно не сохранять, если все его содержимое - это параметры конструктора,
            # которые мы и так сохраняем.
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты SellerCategoryMixDetector."""
        # Базовый load_model уже попытался загрузить self.model (из 'model_state'), 
        # self.scaler (будет None для этого детектора), self.min_score_, self.max_score_.

        if self.model is not None: # self.model содержит то, что было в 'model_state'
            self.global_stats_categories = self.model
        else:
            # Если model_state не было, пытаемся загрузить global_stats_categories по старому ключу (если он там есть)
            self.global_stats_categories = loaded_data.get('global_stats_categories', {})
            if not self.global_stats_categories:
                 logger.warning(f"global_stats_categories не найдены при загрузке {self.model_name} (ни как model_state, ни как global_stats_categories).")
        
        # Загружаем параметры конструктора. Используем значения из loaded_data, 
        # если они есть, иначе оставляем текущие значения экземпляра (которые были установлены в __init__).
        self.diversity_threshold = loaded_data.get('diversity_threshold', self.diversity_threshold)
        self.min_transactions = loaded_data.get('min_transactions', self.min_transactions)

        # self.params в этом детекторе, похоже, не используется как основной источник порогов после инициализации,
        # так как пороги хранятся в self.diversity_threshold и self.min_transactions.
        # Если бы self.params активно использовался после load_model, его нужно было бы восстановить:
        # self.params = loaded_data.get('params', {'diversity_threshold': self.diversity_threshold, 'min_transactions': self.min_transactions})

        # Определяем is_trained
        valid_stats = isinstance(self.global_stats_categories, dict) and bool(self.global_stats_categories)
        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_stats and normalizer_loaded:
            self.is_trained = True
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_stats:
                missing_parts_log.append("global_stats_categories")
            if not normalizer_loaded:
                missing_parts_log.append("параметры нормализации")
            
            if missing_parts_log:
                logger.warning(f"{self.model_name} не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")
        
        self._custom_is_trained_logic_applied_in_load_additional = True
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детальные объяснения для обнаруженных аномалий в разнообразии категорий продавца.
        
        Args:
            data_for_explanation_raw: DataFrame с данными продавцов (агрегированные данные)
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None, если возникла ошибка
        """
        if not self.is_trained or not isinstance(self.global_stats_categories, dict):
            logger.warning(f"({self.model_name}) Детектор не обучен или отсутствуют необходимые статистики. Невозможно предоставить объяснения.")
            return None
            
        try:
            explanations = []
            
            for _, row in data_for_explanation_raw.iterrows():
                # Проверяем наличие необходимых колонок
                required_cols = ['seller_id', 'category_diversity', 'normalized_diversity', 
                                 'unique_categories', 'count']
                missing_cols = [col for col in required_cols if col not in row.index]
                
                if missing_cols:
                    logger.warning(f"({self.model_name}) Отсутствуют необходимые колонки для объяснения: {missing_cols}")
                    explanations.append({"detector_specific_info": {"error": f"Missing columns: {missing_cols}"}})
                    continue
                
                # Проверяем минимальное количество транзакций
                if row['count'] < self.min_transactions:
                    explanations.append({"detector_specific_info": {
                        "warning": f"Недостаточно транзакций для надежного анализа ({int(row['count'])} < {self.min_transactions})"
                    }})
                    continue
                
                # Получаем статистики из обучения
                diversity_median = self.global_stats_categories.get('diversity', {}).get('median', 0)
                diversity_q95 = self.global_stats_categories.get('diversity', {}).get('q95', 1)
                norm_diversity_median = self.global_stats_categories.get('normalized_diversity', {}).get('median', 0)
                norm_diversity_q95 = self.global_stats_categories.get('normalized_diversity', {}).get('q95', 1)
                
                # Проверяем, достигает ли разнообразие категорий порогов аномальности
                is_diversity_anomaly = row['category_diversity'] > diversity_q95
                is_norm_diversity_anomaly = row['normalized_diversity'] > norm_diversity_q95
                
                # Создаем объяснение
                explanation_text = ""
                anomaly_factors = []
                
                if is_diversity_anomaly:
                    diversity_ratio = row['category_diversity'] / diversity_median if diversity_median > 0 else float('inf')
                    factor = {
                        "factor": "category_diversity",
                        "value": float(row['category_diversity']),
                        "threshold": float(diversity_q95),
                        "median": float(diversity_median),
                        "ratio_to_median": float(diversity_ratio),
                        "description": f"Необычно высокое разнообразие категорий ({row['category_diversity']:.2f} > {diversity_q95:.2f}, что в {diversity_ratio:.1f}x выше медианы)"
                    }
                    anomaly_factors.append(factor)
                    explanation_text = f"Продавец имеет {int(row['unique_categories'])} уникальных категорий товаров на {int(row['count'])} транзакций " \
                                      f"(соотношение {row['category_diversity']:.2f}), что значительно выше типичного значения ({diversity_median:.2f})."
                
                if is_norm_diversity_anomaly:
                    norm_diversity_ratio = row['normalized_diversity'] / norm_diversity_median if norm_diversity_median > 0 else float('inf')
                    factor = {
                        "factor": "normalized_diversity",
                        "value": float(row['normalized_diversity']),
                        "threshold": float(norm_diversity_q95),
                        "median": float(norm_diversity_median),
                        "ratio_to_median": float(norm_diversity_ratio),
                        "description": f"Необычно высокое нормализованное разнообразие категорий ({row['normalized_diversity']:.2f} > {norm_diversity_q95:.2f}, что в {norm_diversity_ratio:.1f}x выше медианы)"
                    }
                    anomaly_factors.append(factor)
                    if not explanation_text:
                        explanation_text = f"Продавец имеет аномально высокое нормализованное разнообразие категорий ({row['normalized_diversity']:.2f}), " \
                                          f"что значительно выше типичного значения ({norm_diversity_median:.2f})."
                
                if not anomaly_factors:
                    explanation_text = "Конкретных факторов аномальности в разнообразии категорий не выявлено, но общий паттерн продавца нетипичен."
                
                # Формируем итоговое объяснение
                explanation = {
                    "detector_specific_info": {
                        "seller_id": str(row.get('seller_id', 'N/A')),
                        "anomaly_factors": anomaly_factors,
                        "text_explanation": explanation_text,
                        "raw_metrics": {
                            "category_diversity": float(row['category_diversity']),
                            "normalized_diversity": float(row['normalized_diversity']),
                            "unique_categories": int(row['unique_categories']),
                            "transaction_count": int(row['count'])
                        },
                        "thresholds": {
                            "diversity_threshold": float(diversity_q95),
                            "normalized_diversity_threshold": float(norm_diversity_q95)
                        }
                    }
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None

class BehaviorIsolationForestDetector(IsolationForestDetector):
    """
    Специализированный Isolation Forest для обнаружения аномалий в поведении продавцов.
    Агрегирует данные по продавцам, вычисляет поведенческие признаки и применяет IF.
    """
    def __init__(self, 
                 features: Optional[List[str]] = None, # Сделаем features опциональным, как в базовом
                 n_estimators: int = 100, 
                 contamination: Union[str, float] = 'auto', # Тип Union как в sklearn
                 random_state: Optional[int] = None, # Optional как в sklearn
                 model_name: str = "behavior_isolation_forest"):
        """
        Инициализирует детектор.
        
        Args:
            features: Список признаков для использования. Если None, будут использованы дефолтные.
            n_estimators: Количество базовых оценщиков в ансамбле.
            contamination: Доля выбросов в наборе данных. Может быть 'auto' или float.
            random_state: Контролирует случайность.
            model_name: Имя модели.
        """
        # Если признаки не заданы, используем дефолтный набор для поведенческого анализа
        default_behavior_features = [
            'mean_price', 'std_price', 'count', 'min_price', 'max_price',
            'mean_freight', 'std_freight', 'category_count', 'price_volatility',
            'price_range_ratio', 'freight_to_price_ratio',
            'unique_categories', 'category_diversity', 'normalized_diversity'
        ]
        
        # Используем предоставленные features или дефолтные, если features is None
        current_features = features if features is not None else default_behavior_features
        
        super().__init__(
            features=current_features, 
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state, # Передаем random_state напрямую
            model_name=model_name
        )
        # self.features будет установлен в super().__init__
        # self.random_state также не нужен здесь, он обрабатывается в super
        logger.info(f"BehaviorIsolationForestDetector '{self.model_name}' инициализирован с признаками: {self.features}")

    def _prepare_data_for_behavior(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует данные по seller_id и вычисляет поведенческие признаки.
        """
        df = data.copy()

        # 1. Агрегация данных, если это сырые транзакционные данные
        # (Это дублирует логику из _prepare_behavior_data в multilevel_detector, но делает детектор автономным)
        if 'seller_id' in df.columns and 'mean_price' not in df.columns: # Признак агрегированных данных
            logger.info(f"[{self.model_name}] Входные данные выглядят как сырые. Выполняется агрегация по seller_id...")
            
            required_cols_for_agg = ['seller_id', 'price', 'freight_value', 'product_category_name']
            if not all(col in df.columns for col in required_cols_for_agg):
                missing_cols = [col for col in required_cols_for_agg if col not in df.columns]
                raise ValueError(f"Для агрегации в {self.model_name} отсутствуют столбцы: {missing_cols}")

            seller_agg = df.groupby('seller_id').agg(
                mean_price=('price', 'mean'),
                std_price=('price', 'std'),
                count=('price', 'count'),
                min_price=('price', 'min'),
                max_price=('price', 'max'),
                mean_freight=('freight_value', 'mean'),
                categories_list=('product_category_name', lambda x: list(x.unique()))
            ).reset_index()

            seller_agg['price_volatility'] = seller_agg['std_price'] / seller_agg['mean_price'].replace(0, np.nan)
            seller_agg['price_range_ratio'] = seller_agg['max_price'] / seller_agg['min_price'].replace(0, np.nan)
            seller_agg['freight_to_price_ratio'] = seller_agg['mean_freight'] / seller_agg['mean_price'].replace(0, np.nan)
            seller_agg['unique_categories'] = seller_agg['categories_list'].apply(len)
            seller_agg['category_diversity'] = seller_agg['unique_categories'] / seller_agg['count'].replace(0, np.nan)
            
            df_processed = seller_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            # Данные, вероятно, уже агрегированы
            df_processed = df

        # 2. Выбор только тех признаков, которые есть в данных и в self.features
        available_features = [f for f in self.features if f in df_processed.columns]
        if not available_features:
            raise ValueError(f"Ни один из заданных признаков {self.features} не найден в данных после агрегации.")
        
        # Используем локальную переменную для текущей операции, не изменяя self.features
        current_features_to_use = available_features 
        
        # Логируем, какие признаки будут использованы
        if set(current_features_to_use) != set(self.features):
            logger.warning(f"[{self.model_name}] Не все изначально сконфигурированные признаки {self.features} доступны. "
                           f"Будут использованы: {current_features_to_use}")
        
        # Передаем в базовый preprocess только выбранные существующие признаки
        # Важно: базовый IsolationForestDetector.preprocess ожидает, что его self.features уже установлены правильно
        # Мы здесь подготавливаем df, который содержит ТОЛЬКО нужные колонки для базового preprocess
        # Базовый preprocess затем возьмет df[self.features] - поэтому self.features должен быть уже корректным
        # Но мы изменили логику так, чтобы передавать df ТОЛЬКО с нужными колонками.
        # Поэтому базовый self.features должен быть списком current_features_to_use на время вызова super().preprocess
        
        # Временно сохраняем оригинальные self.features, если они будут меняться в super().preprocess
        # original_self_features = self.features
        # self.features = current_features_to_use # ВРЕМЕННО МЕНЯЕМ self.features для вызова super().preprocess
        
        # Вызываем базовый preprocess, он использует self.features (которые мы сейчас ВРЕМЕННО изменили)
        # и масштабирует df_processed[self.features]
        # processed_for_super = super().preprocess(df_processed[current_features_to_use], fit_scaler=fit_scaler)
        
        # self.features = original_self_features # ВОССТАНАВЛИВАЕМ self.features
        # return processed_for_super

        # ---- УПРОЩЕННЫЙ ПОДХОД ----
        # Базовый IsolationForestDetector.preprocess сам выбирает self.features из DataFrame.
        # Нам нужно только передать ему DataFrame, содержащий ВСЕ ПОТЕНЦИАЛЬНЫЕ признаки.
        # Атрибут self.features экземпляра должен содержать те признаки, которые МОДЕЛЬ БЫЛА ОБУЧЕНА ИСПОЛЬЗОВАТЬ.
        # При вызове preprocess с fit_scaler=True, базовый класс обновит свой self.features.
        
        # Передаем в базовый preprocess DataFrame только с доступными из self.features колонками.
        # Базовый класс уже имеет логику выбора self.features из доступных в DataFrame.
        final_df_for_base_preprocess = df_processed[current_features_to_use]

        return super().preprocess(final_df_for_base_preprocess, fit_scaler=fit_scaler) 