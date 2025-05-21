"""
time_series_detectors.py - Детекторы аномалий для уровня временных рядов
=================================================================
Модуль содержит реализации детекторов для обнаружения аномалий
во временных рядах (необычные всплески или падения продаж).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import joblib
import os

from .detector import AnomalyDetector

# Настраиваем логирование
logger = logging.getLogger(__name__)

class SeasonalDeviationDetector(AnomalyDetector):
    """
    Детектор аномалий на основе отклонений от сезонного паттерна.
    
    Обнаруживает аномальные точки во временных рядах, которые
    сильно отклоняются от ожидаемых сезонных значений.
    """
    
    def __init__(self, 
                 window_size: int = 7, 
                 threshold: float = 3.0,
                 model_name: str = "seasonal_deviation"):
        """
        Инициализирует детектор отклонений от сезонного паттерна.
        
        Args:
            window_size: Размер окна для скользящих статистик
            threshold: Пороговое значение для определения аномалий (в стандартных отклонениях)
            model_name: Уникальное имя модели
        """
        super().__init__(model_name=model_name)
        self.window_size = window_size
        self.threshold = threshold
        
        # Статистики временного ряда
        self.time_series_stats = {}
        
        # Базовые сезонные паттерны
        self.daily_pattern = None
        self.weekly_pattern = None
        self.monthly_pattern = None
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора временных рядов.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        # Проверяем, есть ли уже агрегированные данные по времени
        if 'date' in df.columns and 'total_sales' in df.columns:
            # Данные уже агрегированы, просто проверяем формат datetime
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            return df
            
        # Проверяем наличие необходимых столбцов
        if 'order_purchase_timestamp' not in df.columns:
            raise ValueError("Для работы SeasonalDeviationDetector необходим столбец 'order_purchase_timestamp'")
            
        # Конвертируем в datetime, если нужно
        if not pd.api.types.is_datetime64_dtype(df['order_purchase_timestamp']):
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        # Создаем столбец для даты (без времени)
        df['date'] = df['order_purchase_timestamp'].dt.date
        
        # Агрегируем данные по дням
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Базовые колонки для агрегации
        agg_dict = {
            'order_id': 'nunique'  # Количество уникальных заказов в день
        }
        
        # Добавляем агрегацию для числовых колонок, если они есть
        if 'price' in numeric_columns:
            agg_dict['price'] = 'sum'  # Общая сумма продаж в день
        
        # Группируем данные по дням и получаем агрегаты
        time_series = df.groupby('date').agg(agg_dict)
        
        # Переименовываем колонки
        time_series = time_series.rename(columns={
            'order_id': 'order_count',
            'price': 'total_sales'
        })
        
        # Сбрасываем индекс для удобства работы
        time_series = time_series.reset_index()
        
        # Конвертируем дату в datetime
        time_series['date'] = pd.to_datetime(time_series['date'])
        
        return time_series
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: вычисление статистик и сезонных паттернов.
        
        Args:
            data: DataFrame с обучающими данными
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        
        # Предобработка данных
        df = self.preprocess(data)
        
        if df.empty:
            raise ValueError("После предобработки данных получен пустой DataFrame")
        
        # Проверяем наличие необходимых колонок
        if 'date' not in df.columns or ('total_sales' not in df.columns and 'order_count' not in df.columns):
            raise ValueError("Отсутствуют необходимые колонки для обучения детектора временных рядов")
        
        # Используем имеющуюся колонку в качестве основного показателя
        main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Вычисляем базовые статистики временного ряда
        self.time_series_stats = {
            'metric': main_metric,
            'mean': df[main_metric].mean(),
            'std': df[main_metric].std(),
            'median': df[main_metric].median(),
            'min': df[main_metric].min(),
            'max': df[main_metric].max(),
            'start_date': df['date'].min(),
            'end_date': df['date'].max()
        }
        
        # Вычисляем скользящие статистики
        df['rolling_mean'] = df[main_metric].rolling(window=self.window_size, center=True, min_periods=1).mean()
        df['rolling_std'] = df[main_metric].rolling(window=self.window_size, center=True, min_periods=1).std()
        
        # Заполняем NaN медианой
        df['rolling_std'] = df['rolling_std'].fillna(df['rolling_std'].median())
        
        # Если есть достаточно данных, вычисляем сезонные паттерны
        if len(df) >= 14:  # Хотя бы 2 недели данных
            # Дневной паттерн (день недели)
            if 'date' in df.columns:
                df['day_of_week'] = df['date'].dt.dayofweek
                self.weekly_pattern = df.groupby('day_of_week')[main_metric].mean().to_dict()
                
            # Месячный паттерн
            if 'date' in df.columns:
                df['day_of_month'] = df['date'].dt.day
                self.monthly_pattern = df.groupby('day_of_month')[main_metric].mean().to_dict()
        
        # Обучаем нормализатор скоров
        # Вычисляем Z-скоры отклонений от скользящего среднего
        z_scores = np.abs((df[main_metric] - df['rolling_mean']) / df['rolling_std'].replace(0, 0.001))
        
        # Обучаем нормализатор
        self.fit_normalizer(z_scores.values)
        
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: проанализировано {len(df)} временных точек")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий во временном ряду.
        
        Args:
            data: DataFrame с временным рядом для анализа
            
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
        
        # Проверяем наличие необходимой колонки
        main_metric = self.time_series_stats.get('metric', 'total_sales')
        if main_metric not in df.columns:
            logger.warning(f"Метрика {main_metric} не найдена в данных. Используем доступную метрику.")
            main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Вычисляем скользящие статистики
        df['rolling_mean'] = df[main_metric].rolling(window=self.window_size, center=True, min_periods=1).mean()
        df['rolling_std'] = df[main_metric].rolling(window=self.window_size, center=True, min_periods=1).std()
        
        # Заполняем NaN медианой
        df['rolling_std'] = df['rolling_std'].fillna(df['rolling_std'].median())
        
        # Добавляем ожидаемое значение на основе сезонного паттерна, если он есть
        if self.weekly_pattern is not None and 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['seasonal_expected'] = df['day_of_week'].map(self.weekly_pattern)
            
            # Если есть пропуски, используем скользящее среднее
            df['seasonal_expected'] = df['seasonal_expected'].fillna(df['rolling_mean'])
            
            # Вычисляем отклонение от сезонного ожидания
            df['seasonal_deviation'] = np.abs((df[main_metric] - df['seasonal_expected']) / df['rolling_std'].replace(0, 0.001))
        else:
            # Если нет сезонного паттерна, используем отклонение от скользящего среднего
            df['seasonal_deviation'] = np.abs((df[main_metric] - df['rolling_mean']) / df['rolling_std'].replace(0, 0.001))
        
        # Добавляем скоры в результат
        df['anomaly_score'] = df['seasonal_deviation']
        
        # Нормализуем скоры
        normalized_scores = self.normalize_score(df['anomaly_score'].values)
        df['normalized_score'] = normalized_scores
        
        # Определяем аномалии
        df['is_anomaly'] = df['anomaly_score'] > self.threshold
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} временных точек")
        
        return df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты SeasonalDeviationDetector для сохранения."""
        # min_score_, max_score_ сохраняются базовым классом.
        # Статистики и паттерны объединяем в 'model_state'.
        # Параметры конструктора сохраняем отдельно.
        return {
            'model_state': { 
                'time_series_stats': self.time_series_stats,
                'daily_pattern': self.daily_pattern,
                'weekly_pattern': self.weekly_pattern,
                'monthly_pattern': self.monthly_pattern
            },
            'window_size': self.window_size,
            'threshold': self.threshold
            # scaler здесь None.
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты SeasonalDeviationDetector."""
        # Базовый load_model уже попытался загрузить self.model (из 'model_state'), 
        # self.scaler (None), self.min_score_, self.max_score_.

        # Распаковываем данные из self.model (если он был загружен из model_state)
        if self.model and isinstance(self.model, dict):
            self.time_series_stats = self.model.get('time_series_stats', {})
            self.daily_pattern = self.model.get('daily_pattern')
            self.weekly_pattern = self.model.get('weekly_pattern')
            self.monthly_pattern = self.model.get('monthly_pattern')
        else:
            # Fallback, если 'model_state' не было или было некорректным
            # Пытаемся загрузить по старым индивидуальным ключам, если они есть
            logger.info(f"'{self.model_name}': 'model_state' не найден или некорректен. Попытка загрузить атрибуты индивидуально.")
            self.time_series_stats = loaded_data.get('time_series_stats', {})
            self.daily_pattern = loaded_data.get('daily_pattern')
            self.weekly_pattern = loaded_data.get('weekly_pattern')
            self.monthly_pattern = loaded_data.get('monthly_pattern')
            
            if not self.time_series_stats and not self.daily_pattern and not self.weekly_pattern and not self.monthly_pattern:
                logger.warning(f"Не удалось загрузить какие-либо статистики или паттерны для {self.model_name}.")

        # Загружаем параметры конструктора
        self.window_size = loaded_data.get('window_size', self.window_size)
        self.threshold = loaded_data.get('threshold', self.threshold)

        # Определяем is_trained
        # Обучен, если есть time_series_stats (хотя бы) и параметры нормализатора.
        # Паттерны могут быть None, если данных было недостаточно для их вычисления.
        valid_stats_loaded = isinstance(self.time_series_stats, dict) and bool(self.time_series_stats)
        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_stats_loaded and normalizer_loaded:
            self.is_trained = True
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_stats_loaded:
                missing_parts_log.append("time_series_stats")
            if not normalizer_loaded:
                missing_parts_log.append("параметры нормализации")
            
            if missing_parts_log:
                logger.warning(f"{self.model_name} не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")
        
        self._custom_is_trained_logic_applied_in_load_additional = True

    def get_explanation_details(self, data_for_explanation: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Предоставляет подробное объяснение для конкретной аномалии.
        
        Args:
            data_for_explanation: DataFrame с данными для объяснения (обычно одна строка)
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None при ошибке
        """
        if not self.is_trained:
            logger.warning(f"({self.model_name}) Модель не обучена. Невозможно предоставить объяснение.")
            return None
        
        try:
            explanations = []
            
            for _, row in data_for_explanation.iterrows():
                # Определяем основную метрику
                main_metric = self.time_series_stats.get('metric', 'total_sales')
                date_value = None
                
                # Извлекаем дату из row
                if 'date' in row:
                    date_value = pd.to_datetime(row['date'])
                elif 'order_purchase_timestamp' in row:
                    # Если есть только timestamp заказа, используем его дату
                    date_value = pd.to_datetime(row['order_purchase_timestamp']).date()
                
                if date_value is None:
                    logger.warning(f"({self.model_name}) Не удалось определить дату для объяснения.")
                    continue
                
                # Вычисляем день недели для проверки сезонного паттерна
                day_of_week = date_value.dayofweek
                
                # Получаем текущее значение метрики
                current_value = None
                if main_metric in row:
                    current_value = row[main_metric]
                
                # Получаем ожидаемое значение на основе сезонного паттерна
                expected_value = None
                if self.weekly_pattern is not None and day_of_week in self.weekly_pattern:
                    expected_value = self.weekly_pattern[day_of_week]
                
                # Если нет текущего значения, попробуем взять данные из агрегированного временного ряда
                if current_value is None or expected_value is None:
                    # Здесь может потребоваться доступ к сохраненным временным рядам,
                    # но мы предполагаем, что основные метрики уже доступны в row
                    logger.warning(f"({self.model_name}) Недостаточно данных для детального объяснения.")
                    explanation = {
                        "detector_specific_info": {
                            "metric_name": main_metric,
                            "date": str(date_value),
                            "threshold": float(self.threshold),
                            "explanation_text": f"Недостаточно данных для детального анализа аномалии на дату {date_value}."
                        }
                    }
                    explanations.append(explanation)
                    continue
                
                # Вычисляем отклонение
                deviation = current_value - expected_value if expected_value != 0 else 0
                relative_deviation = deviation / expected_value if expected_value != 0 else float('inf')
                
                # Определяем причину аномалии
                explanation_text = ""
                anomaly_factors = []
                
                if abs(relative_deviation) > self.threshold:
                    if relative_deviation > 0:
                        explanation_text = f"Значение {main_metric} ({current_value:.2f}) на {(relative_deviation*100):.1f}% выше ожидаемого сезонного значения ({expected_value:.2f})."
                        anomaly_factors.append({
                            "factor": "seasonal_high",
                            "current_value": float(current_value),
                            "expected_value": float(expected_value),
                            "deviation_percent": float(relative_deviation*100),
                            "description": explanation_text
                        })
                    else:
                        explanation_text = f"Значение {main_metric} ({current_value:.2f}) на {(abs(relative_deviation)*100):.1f}% ниже ожидаемого сезонного значения ({expected_value:.2f})."
                        anomaly_factors.append({
                            "factor": "seasonal_low",
                            "current_value": float(current_value),
                            "expected_value": float(expected_value),
                            "deviation_percent": float(-relative_deviation*100),
                            "description": explanation_text
                        })
                else:
                    explanation_text = f"Для даты {date_value} значение {main_metric} ({current_value:.2f}) в пределах нормальных отклонений от сезонного паттерна ({expected_value:.2f})."
                
                # Формируем детальное объяснение
                explanation = {
                    "detector_specific_info": {
                        "metric_name": main_metric,
                        "date": str(date_value),
                        "current_value": float(current_value) if current_value is not None else None,
                        "expected_value": float(expected_value) if expected_value is not None else None,
                        "absolute_deviation": float(deviation) if deviation is not None else None,
                        "relative_deviation": float(relative_deviation) if not pd.isna(relative_deviation) and relative_deviation != float('inf') else None,
                        "threshold": float(self.threshold),
                        "day_of_week": int(day_of_week),
                        "anomaly_factors": anomaly_factors,
                        "explanation_text": explanation_text
                    }
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснения: {e}", exc_info=True)
            return None

class MovingAverageVolatilityDetector(AnomalyDetector):
    """
    Детектор аномалий на основе волатильности вокруг скользящей средней.
    
    Выявляет периоды с необычно высокой волатильностью относительно
    скользящего среднего, что может указывать на аномальные всплески
    или падения в продажах.
    """
    
    def __init__(self, 
                 long_window: int = 30, 
                 short_window: int = 7,
                 volatility_threshold: float = 2.0,
                 min_periods: int = 5,
                 model_name: str = "moving_average_volatility"):
        """
        Инициализирует детектор волатильности относительно скользящей средней.
        
        Args:
            long_window: Размер длинного окна для скользящей средней (базовый тренд)
            short_window: Размер короткого окна для волатильности
            volatility_threshold: Пороговое значение для определения аномальной волатильности
            min_periods: Минимальное количество точек для вычисления статистик
            model_name: Уникальное имя модели
        """
        super().__init__(model_name=model_name)
        self.long_window = long_window
        self.short_window = short_window
        self.volatility_threshold = volatility_threshold
        self.min_periods = min_periods
        
        # Статистики волатильности
        self.series_stats = {}
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора волатильности.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        # Проверяем, есть ли уже агрегированные данные по времени
        if 'date' in df.columns and ('total_sales' in df.columns or 'order_count' in df.columns):
            # Данные уже агрегированы, просто проверяем формат datetime
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            return df
            
        # Проверяем наличие необходимых столбцов
        if 'order_purchase_timestamp' not in df.columns:
            raise ValueError("Для работы MovingAverageVolatilityDetector необходим столбец 'order_purchase_timestamp'")
            
        # Конвертируем в datetime, если нужно
        if not pd.api.types.is_datetime64_dtype(df['order_purchase_timestamp']):
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        # Создаем столбец для даты (без времени)
        df['date'] = df['order_purchase_timestamp'].dt.date
        
        # Агрегируем данные по дням
        time_series = df.groupby('date').agg(
            order_count=('order_id', 'nunique'),
            total_sales=('price', 'sum') if 'price' in df.columns else ('order_id', 'count')
        ).reset_index()
        
        # Конвертируем дату в datetime
        time_series['date'] = pd.to_datetime(time_series['date'])
        
        return time_series
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: вычисление статистик волатильности.
        
        Args:
            data: DataFrame с обучающими данными
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        
        # Предобработка данных
        df = self.preprocess(data)
        
        if df.empty:
            raise ValueError("После предобработки данных получен пустой DataFrame")
        
        # Проверяем наличие необходимых колонок
        if 'date' not in df.columns or ('total_sales' not in df.columns and 'order_count' not in df.columns):
            raise ValueError("Отсутствуют необходимые колонки для обучения детектора волатильности")
        
        # Используем имеющуюся колонку в качестве основного показателя
        main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Вычисляем скользящие средние
        df['long_ma'] = df[main_metric].rolling(window=self.long_window, min_periods=self.min_periods).mean()
        df['short_ma'] = df[main_metric].rolling(window=self.short_window, min_periods=self.min_periods).mean()
        
        # Вычисляем волатильность (стандартное отклонение в коротком окне)
        df['volatility'] = df[main_metric].rolling(window=self.short_window, min_periods=self.min_periods).std()
        
        # Вычисляем относительную волатильность (стд / среднее)
        df['relative_volatility'] = df['volatility'] / df['short_ma'].replace(0, 0.001)
        
        # Заполняем NaN
        df[['long_ma', 'short_ma', 'volatility', 'relative_volatility']] = df[['long_ma', 'short_ma', 'volatility', 'relative_volatility']].fillna(method='bfill').fillna(method='ffill')
        
        # Вычисляем статистики волатильности
        self.series_stats = {
            'metric': main_metric,
            'mean_volatility': df['volatility'].mean(),
            'std_volatility': df['volatility'].std(),
            'mean_relative_volatility': df['relative_volatility'].mean(),
            'std_relative_volatility': df['relative_volatility'].std(),
            'q75_relative_volatility': df['relative_volatility'].quantile(0.75),
            'q95_relative_volatility': df['relative_volatility'].quantile(0.95)
        }
        
        # Обучаем нормализатор скоров
        # Используем относительную волатильность как скор
        volatility_scores = df['relative_volatility'].values
        
        # Обучаем нормализатор
        self.fit_normalizer(volatility_scores)
        
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: проанализировано {len(df)} временных точек")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий волатильности во временном ряду.
        
        Args:
            data: DataFrame с временным рядом для анализа
            
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
        
        # Проверяем наличие необходимой колонки
        main_metric = self.series_stats.get('metric', 'total_sales')
        if main_metric not in df.columns:
            logger.warning(f"Метрика {main_metric} не найдена в данных. Используем доступную метрику.")
            main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Вычисляем скользящие средние и волатильность
        df['long_ma'] = df[main_metric].rolling(window=self.long_window, min_periods=self.min_periods).mean()
        df['short_ma'] = df[main_metric].rolling(window=self.short_window, min_periods=self.min_periods).mean()
        df['volatility'] = df[main_metric].rolling(window=self.short_window, min_periods=self.min_periods).std()
        df['relative_volatility'] = df['volatility'] / df['short_ma'].replace(0, 0.001)
        
        # Заполняем NaN
        df[['long_ma', 'short_ma', 'volatility', 'relative_volatility']] = df[['long_ma', 'short_ma', 'volatility', 'relative_volatility']].fillna(method='bfill').fillna(method='ffill')
        
        # Вычисляем Z-скоры относительно статистик обучающей выборки
        z_volatility = (df['relative_volatility'] - self.series_stats['mean_relative_volatility']) / self.series_stats['std_relative_volatility']
        
        # Используем относительную волатильность как скор аномалии
        df['anomaly_score'] = df['relative_volatility']
        
        # Нормализуем скоры
        normalized_scores = self.normalize_score(df['anomaly_score'].values)
        df['normalized_score'] = normalized_scores
        
        # Определяем аномалии на основе порога
        q95_threshold = self.series_stats['q95_relative_volatility']
        df['is_anomaly'] = df['relative_volatility'] > q95_threshold
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} временных точек")
        
        return df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты MovingAverageVolatilityDetector для сохранения."""
        return {
            'model_state': self.series_stats, # series_stats как основное состояние модели
            # min_score_, max_score_ сохраняются базовым классом.
            'long_window': self.long_window,
            'short_window': self.short_window,
            'volatility_threshold': self.volatility_threshold,
            'min_periods': self.min_periods
            # scaler здесь None.
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты MovingAverageVolatilityDetector."""
        # Базовый load_model уже попытался загрузить self.model (из 'model_state'), 
        # self.scaler (None), self.min_score_, self.max_score_.

        if self.model and isinstance(self.model, dict):
            self.series_stats = self.model # model_state содержит series_stats
        else:
            # Fallback, если 'model_state' не было или было некорректным
            logger.info(f"'{self.model_name}': 'model_state' не найден или некорректен. Попытка загрузить 'series_stats' индивидуально.")
            self.series_stats = loaded_data.get('series_stats', {})
            if not self.series_stats:
                 logger.warning(f"Не удалось загрузить 'series_stats' для {self.model_name}.")

        # Загружаем параметры конструктора
        self.long_window = loaded_data.get('long_window', self.long_window)
        self.short_window = loaded_data.get('short_window', self.short_window)
        self.volatility_threshold = loaded_data.get('volatility_threshold', self.volatility_threshold)
        self.min_periods = loaded_data.get('min_periods', self.min_periods)

        # Определяем is_trained
        valid_stats_loaded = isinstance(self.series_stats, dict) and bool(self.series_stats)
        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_stats_loaded and normalizer_loaded:
            self.is_trained = True
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_stats_loaded:
                missing_parts_log.append("series_stats")
            if not normalizer_loaded:
                missing_parts_log.append("параметры нормализации")
            
            if missing_parts_log:
                logger.warning(f"{self.model_name} не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")
        
        self._custom_is_trained_logic_applied_in_load_additional = True

    def get_explanation_details(self, data_for_explanation: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детальное объяснение для аномалии волатильности временного ряда.
        
        Args:
            data_for_explanation: DataFrame с данными для объяснения (обычно одна строка)
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки или None при ошибке
        """
        if not self.is_trained:
            logger.warning(f"({self.model_name}) Детектор не обучен. Невозможно предоставить объяснения.")
            return None
            
        try:
            explanations = []
            
            for _, row in data_for_explanation.iterrows():
                # Определяем основную метрику
                main_metric = self.series_stats.get('metric', 'total_sales')
                
                # Получаем значения из строки данных
                date_value = None
                current_value = None
                long_ma_value = None
                short_ma_value = None
                volatility_value = None
                
                if 'date' in row:
                    date_value = pd.to_datetime(row['date'])
                elif 'order_purchase_timestamp' in row:
                    date_value = pd.to_datetime(row['order_purchase_timestamp'])
                
                if main_metric in row:
                    current_value = row[main_metric]
                
                if 'long_ma' in row:
                    long_ma_value = row['long_ma']
                
                if 'short_ma' in row:
                    short_ma_value = row['short_ma']
                
                if 'volatility' in row:
                    volatility_value = row['volatility']
                
                # Если недостаточно данных для детального объяснения
                if date_value is None or current_value is None:
                    explanation = {
                        "detector_specific_info": {
                            "date": str(date_value) if date_value else "unknown",
                            "explanation_text": "Недостаточно данных для детального объяснения аномалии волатильности."
                        }
                    }
                    explanations.append(explanation)
                    continue
                
                # Формируем объяснение на основе доступных данных
                explanation_text = ""
                anomaly_factors = []
                
                if long_ma_value is not None and short_ma_value is not None:
                    ma_diff = short_ma_value - long_ma_value
                    ma_ratio = short_ma_value / long_ma_value if long_ma_value > 0 else 0
                    
                    if abs(ma_ratio - 1) > 0.3:  # 30% отклонение между средними
                        if ma_ratio > 1:
                            anomaly_factors.append({
                                "factor": "short_term_increase",
                                "short_ma": float(short_ma_value),
                                "long_ma": float(long_ma_value),
                                "ratio": float(ma_ratio),
                                "description": f"Краткосрочное среднее ({short_ma_value:.2f}) на {(ma_ratio-1)*100:.1f}% выше долгосрочного тренда ({long_ma_value:.2f})"
                            })
                        else:
                            anomaly_factors.append({
                                "factor": "short_term_decrease",
                                "short_ma": float(short_ma_value),
                                "long_ma": float(long_ma_value),
                                "ratio": float(ma_ratio),
                                "description": f"Краткосрочное среднее ({short_ma_value:.2f}) на {(1-ma_ratio)*100:.1f}% ниже долгосрочного тренда ({long_ma_value:.2f})"
                            })
                
                if volatility_value is not None and volatility_value > self.volatility_threshold:
                    anomaly_factors.append({
                        "factor": "high_volatility",
                        "volatility": float(volatility_value),
                        "threshold": float(self.volatility_threshold),
                        "description": f"Высокая волатильность {main_metric} ({volatility_value:.2f}) превышает порог ({self.volatility_threshold:.2f})"
                    })
                
                # Создаем итоговый текст объяснения
                if anomaly_factors:
                    explanation_text = f"Обнаружены следующие аномалии волатильности для даты {date_value.date()}:"
                    for factor in anomaly_factors:
                        explanation_text += f"\n- {factor['description']}"
                else:
                    explanation_text = f"Для даты {date_value.date()} специфических факторов аномальности волатильности не выявлено, но общий паттерн нетипичен."
                
                # Формируем итоговое объяснение
                explanation = {
                    "detector_specific_info": {
                        "date": str(date_value.date()),
                        "metric_name": main_metric,
                        "current_value": float(current_value) if current_value is not None else None,
                        "long_ma": float(long_ma_value) if long_ma_value is not None else None,
                        "short_ma": float(short_ma_value) if short_ma_value is not None else None,
                        "volatility": float(volatility_value) if volatility_value is not None else None,
                        "volatility_threshold": float(self.volatility_threshold),
                        "anomaly_factors": anomaly_factors,
                        "text_explanation": explanation_text
                    }
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None

class CumulativeSumDetector(AnomalyDetector):
    """
    Детектор аномалий на основе кумулятивной суммы отклонений.
    
    Обнаруживает устойчивые изменения в поведении временного ряда,
    выявляя точки, где начинаются систематические отклонения от нормы.
    """
    
    def __init__(self, 
                 window_size: int = 14, 
                 threshold: float = 3.0,
                 drift_threshold: float = 5.0,
                 model_name: str = "cumulative_sum"):
        """
        Инициализирует детектор кумулятивной суммы отклонений.
        
        Args:
            window_size: Размер окна для расчета базовой статистики
            threshold: Пороговое значение для определения аномальных отклонений
            drift_threshold: Пороговое значение для определения устойчивого дрифта
            model_name: Уникальное имя модели
        """
        super().__init__(model_name=model_name)
        self.window_size = window_size
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        
        # Базовые статистики
        self.baseline_mean = None
        self.baseline_std = None
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора кумулятивной суммы.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        df = data.copy()
        
        # Проверяем, есть ли уже агрегированные данные по времени
        if 'date' in df.columns and ('total_sales' in df.columns or 'order_count' in df.columns):
            # Данные уже агрегированы, просто проверяем формат datetime
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            return df
            
        # Проверяем наличие необходимых столбцов
        if 'order_purchase_timestamp' not in df.columns:
            raise ValueError("Для работы CumulativeSumDetector необходим столбец 'order_purchase_timestamp'")
            
        # Конвертируем в datetime, если нужно
        if not pd.api.types.is_datetime64_dtype(df['order_purchase_timestamp']):
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        # Создаем столбец для даты (без времени)
        df['date'] = df['order_purchase_timestamp'].dt.date
        
        # Агрегируем данные по дням
        time_series = df.groupby('date').agg(
            order_count=('order_id', 'nunique'),
            total_sales=('price', 'sum') if 'price' in df.columns else ('order_id', 'count')
        ).reset_index()
        
        # Конвертируем дату в datetime
        time_series['date'] = pd.to_datetime(time_series['date'])
        
        return time_series
    
    def train(self, data: pd.DataFrame):
        """
        Обучение модели: определение базовой статистики.
        
        Args:
            data: DataFrame с обучающими данными
        """
        logger.info(f"Обучение детектора {self.model_name}...")
        
        # Предобработка данных
        df = self.preprocess(data)
        
        if df.empty:
            raise ValueError("После предобработки данных получен пустой DataFrame")
        
        # Проверяем наличие необходимых колонок
        if 'date' not in df.columns or ('total_sales' not in df.columns and 'order_count' not in df.columns):
            raise ValueError("Отсутствуют необходимые колонки для обучения детектора кумулятивной суммы")
        
        # Используем имеющуюся колонку в качестве основного показателя
        main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Определяем базовую статистику (используем первое окно данных)
        baseline_data = df.head(self.window_size)
        self.baseline_mean = baseline_data[main_metric].mean()
        self.baseline_std = baseline_data[main_metric].std()
        
        if self.baseline_std < 0.001:
            self.baseline_std = 0.001  # Избегаем деления на очень маленькие значения
        
        # Вычисляем отклонения от базовой средней
        df['deviation'] = (df[main_metric] - self.baseline_mean) / self.baseline_std
        
        # Вычисляем кумулятивную сумму положительных и отрицательных отклонений
        df['pos_cusum'] = 0.0
        df['neg_cusum'] = 0.0
        
        pos_cusum = 0
        neg_cusum = 0
        
        for i, row in df.iterrows():
            # Положительное отклонение (превышение среднего)
            pos_cusum = max(0, pos_cusum + row['deviation'])
            # Отрицательное отклонение (ниже среднего)
            neg_cusum = max(0, neg_cusum - row['deviation'])
            
            df.at[i, 'pos_cusum'] = pos_cusum
            df.at[i, 'neg_cusum'] = neg_cusum
        
        # Объединяем положительные и отрицательные отклонения в один скор
        df['cusum_score'] = np.maximum(df['pos_cusum'], df['neg_cusum'])
        
        # Обучаем нормализатор скоров
        self.fit_normalizer(df['cusum_score'].values)
        
        self.is_trained = True
        logger.info(f"Детектор {self.model_name} обучен: базовое среднее = {self.baseline_mean:.2f}, стд = {self.baseline_std:.2f}")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий с использованием метода кумулятивной суммы.
        
        Args:
            data: DataFrame с временным рядом для анализа
            
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
        
        # Проверяем наличие необходимой колонки
        main_metric = 'total_sales' if 'total_sales' in df.columns else 'order_count'
        
        # Сортируем по дате
        df = df.sort_values('date')
        
        # Вычисляем отклонения от базовой средней
        df['deviation'] = (df[main_metric] - self.baseline_mean) / self.baseline_std
        
        # Вычисляем кумулятивную сумму положительных и отрицательных отклонений
        df['pos_cusum'] = 0.0
        df['neg_cusum'] = 0.0
        
        pos_cusum = 0
        neg_cusum = 0
        
        for i, row in df.iterrows():
            # Положительное отклонение (превышение среднего)
            pos_cusum = max(0, pos_cusum + row['deviation'])
            # Отрицательное отклонение (ниже среднего)
            neg_cusum = max(0, neg_cusum - row['deviation'])
            
            df.at[i, 'pos_cusum'] = pos_cusum
            df.at[i, 'neg_cusum'] = neg_cusum
        
        # Объединяем положительные и отрицательные отклонения в один скор
        df['anomaly_score'] = np.maximum(df['pos_cusum'], df['neg_cusum'])
        
        # Нормализуем скоры
        normalized_scores = self.normalize_score(df['anomaly_score'].values)
        df['normalized_score'] = normalized_scores
        
        # Определяем аномалии:
        # 1. Точки, где кумулятивная сумма превышает порог
        df['is_anomaly'] = df['anomaly_score'] > self.threshold
        
        # 2. Точки, где начинается устойчивый дрифт (резкое изменение кумулятивной суммы)
        df['cusum_diff'] = df['anomaly_score'].diff().fillna(0)
        drift_points = df['cusum_diff'] > (self.drift_threshold * df['cusum_diff'].std())
        df.loc[drift_points, 'is_anomaly'] = True
        
        logger.info(f"Детектор {self.model_name}: обнаружено {df['is_anomaly'].sum()} аномалий из {len(df)} временных точек")
        
        return df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты CumulativeSumDetector для сохранения."""
        return {
            'model_state': { 
                'baseline_mean': self.baseline_mean,
                'baseline_std': self.baseline_std 
            },
            'window_size': self.window_size,
            'threshold': self.threshold,
            'drift_threshold': self.drift_threshold
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты CumulativeSumDetector."""
        model_state = loaded_data.get('model_state', {}) # self.model может быть установлен базовым классом
        if isinstance(model_state, dict):
            self.baseline_mean = model_state.get('baseline_mean')
            self.baseline_std = model_state.get('baseline_std')
            logger.info(f"({self.model_name}) Загружены baseline_mean={self.baseline_mean}, baseline_std={self.baseline_std} из model_state.")
        else:
            logger.warning(f"({self.model_name}) 'model_state' не найден или некорректен в loaded_data. Попытка загрузить атрибуты baseline_mean/std индивидуально.")
            # Для совместимости со старыми моделями, где атрибуты могли сохраняться напрямую
            self.baseline_mean = loaded_data.get('baseline_mean', self.baseline_mean) 
            self.baseline_std = loaded_data.get('baseline_std', self.baseline_std)
            if self.baseline_mean is None or self.baseline_std is None:
                 logger.warning(f"({self.model_name}) Не удалось загрузить baseline_mean или baseline_std из корневого уровня loaded_data.")
        
        self.window_size = loaded_data.get('window_size', self.window_size)
        self.threshold = loaded_data.get('threshold', self.threshold)
        self.drift_threshold = loaded_data.get('drift_threshold', self.drift_threshold)

        # min_score_ и max_score_ должны быть загружены базовым AnomalyDetector.load_model
        valid_baseline_stats = self.baseline_mean is not None and self.baseline_std is not None
        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_baseline_stats and normalizer_loaded:
            self.is_trained = True
            logger.info(f"({self.model_name}) Детектор CUSUM успешно загружен и помечен как обученный.")
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_baseline_stats:
                missing_parts_log.append("статистики baseline (mean/std)")
            if not normalizer_loaded:
                missing_parts_log.append("параметры нормализации (min_score_/max_score_)")
            
            if missing_parts_log:
                logger.warning(f"({self.model_name}) не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")
        
        self._custom_is_trained_logic_applied_in_load_additional = True

    def get_explanation_details(self, data_for_explanation: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Предоставляет подробное объяснение для конкретной аномалии, обнаруженной CUSUM детектором.
        
        Args:
            data_for_explanation: DataFrame с данными для объяснения (обычно одна строка)
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None при ошибке
        """
        if not self.is_trained or self.baseline_mean is None:
            logger.warning(f"({self.model_name}) Модель не обучена или отсутствует baseline_mean. Невозможно предоставить объяснение.")
            return None
        
        try:
            explanations = []
            
            for _, row in data_for_explanation.iterrows():
                # Определяем основную метрику
                main_metric = 'total_sales'  # По умолчанию, можно было бы сохранять в self при обучении
                date_value = None
                
                # Извлекаем дату из row
                if 'date' in row:
                    date_value = pd.to_datetime(row['date'])
                elif 'order_purchase_timestamp' in row:
                    # Если есть только timestamp заказа, используем его дату
                    date_value = pd.to_datetime(row['order_purchase_timestamp']).date()
                
                if date_value is None:
                    logger.warning(f"({self.model_name}) Не удалось определить дату для объяснения.")
                    continue
                
                # Получаем текущее значение метрики, cumulative sum и другие данные
                current_value = None
                pos_cusum = None
                neg_cusum = None
                
                if main_metric in row:
                    current_value = row[main_metric]
                
                if 'pos_cusum' in row:
                    pos_cusum = row['pos_cusum']
                
                if 'neg_cusum' in row:
                    neg_cusum = row['neg_cusum']
                
                # Если не хватает данных для объяснения
                if current_value is None:
                    logger.warning(f"({self.model_name}) Недостаточно данных для детального объяснения.")
                    explanation = {
                        "detector_specific_info": {
                            "date": str(date_value),
                            "baseline_mean": float(self.baseline_mean),
                            "baseline_std": float(self.baseline_std),
                            "threshold": float(self.threshold),
                            "explanation_text": f"Недостаточно данных для детального анализа CUSUM аномалии на дату {date_value}."
                        }
                    }
                    explanations.append(explanation)
                    continue
                
                # Вычисляем отклонение от базовой линии
                deviation = current_value - self.baseline_mean
                deviation_in_std = deviation / self.baseline_std if self.baseline_std != 0 else 0
                
                # Определяем тип аномалии (позитивное или негативное отклонение)
                triggered_cusum = None
                max_cusum_value = 0
                anomaly_factors = []
                
                if pos_cusum is not None and neg_cusum is not None:
                    max_cusum_value = max(pos_cusum, neg_cusum)
                    if pos_cusum > self.threshold and pos_cusum >= neg_cusum:
                        triggered_cusum = "positive"
                        anomaly_factors.append({
                            "factor": "positive_cusum",
                            "pos_cusum": float(pos_cusum),
                            "threshold": float(self.threshold),
                            "description": f"Накопленная сумма положительных отклонений ({pos_cusum:.2f}) превысила порог ({self.threshold:.2f})"
                        })
                    elif neg_cusum > self.threshold and neg_cusum > pos_cusum:
                        triggered_cusum = "negative"
                        anomaly_factors.append({
                            "factor": "negative_cusum",
                            "neg_cusum": float(neg_cusum),
                            "threshold": float(self.threshold),
                            "description": f"Накопленная сумма отрицательных отклонений ({neg_cusum:.2f}) превысила порог ({self.threshold:.2f})"
                        })
                
                # Формируем текст объяснения
                explanation_text = ""
                if triggered_cusum == "positive":
                    explanation_text = f"Обнаружено устойчивое положительное отклонение от среднего. Текущее значение ({current_value:.2f}) на {deviation:.2f} выше базовой линии ({self.baseline_mean:.2f}). Накопленный скор CUSUM+ достиг {pos_cusum:.2f}, что превышает порог {self.threshold:.2f}."
                elif triggered_cusum == "negative":
                    explanation_text = f"Обнаружено устойчивое отрицательное отклонение от среднего. Текущее значение ({current_value:.2f}) на {abs(deviation):.2f} ниже базовой линии ({self.baseline_mean:.2f}). Накопленный скор CUSUM- достиг {neg_cusum:.2f}, что превышает порог {self.threshold:.2f}."
                else:
                    explanation_text = f"Значение ({current_value:.2f}) отклоняется от базовой линии ({self.baseline_mean:.2f}) на {deviation:.2f}, но накопленный эффект не достаточен для объявления аномалии."
                
                # Формируем детальное объяснение
                explanation = {
                    "detector_specific_info": {
                        "date": str(date_value),
                        "current_value": float(current_value),
                        "baseline_mean": float(self.baseline_mean),
                        "baseline_std": float(self.baseline_std),
                        "deviation": float(deviation),
                        "deviation_in_std": float(deviation_in_std),
                        "pos_cusum": float(pos_cusum) if pos_cusum is not None else None,
                        "neg_cusum": float(neg_cusum) if neg_cusum is not None else None,
                        "max_cusum": float(max_cusum_value),
                        "threshold": float(self.threshold),
                        "triggered_cusum_type": triggered_cusum,
                        "anomaly_factors": anomaly_factors,
                        "explanation_text": explanation_text
                    }
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснения CUSUM: {e}", exc_info=True)
            return None

# Конец файла, если это последний детектор 