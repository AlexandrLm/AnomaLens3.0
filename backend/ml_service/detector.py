"""
detector.py - Базовые и конкретные классы детекторов аномалий
=============================================================
Модуль содержит определения базового класса детектора аномалий
и реализации конкретных алгоритмов обнаружения аномалий.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import joblib
import os
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional, List, Tuple, Union
import logging 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

try:
    import shap
except ImportError:
    shap = None
    print("Предупреждение: Библиотека SHAP не установлена. Функционал объяснения моделей будет недоступен.")
# -----------------------------

logger = logging.getLogger(__name__) 

# =============================================================================
# Абстрактный базовый класс детектора аномалий
# =============================================================================

class AnomalyDetector(ABC):
    """
    Абстрактный базовый класс для всех детекторов аномалий.
    
    Определяет общий интерфейс, который должны реализовать все детекторы:
    предобработка данных, обучение, обнаружение аномалий, сохранение/загрузка модели.
    Также реализует общую функциональность нормализации скоров.
    """

    def __init__(self, model_name: str):
        """
        Инициализирует детектор аномалий.
        
        Args:
            model_name: Уникальное имя модели/детектора
        """
        self.model_name: str = model_name
        self.model: Optional[Any] = None        # Обученная модель/статистики
        self.scaler: Optional[StandardScaler] = None       # Скейлер для предобработки данных
        self.is_trained: bool = False  # Флаг обученности модели
        
        # Параметры для нормализации скоров
        self.min_score_: Optional[float] = None   # Минимальный скор на обучающей выборке
        self.max_score_: Optional[float] = None   # Максимальный скор на обучающей выборке

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для детектора.
        
        Args:
            data: Исходный DataFrame с данными
            
        Returns:
            Предобработанный DataFrame
        """
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """
        Обучение модели/вычисление статистик.
        
        Args:
            data: DataFrame с обучающими данными
        """
        pass

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в новых данных.
        
        Args:
            data: DataFrame с тестовыми данными
            
        Returns:
            DataFrame с оригинальными данными и добавленными скорами
        """
        pass

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """
        Возвращает словарь с дополнительными, специфичными для детектора атрибутами для сохранения.
        Ключи этого словаря могут перезаписать базовые ключи, если это необходимо.
        Например, если дочерний класс хочет сохранить 'model' (или 'model_state_dict') по-своему.
        """
        return {}

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """
        Загружает дополнительные, специфичные для детектора атрибуты из предоставленного словаря.
        Этот метод вызывается после загрузки базовых атрибутов.
        Дочерние классы должны переопределить это для восстановления своего полного состояния.
        Также здесь может быть установлены self.is_trained и self.model, если они обрабатываются специфично.
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Сохранение состояния детектора, включая базовые и специфичные атрибуты.
        """
        if not self.is_trained:
            logger.warning(f"Детектор {self.model_name} не обучен. Сохранение не выполнено.")
            return
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Собираем базовое состояние
            basic_save_dict = {
                'model_state': self.model, 
                'scaler': self.scaler,
                'min_score_': self.min_score_,
                'max_score_': self.max_score_
            }
            
            # Получаем специфичные для детектора атрибуты
            additional_attrs = self._get_attributes_to_save()
            
            # Объединяем словари. Атрибуты из additional_attrs перезапишут базовые при совпадении ключей.
            final_save_dict = {**basic_save_dict, **additional_attrs}
            
            joblib.dump(final_save_dict, path)
            logger.info(f"Состояние детектора {self.model_name} сохранено в {path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели {self.model_name}: {e}", exc_info=True)

    def load_model(self, filepath: str) -> None:
        """Загружает состояние детектора из файла."""
        try:
            # В начале предполагаем, что модель не будет обучена после этой попытки загрузки
            self.is_trained = False 
            logger.info(f"Попытка загрузки модели {self.model_name} из {filepath}...")
            
            with open(filepath, 'rb') as f:
                saved_state = joblib.load(f)
            
            # Базовые атрибуты (могут быть переопределены или дополнены в _load_additional_attributes)
            self.model = saved_state.get('model_state') # Может быть None, если дочерний класс обрабатывает модель иначе
            self.scaler = saved_state.get('scaler')
            self.min_score_ = saved_state.get('min_score_')
            self.max_score_ = saved_state.get('max_score_')
            
            # Загружаем специфичные для детектора атрибуты
            self._load_additional_attributes(saved_state)
            
            # Если все прошло успешно, включая _load_additional_attributes,
            # и _load_additional_attributes сам не установил is_trained в False,
            # то считаем модель обученной.
            # Если _load_additional_attributes установил is_trained (в True или False), его значение будет использовано.
            # Если он не трогал is_trained, то здесь мы подтверждаем успешную загрузку.
            if not hasattr(self, '_custom_is_trained_handling_in_load_additional') or \
               not self._custom_is_trained_handling_in_load_additional:
                self.is_trained = True # Основной случай: загрузка успешна

            # Логируем финальное состояние is_trained
            logger.info(f"Модель {self.model_name} успешно обработана при загрузке из {filepath}. is_trained={self.is_trained}")

        except FileNotFoundError:
            logger.error(f"Файл модели {self.model_name} не найден по пути: {filepath}")
            self.is_trained = False # Убедимся, что флаг снят
            # Не перевыбрасываем, чтобы приложение могло продолжить работу с необученным детектором
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name} из {filepath}: {e}", exc_info=True)
            self.is_trained = False # Убедимся, что флаг снят
            # Можно перевыбросить, если критично: raise

    def fit_normalizer(self, scores: np.ndarray) -> None:
        """
        Вычисляет параметры для нормализации скоров по шкале [0, 1].
        
        Args:
            scores: Массив скоров аномальности для обучения нормализатора
        """
        # Игнорируем NaN при поиске min/max
        valid_scores = scores[~np.isnan(scores)]
        
        if len(valid_scores) == 0:
            logger.warning(f"({self.model_name}): Нет валидных скоров для обучения нормализатора.")
            self.min_score_ = 0
            self.max_score_ = 1
            return
            
        self.min_score_ = float(np.min(valid_scores))
        self.max_score_ = float(np.max(valid_scores))
        
        # Обработка случая, когда все скоры одинаковые
        if self.max_score_ == self.min_score_:
            logger.warning(f"({self.model_name}): Все скоры одинаковы ({self.min_score_}).")
            # Избегаем деления на ноль, немного увеличивая диапазон
            self.max_score_ += 1e-6 
            
        logger.info(f"Нормализатор для {self.model_name} обучен: min={self.min_score_:.4f}, max={self.max_score_:.4f}")

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """
        Применяет Min-Max нормализацию к скорам аномальности.
        
        Args:
            scores: Массив сырых скоров аномальности
            
        Returns:
            Массив нормализованных скоров в диапазоне [0, 1]
        """
        if self.min_score_ is None or self.max_score_ is None:
            raise RuntimeError(f"Нормализатор для {self.model_name} не был обучен.")
        
        if self.max_score_ == self.min_score_:
            # Если min=max, возвращаем 0.5 для всех значений
            normalized_scores = np.full_like(scores, 0.5, dtype=float)
        else:
            # Стандартная Min-Max нормализация
            normalized_scores = (scores - self.min_score_) / (self.max_score_ - self.min_score_)
             
        # Ограничиваем результат диапазоном [0, 1]
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        
        # Сохраняем NaN в результате, если они были в исходных скорах
        normalized_scores[np.isnan(scores)] = np.nan
        
        return normalized_scores

    def get_config(self) -> Dict[str, Any]:
        # Этот метод может быть переопределен для включения дополнительных параметров
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'features': self.features if hasattr(self, 'features') else None,
            # Добавьте другие общие параметры, если они есть
        }

    def _reset_state(self):
        """
        Сбрасывает состояние детектора, включая модель, скейлер и флаг обученности.
        Дочерние классы должны переопределить этот метод, если у них есть дополнительные атрибуты для сброса.
        """
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.min_score_ = None
        self.max_score_ = None
        logger.debug(f"({self.model_name}) Сброс базового состояния детектора (из AnomalyDetector._reset_state).")
    
    def get_shap_explainer(self) -> Optional[Any]:
        """Возвращает explainer SHAP, если он доступен для этого детектора."""
        return getattr(self, 'shap_explainer', None)
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детали для объяснения аномалий для предоставленных "сырых" данных.
        Каждый элемент списка соответствует строке из data_for_explanation_raw.
        
        Словарь может содержать:
        - 'shap_values': Dict[str, float] - значения SHAP для каждого признака
        - 'detector_specific_info': Dict[str, Any] - дополнительная информация, специфичная для детектора
        
        Args:
            data_for_explanation_raw: DataFrame с "сырыми" данными (до предобработки).
            
        Returns:
            Optional[List[Dict[str, Any]]]: Список объяснений или None, если детектор не поддерживает
                                          объяснения или произошла ошибка.
        """
        logger.debug(f"({self.model_name}) Метод get_explanation_details не реализован специфично, будет использована реализация get_shap_explanations (если есть).")
        if hasattr(self, 'get_shap_explanations') and callable(getattr(self, 'get_shap_explanations')):
            try:
                shap_explanations = self.get_shap_explanations(data_for_explanation_raw)
                if shap_explanations:
                    return [{"shap_values": shap} for shap in shap_explanations]
            except Exception as e:
                logger.error(f"({self.model_name}) Ошибка при получении объяснений SHAP: {e}", exc_info=True)
        return None

# =============================================================================
# Статистический детектор на основе Z-оценки
# =============================================================================

class StatisticalDetector(AnomalyDetector):
    """
    Детектор аномалий на основе Z-оценки (Z-score) для одного числового признака.
    
    Вычисляет стандартное отклонение значения признака от среднего
    и выявляет аномалии на основе заданного порога.
    """
    _model_type_for_factory_ref: str = "statistical" 
 
    def __init__(self, feature: str, threshold: float = 3.0, model_name: str = "statistical_zscore"):
        """
        Инициализирует детектор статистических аномалий.
        
        Args:
            feature: Название признака для анализа
            threshold: Порог для Z-оценки, выше которого точка считается аномалией
            model_name: Уникальное имя модели (опционально)
        """
        super().__init__(model_name)
        self.feature: str = feature
        self.threshold: float = threshold
        # self.model будет содержать статистики (словарь с 'mean' и 'std_dev')
        self.model: Dict[str, float] = {}

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает словарь атрибутов для сохранения."""
        return {
            'feature': self.feature,
            'threshold': self.threshold,
            'model': self.model # Этот ключ перезапишет 'model_state' из базового класса
        }
        
    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает дополнительные атрибуты в экземпляр."""
        self.feature = loaded_data.get('feature', '')
        self.threshold = loaded_data.get('threshold', 3.0)
        # Восстанавливаем модель из загруженных данных. 
        # В StatisticalDetector модель - это словарь со статистиками {'mean': x, 'std_dev': y}
        
        # Вариант 1: Берем из 'model', если есть (приоритет)
        if 'model' in loaded_data and isinstance(loaded_data['model'], dict):
            self.model = loaded_data['model']
        # Вариант 2: Берем из 'model_state', если модель отсутствует (это может быть резервным вариантом)
        elif 'model_state' in loaded_data and isinstance(loaded_data['model_state'], dict):
            self.model = loaded_data['model_state']
        else:
            # Если ни один вариант не сработал, оставляем модель пустой
            self.model = {}
            logger.warning(f"({self.model_name}) Не удалось загрузить модель-статистики из сохраненного состояния.")
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет наличие целевого признака и возвращает DataFrame.
        
        Args:
            data: Исходный DataFrame
            
        Returns:
            DataFrame с обработанным целевым признаком
        """
        if self.feature not in data.columns:
            raise ValueError(f"Признак {self.feature} отсутствует в данных")
        
        # Копируем, чтобы избежать изменения оригинала
        data_copy = data.copy()
        
        # Если нужна более сложная предобработка, добавьте здесь
        return data_copy

    def train(self, data: pd.DataFrame) -> None:
        """
        Вычисляет статистики (среднее и стандартное отклонение) для признака.
        
        Args:
            data: DataFrame с обучающими данными
        """
        processed_data = self.preprocess(data)
        feature_data = processed_data[self.feature]
        
        # Вычисление статистик
        mean_val = feature_data.mean()
        std_dev_val = feature_data.std()
        
        if np.isnan(mean_val) or np.isnan(std_dev_val):
            logger.error(f"({self.model_name}) Не удалось вычислить статистики для {self.feature}: mean={mean_val}, std_dev={std_dev_val}")
            self.is_trained = False
            return
            
        if std_dev_val <= 0:
            logger.warning(f"({self.model_name}) Стандартное отклонение равно нулю. Z-тест не может быть применен корректно.")
            
        # Сохранение статистик
        self.model = {
            'mean': mean_val,
            'std_dev': std_dev_val
        }
        
        self.is_trained = True
        logger.info(f"({self.model_name}) Обучен: mean={mean_val:.4f}, std_dev={std_dev_val:.4f}")
        
        # Обучение нормализатора скоров
        # Получаем ненормализованные скоры аномалий на обучающей выборке
        z_scores = np.abs((feature_data - mean_val) / std_dev_val)
        self.fit_normalizer(z_scores)

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаруживает аномалии, вычисляя Z-оценку.
        
        Args:
            data: DataFrame с данными для анализа
            
        Returns:
            DataFrame с добавленными колонками 'anomaly_score', 'is_anomaly'
            и опционально 'anomaly_score_normalized'
        """
        if not self.is_trained or not self.model:
            logger.error(f"({self.model_name}) Детектор не обучен или нет сохраненных статистик.")
            # Возвращаем исходный DataFrame с пустыми колонками аномалий
            result_df = data.copy()
            result_df['anomaly_score'] = np.nan
            result_df['is_anomaly'] = False
            return result_df
            
        mean_val = self.model.get('mean')
        std_dev_val = self.model.get('std_dev')
        
        if mean_val is None or std_dev_val is None:
            logger.error(f"({self.model_name}) Отсутствуют необходимые статистики для обнаружения аномалий.")
            # Возвращаем исходный DataFrame с пустыми колонками аномалий
            result_df = data.copy()
            result_df['anomaly_score'] = np.nan
            result_df['is_anomaly'] = False
            return result_df
        
        data = self.preprocess(data)
        feature_data = data[self.feature]
        
        # Проверка std_dev на ноль или очень малое значение
        if np.isclose(std_dev_val, 0):
            logger.warning(f"({self.model_name}) Стандартное отклонение ({std_dev_val:.4g}) близко к нулю во время детекции. Z-оценки могут быть некорректны (0 или inf).")
            # Если значение равно mean, z_score = 0. Иначе - inf.
            # np.sign вернет 0 если (feature_data - mean_val) это 0, что корректно.
            z_scores = np.where(feature_data == mean_val, 0.0, np.inf * np.sign(feature_data - mean_val))
        else:
            z_scores = (feature_data - mean_val) / std_dev_val
        
        anomaly_scores_raw = np.abs(z_scores)
        
        data['anomaly_score'] = anomaly_scores_raw
        # Заменяем inf на очень большое число перед сравнением с порогом, если порог не inf
        # Это нужно, если self.threshold не np.inf
        finite_threshold = self.threshold if np.isfinite(self.threshold) else np.finfo(float).max
        data['is_anomaly'] = np.where(np.isinf(anomaly_scores_raw), True, anomaly_scores_raw > finite_threshold)
        
        if self.min_score_ is not None and self.max_score_ is not None:
            data['anomaly_score_normalized'] = self.normalize_score(anomaly_scores_raw.values) # Передаем .values для numpy array
        else:
            logger.warning(f"({self.model_name}) Нормализатор не обучен (min_score_ или max_score_ is None). 'anomaly_score_normalized' не будет рассчитан.")
            data['anomaly_score_normalized'] = np.nan

        return data
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детали для объяснения аномалий на основе статистических показателей.
        
        Args:
            data_for_explanation_raw: DataFrame с "сырыми" данными для объяснения.
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None, если возникла ошибка.
        """
        if not self.is_trained or not self.model or self.model.get('mean') is None or self.model.get('std_dev') is None:
            logger.warning(f"({self.model_name}) Детектор не обучен или отсутствуют статистики. Невозможно предоставить объяснения.")
            return None
        
        try:
            processed_df = self.preprocess(data_for_explanation_raw.copy())
            if processed_df.empty or self.feature not in processed_df.columns:
                logger.warning(f"({self.model_name}) После предобработки данные пусты или отсутствует признак {self.feature}.")
                return None
            
            feature_values = processed_df[self.feature]
            mean_val = self.model['mean']
            std_dev_val = self.model['std_dev']
            
            # Рассчитываем z-оценки для каждой строки
            z_scores = (feature_values - mean_val) / std_dev_val if not np.isclose(std_dev_val, 0) else np.zeros_like(feature_values)
            
            explanations = []
            for i, (index, value) in enumerate(feature_values.items()):
                # Вычисляем необходимые элементы объяснения
                z_score = z_scores.iloc[i] if not np.isnan(z_scores.iloc[i]) else None
                
                explanation = {
                    "detector_specific_info": {
                        "feature_name": self.feature,
                        "feature_value": float(value) if pd.notna(value) else None,
                        "z_score": float(z_score) if pd.notna(z_score) else None,
                        "mean_on_train": float(mean_val),
                        "std_on_train": float(std_dev_val),
                        "threshold": float(self.threshold),
                        "explanation": f"Значение {value:.4g} отклоняется от среднего {mean_val:.4g} на {abs(z_score):.4g} стандартных отклонений (порог: {self.threshold})."
                    }
                }
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None

# --- Реализация Детектора Isolation Forest --- 

class IsolationForestDetector(AnomalyDetector):
    """
    Детектор аномалий на основе алгоритма Isolation Forest.
    """
    _model_type_for_factory_ref: str = "isolation_forest"

    def __init__(self, 
                 features: List[str], 
                 n_estimators: int = 100, 
                 contamination: Union[str, float] = 'auto', 
                 random_state: Optional[int] = 42, # Разрешаем None для random_state
                 model_name: str = "isolation_forest"):
        super().__init__(model_name)
        self.features: List[str] = features
        self.n_estimators: int = n_estimators
        self.contamination: Union[str, float] = contamination
        self.random_state: Optional[int] = random_state
        # self.model будет инициализирован в train или загружен
        self.model: Optional[IsolationForest] = None 
        self.scaler: Optional[StandardScaler] = None # StandardScaler будет создан в preprocess/train
        
        # Для SHAP
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.expected_value_shap: Optional[Any] = None # Может быть float или массив, зависит от вывода SHAP

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Выбирает указанные признаки и применяет стандартизацию.
        Если fit_scaler=True, то StandardScaler обучается и сохраняется.
        """
        if not self.features:
            logger.warning(f"({self.model_name}) Список признаков 'features' пуст. Предобработка не будет выполнена.")
            return data.copy() # Возвращаем копию, чтобы избежать изменения оригинала

        data_processed = data[self.features].copy()

        # Приведение типов к числовому, если возможно, и обработка ошибок
        for feature in self.features:
            if not pd.api.types.is_numeric_dtype(data_processed[feature]):
                try:
                    data_processed[feature] = pd.to_numeric(data_processed[feature], errors='raise')
                except ValueError as e:
                    raise TypeError(f"Признак '{feature}' для IsolationForest содержит нечисловые значения, которые не могут быть преобразованы: {e}")
        
        # Замена inf на NaN для корректной работы StandardScaler
        data_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Обработка NaN перед масштабированием
        # Вариант 1: Заполнение средним (или медианой, нулем и т.д.)
        # Простой вариант: если есть NaN после замены inf, вызываем ошибку или предупреждение, 
        # так как Isolation Forest может плохо работать с NaN в некоторых реализациях sklearn или при explainer'ах.
        if data_processed.isnull().values.any():
            # Принятие решения: либо ошибка, либо заполнение, либо передача дальше, если модель поддерживает
            # Для IsolationForest лучше заполнить, чтобы избежать проблем ниже, особенно с SHAP
            logger.warning(f"({self.model_name}) Обнаружены NaN значения в признаках {self.features} перед масштабированием. Будет применено заполнение средним.")
            for col in data_processed.columns[data_processed.isnull().any()]:
                data_processed[col].fillna(data_processed[col].mean(), inplace=True)
            # Проверим еще раз, остались ли NaN (если среднее тоже NaN, например, вся колонка NaN)
            if data_processed.isnull().values.any():
                cols_with_nan_after_fill = data_processed.columns[data_processed.isnull().any()].tolist()
                logger.error(f"({self.model_name}) После заполнения NaN средним, все еще остались NaN в колонках: {cols_with_nan_after_fill}. Это может привести к ошибкам.")
                # Можно либо рейзить ошибку, либо обработать иначе
                raise ValueError(f"Не удалось обработать все NaN в данных для Isolation Forest в колонках: {cols_with_nan_after_fill}")

        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(data_processed)
            logger.info(f"({self.model_name}) StandardScaler обучен.")
        
        if self.scaler is not None:
            scaled_data = self.scaler.transform(data_processed)
            data_processed = pd.DataFrame(scaled_data, columns=self.features, index=data_processed.index)
        else:
            logger.warning(f"({self.model_name}) StandardScaler не обучен. Данные не будут масштабироваться.")
            # Это может быть ожидаемым поведением, если fit_scaler=False и scaler не был загружен

        return data_processed

    def train(self, data: pd.DataFrame) -> None:
        if not self.features:
            logger.error(f"({self.model_name}) Невозможно обучить Isolation Forest: список признаков 'features' пуст.")
            self.is_trained = False
            return

        logger.info(f"({self.model_name}) Начало обучения Isolation Forest на признаках: {self.features}")
        # Предобработка данных с обучением скейлера
        try:
            train_data_processed = self.preprocess(data, fit_scaler=True)
        except (ValueError, TypeError) as e_prep:
            logger.error(f"({self.model_name}) Ошибка предобработки данных при обучении: {e_prep}")
            self.is_trained = False
            return

        if train_data_processed.empty:
            logger.warning(f"({self.model_name}) Нет данных для обучения после предобработки. Модель не будет обучена.")
            self.is_trained = False
            return

        self.model = IsolationForest(
            n_estimators=self.n_estimators, 
            contamination=self.contamination, 
            random_state=self.random_state,
            # bootstrap=True, # Можно добавить, если нужно и если версия sklearn поддерживает и это дефолт для SHAP explainer
            # verbose=1 # для отладки
        )
        
        try:
            self.model.fit(train_data_processed)
            self.is_trained = True
            logger.info(f"({self.model_name}) Модель Isolation Forest обучена.")

            # Обучаем нормализатор на скорах (decision_function возвращает противоположные значения для IsolationForest)
            # Чем меньше значение, тем более аномальный. Мы инвертируем их для консистентности (больше = аномальнее).
            # Однако, для fit_normalizer нам нужны "сырые" скоры в том виде, как они будут использоваться для нормализации.
            # SHAP для IsolationForest использует decision_function. Скоры из decision_function обычно отрицательные для нормальных
            # и положительные (или менее отрицательные) для аномалий. 
            # Чтобы привести к стандартному виду (больше = аномальнее), часто делают -model.decision_function()
            # Мы будем нормализовывать именно эти инвертированные скоры.
            
            # Получаем скоры на обучающей выборке
            # scores_train = self.model.decision_function(train_data_processed)
            # anomaly_scores_train = -scores_train # Инвертируем: теперь чем больше, тем аномальнее
            
            # Исправлено: contamination в IsolationForest означает долю выбросов. 
            # decision_function(X) возвращает скор для каждого сэмпла. 
            # Отрицательные значения обычно для нормальных, положительные для выбросов (в некоторых версиях наоборот).
            # predict(X) возвращает -1 для выбросов, 1 для нормальных.
            # score_samples(X) возвращает противоположность anomaly score. Чем выше, тем нормальнее.
            # Мы будем использовать score_samples и инвертировать его, чтобы большие значения означали большую аномальность.
            scores_for_normalizer = -self.model.score_samples(train_data_processed) # score_samples: Higher values are more normal.
            self.fit_normalizer(scores_for_normalizer)

            # Инициализация SHAP Explainer, если SHAP доступен
            if shap is not None and hasattr(shap, 'TreeExplainer'):
                try:
                    # Для TreeExplainer обычно передают саму модель
                    self.shap_explainer = shap.TreeExplainer(self.model, data=train_data_processed, feature_perturbation="interventional") 
                    # self.expected_value_shap = self.shap_explainer.expected_value # Для некоторых моделей это одно число
                    # Для многоклассовых или многовыходных моделей expected_value может быть массивом
                    # Для IsolationForest это обычно одно значение, если contamination не используется для разделения на "классы"
                    logger.info(f"({self.model_name}) SHAP TreeExplainer успешно инициализирован.")
                except Exception as e_shap_init:
                    logger.error(f"({self.model_name}) Ошибка при инициализации SHAP TreeExplainer: {e_shap_init}. Объяснения могут быть недоступны.", exc_info=True)
                    self.shap_explainer = None # Сбрасываем, если ошибка
            else:
                logger.warning(f"({self.model_name}) Библиотека SHAP или TreeExplainer не доступен. Объяснения SHAP не будут генерироваться.")
                self.shap_explainer = None

        except Exception as e_fit:
            logger.error(f"({self.model_name}) Ошибка при обучении модели Isolation Forest: {e_fit}", exc_info=True)
            self.is_trained = False
            self.model = None # Сбрасываем модель

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            logger.warning(f"({self.model_name}) Модель Isolation Forest не обучена. Возвращается DataFrame с NaN скорами.")
            data_copy = data.copy()
            data_copy['anomaly_score'] = np.nan 
            data_copy['is_anomaly'] = False
            data_copy['anomaly_score_normalized'] = np.nan
            return data_copy
        
        if not self.features:
            logger.error(f"({self.model_name}) Невозможно выполнить детекцию: список признаков 'features' пуст.")
            data_copy = data.copy()
            data_copy['anomaly_score'] = np.nan
            data_copy['is_anomaly'] = False
            data_copy['anomaly_score_normalized'] = np.nan
            return data_copy

        logger.info(f"({self.model_name}) Начало детекции с Isolation Forest на признаках: {self.features}")
        try:
            # Предобработка данных без обучения скейлера (он должен быть уже обучен)
            data_processed = self.preprocess(data, fit_scaler=False) 
        except (ValueError, TypeError) as e_prep:
            logger.error(f"({self.model_name}) Ошибка предобработки данных при детекции: {e_prep}. Возвращается DataFrame с NaN скорами.")
            data_copy = data.copy()
            data_copy['anomaly_score'] = np.nan
            data_copy['is_anomaly'] = False
            data_copy['anomaly_score_normalized'] = np.nan
            return data_copy

        if data_processed.empty:
            logger.warning(f"({self.model_name}) Нет данных для детекции после предобработки.")
            data_copy = data.copy()
            data_copy['anomaly_score'] = np.nan
            data_copy['is_anomaly'] = False
            data_copy['anomaly_score_normalized'] = np.nan
            return data_copy

        try:
            # decision_function: чем меньше, тем более аномальный. 
            # score_samples: чем выше, тем более "нормальный" (меньше аномальный). 
            # predict: -1 для аномалий (outliers), 1 для нормальных (inliers).
            
            # Используем score_samples и инвертируем, чтобы большие значения = более аномальные
            # Это согласуется с тем, как мы обучали нормализатор
            raw_anomaly_scores = -self.model.score_samples(data_processed)
            
            # Получение флага is_anomaly на основе предсказаний модели
            # predict возвращает -1 для выбросов, 1 для нормальных
            predictions = self.model.predict(data_processed) 
            is_anomaly_flags = (predictions == -1)

            data_copy = data.copy() # Создаем копию оригинального DataFrame для добавления результатов
            data_copy['anomaly_score'] = raw_anomaly_scores
            data_copy['is_anomaly'] = is_anomaly_flags
            
            # Нормализация скоров
            if self.min_score_ is not None and self.max_score_ is not None:
                data_copy['anomaly_score_normalized'] = self.normalize_score(raw_anomaly_scores)
            else:
                logger.warning(f"({self.model_name}) Нормализатор не обучен. 'anomaly_score_normalized' не будет рассчитан.")
                data_copy['anomaly_score_normalized'] = np.nan
                
            logger.info(f"({self.model_name}) Детекция с Isolation Forest завершена. Обнаружено {data_copy['is_anomaly'].sum()} аномалий.")
            return data_copy
            
        except Exception as e_det:
            logger.error(f"({self.model_name}) Ошибка во время детекции Isolation Forest: {e_det}", exc_info=True)
            data_copy = data.copy()
            data_copy['anomaly_score'] = np.nan
            data_copy['is_anomaly'] = False
            data_copy['anomaly_score_normalized'] = np.nan
            return data_copy


    def get_shap_explanations(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, float]]]:
        """
        Генерирует SHAP объяснения для предоставленных "сырых" данных.
        Выполняет предобработку внутри.
        """
        if not self.is_trained or self.model is None:
            logger.warning(f"({self.model_name}) Модель не обучена. SHAP объяснения не могут быть сгенерированы.")
            return None
        
        if self.shap_explainer is None:
            # Попытка ленивой инициализации, если возможно (может потребоваться фон)
            # На данный момент, если эксплейнер не создан в train, то он не будет создан здесь.
            # TODO: Рассмотреть возможность ленивой инициализации, если это безопасно и есть фоновые данные
            logger.warning(f"({self.model_name}) SHAP explainer не инициализирован. Объяснения не могут быть сгенерированы.")
            return None

        if data_for_explanation_raw.empty:
            logger.warning(f"({self.model_name}) DataFrame для SHAP объяснений пуст.")
            return [] # Возвращаем пустой список, а не None, если данные пустые, но модель готова
            
        if not self.features:
            logger.error(f"({self.model_name}) Атрибут 'features' не определен. Невозможно сгенерировать SHAP объяснения.")
            return None

        # Проверка наличия всех необходимых признаков в сырых данных
        missing_features = [f for f in self.features if f not in data_for_explanation_raw.columns]
        if missing_features:
            logger.error(f"({self.model_name}) Отсутствуют необходимые признаки {missing_features} в 'data_for_explanation_raw' для SHAP.")
            return None
            
        logger.info(f"({self.model_name}) Генерация SHAP объяснений для {len(data_for_explanation_raw)} сэмплов (IsolationForest)...")
        try:
            # 1. Предобработка данных
            # Используем fit_scaler=False, так как скейлер должен быть уже обучен
            data_processed_for_shap = self.preprocess(data_for_explanation_raw, fit_scaler=False)

            if data_processed_for_shap.empty:
                logger.warning(f"({self.model_name}) SHAP: Нет данных после предобработки.")
                return []

            # Убедимся, что данные для SHAP содержат только нужные признаки и в правильном порядке,
            # хотя preprocess уже должен был это сделать.
            # Это скорее перестраховка, если preprocess вернул лишние колонки (что не должен).
            try:
                data_for_shap_final = data_processed_for_shap[self.features]
            except KeyError as e:
                logger.error(f"({self.model_name}) Ошибка: После предобработки отсутствуют необходимые признаки для SHAP: {e}.")
                return None
            
            # 2. Генерация SHAP values
            # shap_values = self.shap_explainer.shap_values(data_for_shap_final, check_additivity=False) 
            shap_values_result = self.shap_explainer(data_for_shap_final) 
            
            if hasattr(shap_values_result, 'values'):
                shap_values_arr = shap_values_result.values
            else:
                shap_values_arr = shap_values_result

            if not isinstance(shap_values_arr, np.ndarray):
                logger.error(f"({self.model_name}) SHAP explainer вернул неожиданный тип: {type(shap_values_arr)}. Ожидался np.ndarray.")
                return None
            
            if shap_values_arr.ndim != 2 or shap_values_arr.shape[1] != len(self.features):
                logger.error(f"({self.model_name}) SHAP values имеют некорректную размерность: {shap_values_arr.shape}. Ожидалось ({len(data_for_shap_final)}, {len(self.features)}).")
                return None

            explanations_list: List[Dict[str, float]] = []
            for i in range(shap_values_arr.shape[0]):
                instance_shap_values = shap_values_arr[i]
                shap_dict = {feature: float(shap_value) for feature, shap_value in zip(self.features, instance_shap_values)}
                explanations_list.append(shap_dict)
            
            logger.info(f"({self.model_name}) SHAP объяснения успешно сгенерированы для IsolationForest.")
            return explanations_list

        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации SHAP объяснений для IsolationForest: {e}", exc_info=True)
            return None

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Сохраняет специфичные для IsolationForest атрибуты."""
        # Модель (self.model) и скейлер (self.scaler) сохраняются базовым классом AnomalyDetector.
        # Основные параметры __init__ (features, n_estimators, contamination, random_state) 
        # являются частью конфигурации и обычно передаются при создании экземпляра,
        # но для полноты можно их тоже сохранять, если они могут меняться.
        # SHAP explainer и expected_value обычно не сериализуются напрямую с моделью, 
        # а пересоздаются при загрузке/обучении, если это возможно.
        
        # Возвращаем только то, что критично для восстановления состояния, помимо model и scaler.
        # В данном случае, это features, так как они нужны для preprocess и SHAP.
        attrs = {
            'features': self.features, # Важно для preprocess и SHAP
            'n_estimators': self.n_estimators,
            'contamination': self.contamination,
            'random_state': self.random_state
            # Не сохраняем self.shap_explainer и self.expected_value_shap, т.к. они часто несериализуемы
            # или их лучше пересоздавать при загрузке на основе обученной модели.
        }
        # self.model и self.scaler будут сохранены базовым классом
        # если они не переопределены в 'model_state' или 'scaler' здесь.
        return attrs

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные для IsolationForest атрибуты."""
        # self.model и self.scaler загружены базовым классом AnomalyDetector.
        self.features = loaded_data.get('features', self.features if hasattr(self, 'features') else [])
        self.n_estimators = loaded_data.get('n_estimators', self.n_estimators if hasattr(self, 'n_estimators') else 100)
        self.contamination = loaded_data.get('contamination', self.contamination if hasattr(self, 'contamination') else 'auto')
        self.random_state = loaded_data.get('random_state', self.random_state if hasattr(self, 'random_state') else 42)

        # После загрузки модели, если она обучена, можно попытаться пересоздать SHAP explainer.
        # Это делается здесь, а не в train, чтобы explainer был доступен и для загруженных моделей.
        if self.is_trained and self.model is not None and shap is not None and hasattr(shap, 'TreeExplainer'):
            if not self.features:
                logger.warning(f"({self.model_name}) Невозможно инициализировать SHAP explainer после загрузки: список 'features' пуст.")
                self.shap_explainer = None
                return
            
            logger.info(f"({self.model_name}) Попытка пересоздания SHAP TreeExplainer после загрузки модели...")
            try:
                # Для TreeExplainer нужен доступ к данным или статистикам, на которых он был обучен.
                # Если мы не сохраняем эти данные, то можно передать placeholder или None,
                # но это может ограничить функционал или потребовать feature_perturbation="tree_path_dependent".
                # Простой вариант: если модель загружена, explainer не будет иметь доступа к обучающим данным.
                # Можно использовать বৈশিষ্ট্য Perturbation="interventional" и передать фоновые данные, если они есть.
                # Либо просто инициализировать без данных, если explainer это поддерживает.
                
                # Если SHAP explainer требует данные, которых у нас нет при загрузке (например, train_data_processed),
                # то мы не сможем его здесь корректно инициализировать без этих данных.
                # В таком случае, SHAP объяснения будут доступны только если модель обучалась в текущей сессии.
                # Альтернатива: сохранить небольшой сэмпл данных вместе с моделью для SHAP.
                
                # Пока что, если модель есть, пытаемся инициализировать без данных или с предупреждением.
                # self.shap_explainer = shap.TreeExplainer(self.model) # Может работать для некоторых версий/случаев
                # logger.info(f"({self.model_name}) SHAP TreeExplainer пересоздан после загрузки (без явных данных). Feature names: {self.features}")
                # Если выше не работает, можно не инициализировать здесь, а только в train. 
                # Или требовать, чтобы пользователь передавал фоновые данные для get_shap_explanations.
                # Оставляем инициализацию в train, а здесь только сбрасываем, если он не был загружен.
                if not hasattr(self, 'shap_explainer') or self.shap_explainer is None:
                    logger.warning(f"({self.model_name}) SHAP explainer не был загружен и не будет пересоздан здесь. Он должен быть создан во время train.")
                    self.shap_explainer = None # Убедимся, что он None

            except Exception as e_shap_reload:
                logger.error(f"({self.model_name}) Ошибка при попытке пересоздания SHAP TreeExplainer после загрузки: {e_shap_reload}", exc_info=True)
                self.shap_explainer = None
        elif not self.is_trained or self.model is None:
            self.shap_explainer = None # Если модель не обучена, эксплейнера быть не должно


    def _reset_state(self):
        """Сбрасывает состояние детектора к начальному (перед обучением)."""
        logger.info(f"({self.model_name}) Сброс состояния детектора IsolationForest.")
        self.model = None
        self.scaler = None # Scaler обучается в preprocess(fit_scaler=True)
        self.is_trained = False
        self.min_score_ = None
        self.max_score_ = None
        self.shap_explainer = None
        self.expected_value_shap = None
        # Основные параметры (features, n_estimators и т.д.) остаются, так как они часть конфигурации.

# --- Реализация Детектора Autoencoder (PyTorch) --- 

class Autoencoder(nn.Module):
    """Простая модель Автоэнкодера на PyTorch."""
    def __init__(self, input_dim, encoding_dim=16, hidden_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU() # Или другая активация для bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            # Последний слой БЕЗ активации, если используем MSELoss
            # или с Sigmoid, если данные нормализованы в [0,1] и используем BCELoss
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector(AnomalyDetector):
    """
    Детектор аномалий на основе Автоэнкодера (PyTorch).
    """
    _model_type_for_factory_ref: str = "autoencoder"
    _custom_is_trained_handling_in_load_additional = True # Флаг для базового класса

    # Внутренний класс-обертка для SHAP
    class _AEModelWrapperForSHAP(nn.Module):
        def __init__(self, autoencoder_model: Autoencoder): # Autoencoder - ваша модель nn.Module
            super().__init__()
            self.autoencoder_model = autoencoder_model
            self.autoencoder_model.eval() # Переводим модель в режим оценки

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Хотим объяснить ошибку реконструкции
            with torch.no_grad(): # SHAP сам будет управлять градиентами
                x_reconstructed = self.autoencoder_model(x)
                # Ошибка реконструкции (MSE) для каждого сэмпла, усредненная по признакам
                reconstruction_error_per_sample = torch.mean((x - x_reconstructed)**2, dim=tuple(range(1, x.ndim)))
            return reconstruction_error_per_sample

    def __init__(self, 
                 features: List[str], 
                 encoding_dim: int = 16, 
                 hidden_dim: int = 64,
                 epochs: int = 10,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 threshold_std_factor: float = 3.0, # Множитель для std для определения порога
                 model_name: str = "autoencoder",
                 shap_background_samples: int = 100): # Новый параметр
        
        super().__init__(model_name)
        if not features:
            logger.error(f"({self.model_name}) Список признаков 'features' не может быть пустым.")
            raise ValueError("Список признаков 'features' не может быть пустым.")
            
        self.features = features
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_std_factor = threshold_std_factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"({self.model_name}) Используемое устройство: {self.device}")
        
        self.shap_background_samples = shap_background_samples
        self.explainer: Optional[shap.GradientExplainer] = None 
        self.background_data_for_shap: Optional[torch.Tensor] = None
        
        self.model: Optional[Autoencoder] = None # Явно указываем тип модели
        self.threshold_: Optional[float] = None  # Порог для определения аномалии
        
        # self.scaler уже инициализирован в AnomalyDetector.__init__
        # self.min_score_ и self.max_score_ также из AnomalyDetector
        self._reset_state() # Сбрасывает модель, порог и флаг is_trained

    def _reset_state(self) -> None:
        logger.info(f"({self.model_name}) Сброс состояния детектора Autoencoder.")
        self.model = None
        self.scaler = StandardScaler() 
        self.threshold = None
        self.shap_background_data = None
        self.shap_explainer = None
        self.is_trained = False
        self.min_score_ = None
        self.max_score_ = None

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Выбирает признаки, масштабирует и обрабатывает NaN.
        """
        if not all(f in data.columns for f in self.features):
            missing = [f for f in self.features if f not in data.columns]
            raise ValueError(f"Autoencoder: Следующие признаки не найдены: {missing}")
        
        processed_data_df = data[self.features].copy()
        
        # Заполнение пропусков (Важно: делать до масштабирования!)
        for feature in self.features:
            if processed_data_df[feature].isnull().any():
                median_val = processed_data_df[feature].median()
                processed_data_df[feature] = processed_data_df[feature].fillna(median_val)
        
        # Масштабирование
        if processed_data_df.empty:
            logger.warning("AE Предупреждение: Нет данных для масштабирования.")
            # Возвращаем пустой DataFrame с ожидаемыми колонками
            return pd.DataFrame(columns=self.features) 
            
        if fit_scaler:
            self.scaler.fit(processed_data_df)
            
        if self.scaler and hasattr(self.scaler, 'mean_'):
             scaled_data_array = self.scaler.transform(processed_data_df)
             # Преобразуем обратно в DataFrame с сохранением индекса и колонок
             processed_data_df = pd.DataFrame(scaled_data_array, columns=self.features, index=processed_data_df.index)
        else:
             raise RuntimeError("AE StandardScaler не был обучен.")
        
        return processed_data_df

    def train(self, data: pd.DataFrame):
        logger.info(f"Начало обучения детектора {self.model_name} на {self.device}...")
        start_time = time.time()
        
        try:
            # Шаг 1: Предобработка данных (возвращает DataFrame)
            processed_df = self.preprocess(data, fit_scaler=True)
            if processed_df.empty:
                logger.warning(f"Обучение {self.model_name} невозможно: нет данных после предобработки.")
                self.is_trained = False
                return
            
            # Шаг 1.1: Конвертация в тензор
            train_tensor = torch.tensor(processed_df.values, dtype=torch.float32).to(self.device)
            if train_tensor.shape[0] == 0:
                logger.warning(f"Обучение {self.model_name} невозможно: тензор пуст после конвертации.")
                self.is_trained = False
                return

            input_dim = train_tensor.shape[1]
            self.model = Autoencoder(input_dim, self.encoding_dim, self.hidden_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            train_dataset = TensorDataset(train_tensor, train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_features, _ in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_features)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_features.size(0)
                epoch_loss /= len(train_loader.dataset)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f'Эпоха [{epoch+1}/{self.epochs}], Ошибка реконструкции: {epoch_loss:.6f}')
            
            self.model.eval()
            reconstruction_errors_list = []
            with torch.no_grad():
                outputs_full = self.model(train_tensor)
                errors_full = torch.mean((outputs_full - train_tensor) ** 2, dim=1)
                reconstruction_errors_list = errors_full.cpu().numpy()
            
            self.fit_normalizer(reconstruction_errors_list)
            mean_error = np.mean(reconstruction_errors_list)
            std_error = np.std(reconstruction_errors_list)
            self.threshold = mean_error + self.threshold_std_factor * std_error
            self.is_trained = True

            # --- SHAP Background Data and Explainer Initialization ---
            self.explainer = None # Сброс перед новой инициализацией
            self.background_data_for_shap = None

            if shap and self.is_trained and self.model:
                num_samples_shap = min(self.shap_background_samples, train_tensor.shape[0])
                if num_samples_shap > 0:
                    indices = np.random.choice(train_tensor.shape[0], num_samples_shap, replace=False)
                    self.background_data_for_shap = train_tensor[indices]
                    logger.info(f"({self.model_name}) Сохранен фоновый набор данных SHAP ({self.background_data_for_shap.shape[0]} экз.) на устройстве {self.background_data_for_shap.device}.")

                    wrapped_model_for_shap = self._AEModelWrapperForSHAP(self.model)
                    # wrapped_model_for_shap.autoencoder_model.eval() # Уже сделано в __init__ обертки

                    try:
                        self.explainer = shap.GradientExplainer(wrapped_model_for_shap, self.background_data_for_shap)
                        logger.info(f"({self.model_name}) SHAP GradientExplainer инициализирован.")
                    except Exception as e_shap_init:
                        logger.error(f"({self.model_name}) Ошибка при инициализации SHAP GradientExplainer: {e_shap_init}", exc_info=True)
                        self.explainer = None
                        self.background_data_for_shap = None 
                else:
                    logger.warning(f"({self.model_name}) Недостаточно данных ({train_tensor.shape[0]} сэмплов) для создания фонового набора SHAP (требуется {self.shap_background_samples}).")
                    self.explainer = None
                    self.background_data_for_shap = None
            else:
                if not shap: logger.warning(f"({self.model_name}) Библиотека SHAP не доступна. Эксплейнер не будет создан.")
                if not (self.is_trained and self.model): logger.warning(f"({self.model_name}) Модель не обучена или отсутствует. Эксплейнер SHAP не будет создан.")
                self.explainer = None
                self.background_data_for_shap = None
            # ---------------------------------------------------------

            end_time = time.time()
            logger.info(f"Детектор {self.model_name} успешно обучен за {end_time - start_time:.2f} сек.")
            logger.info(f"  Признаки: {self.features}")
            logger.info(f"  Порог ошибки реконструкции: {self.threshold:.6f}")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении детектора {self.model_name}: {e}", exc_info=True)
            self.is_trained = False

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет ошибку реконструкции (СЫРОЙ скор) и определяет аномалии."""
        if not self.is_trained or self.model is None or self.scaler is None or self.threshold is None:
            raise RuntimeError(f"Детектор {self.model_name} должен быть обучен и успешно загружен (включая порог).")

        logger.info(f"Начало детекции аномалий детектором {self.model_name}...")
        original_index = data.index # Сохраняем исходный индекс
        
        try:
            # 1. Предобработка данных (возвращает DataFrame)
            processed_df = self.preprocess(data, fit_scaler=False) 
            if processed_df.empty:
                logger.warning(f"Детекция {self.model_name} невозможна: нет данных после предобработки.")
                result_df = data.copy()
                result_df['is_anomaly'] = False
                result_df['anomaly_score'] = np.nan
                return result_df

            # 1.1 Конвертация в тензор
            input_tensor = torch.tensor(processed_df.values, dtype=torch.float32).to(self.device)
            if input_tensor.shape[0] == 0:
                logger.warning(f"Детекция {self.model_name} невозможна: тензор пуст после конвертации.")
                result_df = data.copy()
                result_df['is_anomaly'] = False
                result_df['anomaly_score'] = np.nan
                return result_df

            # 2. Получение ошибок реконструкции
            self.model.eval() # Режим оценки
            reconstruction_errors = []
            with torch.no_grad():
                outputs = self.model(input_tensor)
                errors = torch.mean((outputs - input_tensor) ** 2, dim=1) # MSE по каждому примеру
                reconstruction_errors = errors.cpu().numpy()

            # 3. Определение аномалий и формирование результата
            is_anomaly = reconstruction_errors > self.threshold_
            
            processed_indices = processed_df.index # Получаем индексы строк без NaN в нужных фичах
            
            if len(reconstruction_errors) != len(processed_indices):
                 logger.warning(f"Предупреждение: Несоответствие длин ({len(reconstruction_errors)} vs {len(processed_indices)}). Результаты могут быть неверны.")
                 result_df = data.copy()
                 result_df['is_anomaly'] = False
                 result_df['anomaly_score'] = np.nan
                 return result_df
            
            results_intermediate = pd.DataFrame({
                'anomaly_score': reconstruction_errors, # Сохраняем СЫРУЮ ошибку
                'is_anomaly': is_anomaly
            }, index=processed_indices)

            # Объединяем с исходным DataFrame, чтобы получить полный результат
            result_df = data.join(results_intermediate)
            
            # Заполняем пропуски для is_anomaly (False) и anomaly_score (NaN) для строк, 
            # которые могли быть отфильтрованы (если бы preprocess удалял строки)
            # В текущей реализации preprocess не удаляет, а заполняет NaN, так что все строки должны быть.
            result_df['is_anomaly'] = result_df['is_anomaly'].fillna(False)
            # anomaly_score для необработанных строк остается NaN
            
            # Убедимся, что порядок строк и сам индекс сохранены
            result_df = result_df.reindex(original_index) 

            print(f"Детекция аномалий детектором {self.model_name} завершена.")
            return result_df

        except Exception as e:
            print(f"Ошибка при детекции аномалий детектором {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            result_df = data.copy()
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = np.nan
            return result_df

    def get_shap_explanations(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, float]]]:
        """
        Генерирует SHAP объяснения для предоставленных "сырых" данных.
        Выполняет предобработку внутри.
        """
        if not self.is_trained or self.model is None:
            logger.warning(f"({self.model_name}) Модель не обучена. SHAP объяснения не могут быть сгенерированы.")
            return None
        
        if self.shap_explainer is None:
            # Попытка ленивой инициализации, если возможно (может потребоваться фон)
            # На данный момент, если эксплейнер не создан в train, то он не будет создан здесь.
            # TODO: Рассмотреть возможность ленивой инициализации, если это безопасно и есть фоновые данные
            logger.warning(f"({self.model_name}) SHAP explainer не инициализирован. Объяснения не могут быть сгенерированы.")
            return None

        if data_for_explanation_raw.empty:
            logger.warning(f"({self.model_name}) DataFrame для SHAP объяснений пуст.")
            return [] # Возвращаем пустой список, а не None, если данные пустые, но модель готова
            
        if not self.features:
            logger.error(f"({self.model_name}) Атрибут 'features' не определен. Невозможно сгенерировать SHAP объяснения.")
            return None

        # Проверка наличия всех необходимых признаков в сырых данных
        missing_features = [f for f in self.features if f not in data_for_explanation_raw.columns]
        if missing_features:
            logger.error(f"({self.model_name}) Отсутствуют необходимые признаки {missing_features} в 'data_for_explanation_raw' для SHAP.")
            return None
            
        logger.info(f"({self.model_name}) Генерация SHAP объяснений для {len(data_for_explanation_raw)} сэмплов (IsolationForest)...")
        try:
            # 1. Предобработка данных
            # Используем fit_scaler=False, так как скейлер должен быть уже обучен
            data_processed_for_shap = self.preprocess(data_for_explanation_raw, fit_scaler=False)

            if data_processed_for_shap.empty:
                logger.warning(f"({self.model_name}) SHAP: Нет данных после предобработки.")
                return []

            # Убедимся, что данные для SHAP содержат только нужные признаки и в правильном порядке,
            # хотя preprocess уже должен был это сделать.
            # Это скорее перестраховка, если preprocess вернул лишние колонки (что не должен).
            try:
                data_for_shap_final = data_processed_for_shap[self.features]
            except KeyError as e:
                logger.error(f"({self.model_name}) Ошибка: После предобработки отсутствуют необходимые признаки для SHAP: {e}.")
                return None
            
            # 2. Генерация SHAP values
            # shap_values = self.shap_explainer.shap_values(data_for_shap_final, check_additivity=False) 
            shap_values_result = self.shap_explainer(data_for_shap_final) 
            
            if hasattr(shap_values_result, 'values'):
                shap_values_arr = shap_values_result.values
            else:
                shap_values_arr = shap_values_result

            if not isinstance(shap_values_arr, np.ndarray):
                logger.error(f"({self.model_name}) SHAP explainer вернул неожиданный тип: {type(shap_values_arr)}. Ожидался np.ndarray.")
                return None
            
            if shap_values_arr.ndim != 2 or shap_values_arr.shape[1] != len(self.features):
                logger.error(f"({self.model_name}) SHAP values имеют некорректную размерность: {shap_values_arr.shape}. Ожидалось ({len(data_for_shap_final)}, {len(self.features)}).")
                return None

            explanations_list: List[Dict[str, float]] = []
            for i in range(shap_values_arr.shape[0]):
                instance_shap_values = shap_values_arr[i]
                shap_dict = {feature: float(shap_value) for feature, shap_value in zip(self.features, instance_shap_values)}
                explanations_list.append(shap_dict)
            
            logger.info(f"({self.model_name}) SHAP объяснения успешно сгенерированы для IsolationForest.")
            return explanations_list

        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации SHAP объяснений для IsolationForest: {e}", exc_info=True)
            return None

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Собирает атрибуты детектора для сохранения."""
        if not self.model:
            logger.warning(f"({self.model_name}) Модель отсутствует при попытке сохранения.")
            model_state_dict_to_save = None
        else:
            model_state_dict_to_save = self.model.state_dict()

        background_data_to_save = None
        if self.background_data_for_shap is not None:
            if isinstance(self.background_data_for_shap, torch.Tensor):
                background_data_to_save = self.background_data_for_shap.cpu().numpy()
            else:
                background_data_to_save = self.background_data_for_shap
            logger.debug(f"({self.model_name}) Подготовлены фоновые данные SHAP для сохранения (форма: {background_data_to_save.shape if hasattr(background_data_to_save, 'shape') else 'N/A'}).")

        attrs = {
            'model_state_dict': model_state_dict_to_save,
            'features': self.features,
            'encoding_dim': self.encoding_dim,
            'hidden_dim': self.hidden_dim, # Добавляем hidden_dim если он используется при создании Autoencoder
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'threshold_std_factor': self.threshold_std_factor,
            'threshold_': self.threshold_,
            'shap_background_samples': self.shap_background_samples,
            'background_data_for_shap': background_data_to_save
            # scaler, min_score_, max_score_ сохраняются базовым классом
        }
        logger.debug(f"({self.model_name}) Атрибуты для сохранения (Autoencoder): { {k: type(v) for k, v in attrs.items() if k != 'model_state_dict'} }")
        return attrs

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает атрибуты, специфичные для AutoencoderDetector."""
        self.is_trained = False # Сначала сбрасываем
        logger.info(f"({self.model_name}) Загрузка дополнительных атрибутов для Autoencoder...")

        self.features = loaded_data.get('features')
        if not self.features:
            logger.error(f"({self.model_name}) Ошибка загрузки: 'features' отсутствуют. Модель не может быть восстановлена.")
            return

        self.encoding_dim = loaded_data.get('encoding_dim', self.encoding_dim)
        self.hidden_dim = loaded_data.get('hidden_dim', self.hidden_dim) # Загружаем hidden_dim
        self.epochs = loaded_data.get('epochs', self.epochs)
        self.batch_size = loaded_data.get('batch_size', self.batch_size)
        self.learning_rate = loaded_data.get('learning_rate', self.learning_rate)
        self.threshold_std_factor = loaded_data.get('threshold_std_factor', self.threshold_std_factor)
        self.threshold_ = loaded_data.get('threshold_')

        self.shap_background_samples = loaded_data.get('shap_background_samples', self.shap_background_samples)
        
        loaded_background_numpy = loaded_data.get('background_data_for_shap')
        self.background_data_for_shap = None
        if loaded_background_numpy is not None:
            try:
                self.background_data_for_shap = torch.from_numpy(loaded_background_numpy).float().to(self.device)
                logger.info(f"({self.model_name}) Фоновые данные SHAP загружены на {self.device} (форма: {self.background_data_for_shap.shape}).")
            except Exception as e_torch_bg:
                logger.error(f"({self.model_name}) Ошибка конвертации фоновых данных SHAP: {e_torch_bg}", exc_info=True)
                self.background_data_for_shap = None
        else:
            logger.info(f"({self.model_name}) Фоновые данные SHAP отсутствуют.")

        model_state_dict = loaded_data.get('model_state_dict')
        self.model = None
        if model_state_dict and self.features:
            logger.info(f"({self.model_name}) Попытка восстановления модели Autoencoder...")
            try:
                current_ae_model = Autoencoder(
                    input_dim=len(self.features), 
                    encoding_dim=self.encoding_dim,
                    hidden_dim=self.hidden_dim # Используем загруженный hidden_dim
                )
                current_ae_model.load_state_dict(model_state_dict)
                current_ae_model.to(self.device)
                current_ae_model.eval()
                self.model = current_ae_model
                logger.info(f"({self.model_name}) Модель Autoencoder успешно восстановлена на {self.device}.")
            except Exception as e_ae_load:
                logger.error(f"({self.model_name}) Ошибка восстановления модели Autoencoder: {e_ae_load}", exc_info=True)
                self.model = None
        elif not self.features:
             logger.error(f"({self.model_name}) Невозможно восстановить модель: 'features' отсутствуют.")
        else: 
            logger.warning(f"({self.model_name}) 'model_state_dict' отсутствует. Модель не восстановлена.")

        self.explainer = None
        if shap and self.model and self.background_data_for_shap is not None and self.background_data_for_shap.shape[0] > 0:
            logger.info(f"({self.model_name}) Попытка пересоздания SHAP GradientExplainer...")
            try:
                wrapped_model = self._AEModelWrapperForSHAP(self.model)
                self.explainer = shap.GradientExplainer(wrapped_model, self.background_data_for_shap)
                logger.info(f"({self.model_name}) SHAP GradientExplainer успешно пересоздан.")
            except Exception as e_shap_reload:
                logger.error(f"({self.model_name}) Ошибка пересоздания SHAP GradientExplainer: {e_shap_reload}", exc_info=True)
                self.explainer = None
        # ... (логирование про отсутствие SHAP, модели или данных) ...

        # Проверка готовности и установка is_trained
        if self.model is not None and \
           self.scaler is not None and \
           self.min_score_ is not None and \
           self.max_score_ is not None and \
           self.threshold_ is not None:
            self.is_trained = True
            logger.info(f"({self.model_name}) Детектор успешно загружен и помечен как обученный. Порог: {self.threshold_:.6f}")
        else:
            self.is_trained = False
            missing_components = []
            if self.model is None: missing_components.append("модель")
            if self.scaler is None: missing_components.append("скейлер")
            if self.min_score_ is None: missing_components.append("min_score_")
            if self.max_score_ is None: missing_components.append("max_score_")
            if self.threshold_ is None: missing_components.append("threshold_")
            logger.warning(f"({self.model_name}) Детектор загружен, но не все компоненты присутствуют: {', '.join(missing_components)}. is_trained=False.")
            