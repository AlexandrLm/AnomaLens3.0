from typing import Protocol, List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Протокол для объекта конфигурации, передаваемого детекторам
class DetectorConfig(Protocol):
    model_name: str
    features: List[str]
    # ... другие общие параметры конфигурации ...

# Протокол для класса детектора аномалий
class AnomalyDetectorProtocol(Protocol):
    model_name: str
    is_trained: bool
    min_score_: Optional[float]
    max_score_: Optional[float]

    def __init__(self, **kwargs: Any) -> None:
        ...

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        ...

    def train(self, data: pd.DataFrame) -> None:
        ...

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        # Возвращает DataFrame с колонками 'anomaly_score', 'is_anomaly', ['anomaly_score_normalized']
        ...

    def save_model(self, path: str) -> None:
        ...

    def load_model(self, path: str) -> None:
        ...
    
    def fit_normalizer(self, scores: np.ndarray) -> None:
        ...

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        ...

    def get_shap_explanations(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, float]]]:
        """
        Генерирует SHAP объяснения для предоставленных данных.
        data_for_explanation_raw: DataFrame с "сырыми" данными (до предобработки).
        """
        ...

    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детали для объяснения аномалий для предоставленных "сырых" данных.
        Каждый элемент списка соответствует строке из data_for_explanation_raw.
        
        Словарь может содержать:
        - 'shap_values': Dict[str, float] - значения SHAP для каждого признака
        - 'detector_specific_info': Dict[str, Any] - дополнительная информация, специфичная для детектора
        
        Returns:
            Optional[List[Dict[str, Any]]]: Список объяснений или None, если детектор не поддерживает
                                          объяснения или произошла ошибка.
        """