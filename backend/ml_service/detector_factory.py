"""
detector_factory.py - Фабрика для создания детекторов аномалий
===============================================================
Модуль реализует шаблон "Фабрика" для создания различных типов детекторов аномалий.
"""

import os
import logging
from typing import Dict, Type, Any, Optional, List, Union

# Импортируем все типы детекторов
from .detector import (
    AnomalyDetector, 
    StatisticalDetector, 
    IsolationForestDetector, 
    AutoencoderDetector
)
from .vae_detector import VAEDetector
from .graph_detector import GraphAnomalyDetector

# Импортируем новые детекторы
from .transaction_detectors import (
    PriceFreightRatioDetector,
    CategoryPriceOutlierDetector,
    MultiFeatureIsolationForestDetector,
    TransactionVAEDetector
)
from .behavior_detectors import (
    SellerPricingBehaviorDetector,
    SellerCategoryMixDetector,
    BehaviorIsolationForestDetector
)
from .time_series_detectors import (
    SeasonalDeviationDetector,
    MovingAverageVolatilityDetector,
    CumulativeSumDetector
)

logger = logging.getLogger(__name__)

class DetectorFactory:
    """
    Фабрика для создания и инициализации детекторов аномалий различных типов.
    
    Позволяет создавать детекторы по имени типа и загружать предобученные модели.
    """
    
    # Словарь соответствия названий детекторов и их классов
    _detector_classes: Dict[str, Type[AnomalyDetector]] = {
        # Базовые детекторы
        "statistical": StatisticalDetector,
        "isolation_forest": IsolationForestDetector,
        "autoencoder": AutoencoderDetector,
        "vae": VAEDetector,
        "graph": GraphAnomalyDetector,
        
        # Транзакционные детекторы
        "price_freight_ratio": PriceFreightRatioDetector,
        "category_price_outlier": CategoryPriceOutlierDetector,
        "transaction_isolation_forest": MultiFeatureIsolationForestDetector,
        "transaction_vae": TransactionVAEDetector,
        
        # Поведенческие детекторы
        "seller_pricing_behavior": SellerPricingBehaviorDetector,
        "seller_category_mix": SellerCategoryMixDetector,
        "behavior_isolation_forest": BehaviorIsolationForestDetector,
        
        # Детекторы временных рядов
        "seasonal_deviation": SeasonalDeviationDetector,
        "moving_average_volatility": MovingAverageVolatilityDetector,
        "cumulative_sum": CumulativeSumDetector,
    }
    
    @classmethod
    def create_detector(
        cls, 
        detector_type: str, 
        **params
    ) -> AnomalyDetector:
        """
        Создает детектор указанного типа с заданными параметрами.
        
        Args:
            detector_type: Тип детектора из списка поддерживаемых
            **params: Параметры для конструктора детектора
            
        Returns:
            Экземпляр детектора заданного типа
            
        Raises:
            ValueError: Если указан неизвестный тип детектора
        """
        if detector_type not in cls._detector_classes:
            raise ValueError(
                f"Неизвестный тип детектора: '{detector_type}'. "
                f"Доступные типы: {list(cls._detector_classes.keys())}"
            )
        
        detector_class = cls._detector_classes[detector_type]
        try:
            # Передаем только те параметры из конфига, которые ожидает __init__ детектора
            init_params = {k: v for k, v in params.items() 
                           if k in detector_class.__init__.__code__.co_varnames and k != 'type'}
            
            # Создаем экземпляр детектора
            detector = detector_class(**init_params)
            logger.info(f"Создан экземпляр детектора {params['model_name']} типа {detector_type}")
            
            # --- Добавляем сохранение model_filename как атрибута --- 
            model_filename = params.get('model_filename')
            if model_filename:
                setattr(detector, 'model_filename', model_filename)
                logger.debug(f"Атрибут model_filename='{model_filename}' установлен для {params['model_name']}")
            # ------------------------------------------------------
                
            return detector
            
        except Exception as e:
            logger.error(f"Ошибка при создании детектора типа {detector_type}: {e}")
            raise
    
    @classmethod
    def create_and_load_detector(
        cls, 
        detector_type: str, 
        model_path: str, 
        **params
    ) -> AnomalyDetector:
        """
        Создает детектор и загружает предобученную модель.
        
        Args:
            detector_type: Тип детектора из списка поддерживаемых
            model_path: Путь к сохраненной модели
            **params: Параметры для конструктора детектора
            
        Returns:
            Экземпляр детектора с загруженной моделью
            
        Raises:
            ValueError: Если указан неизвестный тип детектора или файл модели не найден
        """
        # Создаем детектор
        detector = cls.create_detector(detector_type, **params)
        
        # Проверяем наличие файла модели
        # --- Добавляем логирование для отладки пути ---
        current_cwd = os.getcwd()
        logger.debug(f"Проверка существования файла модели: CWD='{current_cwd}', Путь='{model_path}'")
        # -------------------------------------------
        if not os.path.exists(model_path):
            logger.warning(f"Файл модели не найден: {model_path}. Детектор будет не обучен.")
            return detector
        
        # Загружаем модель
        try:
            detector.load_model(model_path)
            if detector.is_trained:
                logger.info(f"Успешно загружена модель для детектора {detector.model_name} из {model_path}")
            else:
                logger.warning(f"Модель загружена, но детектор не помечен как обученный: {detector.model_name}")
                
            return detector
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {model_path} для детектора {detector.model_name}: {e}")
            logger.info(f"Возвращаем не обученный детектор {detector.model_name}")
            return detector
    
    @classmethod
    def create_detectors_from_config(
        cls, 
        configs: List[Dict[str, Any]], 
        model_base_path: str = "models"
    ) -> Dict[str, AnomalyDetector]:
        """
        Создает несколько детекторов из конфигурации.
        
        Args:
            configs: Список конфигураций для каждого детектора
            model_base_path: Базовый путь к директории с моделями
            
        Returns:
            Словарь с созданными детекторами {detector_name: detector}
        """
        detectors = {}
        
        for config in configs:
            # Извлекаем основные параметры конфигурации
            detector_type = config.get("type")
            model_filename = config.get("model_filename")
            
            # Проверка обязательных параметров
            if not detector_type:
                logger.warning(f"Пропущена конфигурация детектора из-за отсутствия 'type': {config}")
                continue
            
            # Если имя файла не указано, используем стандартное
            if not model_filename:
                model_filename = f"{detector_type}_model.joblib"
                logger.info(f"Имя файла модели не указано, используем: {model_filename}")
            
            # Полный путь к модели
            model_path = os.path.join(model_base_path, model_filename)
            
            # Создаем уникальное имя для детектора
            detector_name = cls._generate_detector_name(detector_type, config)
            
            # Параметры для создания детектора (без служебных)
            params = config.copy()
            params.pop("type", None)
            params.pop("weight", None)  # Вес используется в ансамбле, но не для создания детектора
            
            # Устанавливаем имя детектора
            params["model_name"] = detector_name
            
            try:
                # Создаем и загружаем детектор
                detector = cls.create_and_load_detector(detector_type, model_path, **params)
                detectors[detector_name] = detector
                logger.info(f"Детектор '{detector_name}' успешно создан и загружен")
            except Exception as e:
                logger.error(f"Не удалось создать или загрузить детектор '{detector_name}': {e}")
        
        return detectors
    
    @staticmethod
    def _generate_detector_name(detector_type: str, config: Dict[str, Any]) -> str:
        """
        Генерирует уникальное имя для детектора на основе его типа и параметров.
        
        Args:
            detector_type: Тип детектора
            config: Конфигурация детектора
            
        Returns:
            Сгенерированное имя детектора
        """
        # Формируем часть имени, связанную с признаком/признаками
        feature_name_part = "model"  # Имя по умолчанию
        
        if detector_type == "statistical":
            # Для статистического детектора используем имя признака
            feature_name_part = config.get('feature', 'feature')
        elif detector_type in ["isolation_forest", "autoencoder", "vae", 
                              "transaction_isolation_forest", "transaction_vae"]:
            # Для детекторов с признаками используем первый признак списка (если есть)
            feature_list = config.get('features', [])
            if feature_list:
                feature_name_part = feature_list[0]
        elif detector_type in ["category_price_outlier", "price_freight_ratio", 
                               "seller_pricing_behavior", "seller_category_mix",
                               "seasonal_deviation", "moving_average_volatility", "cumulative_sum"]:
            # Для этих детекторов имя обычно не зависит от конкретного "признака" из конфига,
            # так как они работают с предопределенной логикой или несколькими признаками.
            # Оставляем feature_name_part = "model" или можно использовать сам detector_type.
            # Для большей уникальности, если model_filename задан, можно его использовать или часть.
            # Но пока оставим "model" для простоты, чтобы имя было "detector_type_model".
            pass # feature_name_part остается "model"
        elif detector_type == "graph":
            # Для графового детектора можно не добавлять имя признака
            feature_name_part = "graph_analysis"

        # Формируем итоговое имя
        detector_name = f"{detector_type}_{feature_name_part}"
        
        return detector_name
    
    @classmethod
    def register_detector_class(cls, detector_type: str, detector_class: Type[AnomalyDetector]) -> None:
        """
        Регистрирует новый класс детектора в фабрике.
        
        Args:
            detector_type: Имя типа детектора
            detector_class: Класс детектора (должен быть подклассом AnomalyDetector)
        """
        if not issubclass(detector_class, AnomalyDetector):
            raise TypeError(f"Класс {detector_class.__name__} должен быть подклассом AnomalyDetector")
            
        cls._detector_classes[detector_type] = detector_class
        logger.info(f"Зарегистрирован новый тип детектора: {detector_type}")
    
    @classmethod
    def get_available_detector_types(cls) -> List[str]:
        """
        Возвращает список всех доступных типов детекторов.
        
        Returns:
            Список имен доступных типов детекторов
        """
        return list(cls._detector_classes.keys()) 