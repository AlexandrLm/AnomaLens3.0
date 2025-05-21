"""
multilevel_detector.py - Многоуровневая система обнаружения аномалий
=================================================================
Модуль реализует иерархическую систему обнаружения аномалий на трёх уровнях:
1. Транзакционный - поиск аномалий в отдельных транзакциях
2. Поведенческий - поиск аномалий в поведении продавцов/покупателей
3. Временной - поиск аномалий во временных рядах
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib
from abc import ABC, abstractmethod
import datetime

from .detector import AnomalyDetector
from .detector import StatisticalDetector
from .graph_detector import GraphAnomalyDetector
from .detector_factory import DetectorFactory

# Настраиваем логирование
logger = logging.getLogger(__name__)

class MultilevelAnomalyDetector:
    """
    Многоуровневая система обнаружения аномалий.
    
    Объединяет три уровня обнаружения аномалий:
    1. Транзакционный: аномалии в отдельных транзакциях (необычные цены, доставка)
    2. Поведенческий: аномалии в поведении продавцов/покупателей (нестабильные цены)
    3. Временной: аномалии во временных рядах (необычные всплески/падения продаж)
    
    Система последовательно анализирует данные на каждом уровне и агрегирует результаты,
    снижая количество ложных срабатываний за счет перекрестной проверки.
    """
    
    def __init__(self, config: Dict[str, Any], model_base_path: str = "models"):
        """
        Инициализирует многоуровневую систему обнаружения аномалий.
        
        Args:
            config: Конфигурация для всех уровней детекции, содержащая:
                - transaction_level: конфигурация детекторов транзакционного уровня (список словарей)
                - transaction_level_combination_method: метод комбинирования для транзакционного уровня
                - behavior_level: конфигурация детекторов поведенческого уровня (список словарей)
                - behavior_level_combination_method: метод комбинирования для поведенческого уровня
                - time_series_level: конфигурация детекторов временного уровня (список словарей)
                - time_series_level_combination_method: метод комбинирования для временного уровня
                - combination_weights: веса для объединения результатов уровней
            model_base_path: Базовый путь к директории моделей
        """
        self.model_base_path: str = model_base_path
        self.config: Dict[str, Any] = config
        
        self.level_weights: Dict[str, float] = config.get('combination_weights', {
            'transaction': 0.4,
            'behavior': 0.4, 
            'time_series': 0.2
        })
        
        self.transaction_detectors: Dict[str, AnomalyDetector] = {}
        self.transaction_detector_weights: Dict[str, float] = {}
        self.transaction_level_combination_method: str = config.get("transaction_level_combination_method", "weighted_average")
        
        self.behavior_detectors: Dict[str, AnomalyDetector] = {}
        self.behavior_detector_weights: Dict[str, float] = {}
        self.behavior_level_combination_method: str = config.get("behavior_level_combination_method", "weighted_average")
        
        self.time_series_detectors: Dict[str, AnomalyDetector] = {}
        self.time_series_detector_weights: Dict[str, float] = {}
        self.time_series_level_combination_method: str = config.get("time_series_level_combination_method", "weighted_average")
        
        self._initialize_detectors()
        
    def _initialize_detectors(self) -> None:
        """
        Инициализирует и загружает детекторы для всех уровней,
        а также их веса и методы комбинации.
        """
        logger.info("Инициализация многоуровневой системы обнаружения аномалий...")
        
        detector_configs_by_level = {
            'transaction': self.config.get('transaction_level', []),
            'behavior': self.config.get('behavior_level', []),
            'time_series': self.config.get('time_series_level', [])
        }

        for level_name, level_configs in detector_configs_by_level.items():
            if not level_configs:
                logger.info(f"Конфигурация для уровня '{level_name}' отсутствует или пуста.")
                continue
            
            logger.info(f"Инициализация детекторов для уровня '{level_name}'...")
            
            detectors_on_level = {}
            detector_weights_on_level = {}
            
            for detector_config in level_configs:
                try:
                    detector_type_from_config = detector_config['type']
                    model_filename_from_config = detector_config.get("model_filename")
                    
                    params_for_factory = detector_config.copy()
                    # Генерируем имя так же, как это делает DetectorFactory.create_detectors_from_config
                    # и добавляем его в params_for_factory, чтобы оно было доступно в create_detector
                    detector_name = DetectorFactory._generate_detector_name(detector_type_from_config, params_for_factory) # Передаем params_for_factory
                    params_for_factory['model_name'] = detector_name 

                    # Уберем 'type' из params_for_factory, т.к. он передается первым аргументом в фабрику
                    # Это нужно делать ПОСЛЕ генерации имени, если _generate_detector_name использует 'type' из params_for_factory
                    params_for_factory.pop('type', None)

                    if model_filename_from_config:
                        model_path = os.path.join(self.model_base_path, model_filename_from_config)
                        # create_and_load_detector ожидает detector_type, model_path, и **kwargs для конструктора детектора
                        detector_instance = DetectorFactory.create_and_load_detector(
                            detector_type=detector_type_from_config, # Явно передаем detector_type
                            model_path=model_path,
                            **params_for_factory # Остальные параметры из конфига
                        )
                        logger.info(f"  Детектор '{detector_name}' (тип: {detector_type_from_config}) попытка создания и загрузки из {model_path}.")
                    else:
                        # Если model_filename не указан, просто создаем (не будет загружен)
                        # create_detector ожидает detector_type, и **kwargs
                        detector_instance = DetectorFactory.create_detector(
                            detector_type=detector_type_from_config, # Явно передаем detector_type
                            **params_for_factory # Остальные параметры из конфига
                        )
                        logger.info(f"  Детектор '{detector_name}' (тип: {detector_type_from_config}) создан без файла модели (не будет загружен).")
                    
                    detectors_on_level[detector_name] = detector_instance
                    detector_weights_on_level[detector_name] = detector_config.get("weight", 1.0)
                    logger.info(f"  Детектор '{detector_name}' (тип: {detector_type_from_config}) инициализирован с весом {detector_weights_on_level[detector_name]}.")
                except Exception as e:
                    logger.error(f"Ошибка при инициализации детектора ({detector_config.get('type', 'N/A')}) для уровня '{level_name}': {e}", exc_info=True)
            
            if level_name == 'transaction':
                self.transaction_detectors = detectors_on_level
                self.transaction_detector_weights = detector_weights_on_level
            elif level_name == 'behavior':
                self.behavior_detectors = detectors_on_level
                self.behavior_detector_weights = detector_weights_on_level
            elif level_name == 'time_series':
                self.time_series_detectors = detectors_on_level
                self.time_series_detector_weights = detector_weights_on_level

        logger.info(f"Многоуровневая система инициализирована: "
                    f"{len(self.transaction_detectors)} транзакционных детекторов (метод: {self.transaction_level_combination_method}), "
                    f"{len(self.behavior_detectors)} поведенческих детекторов (метод: {self.behavior_level_combination_method}), "
                    f"{len(self.time_series_detectors)} детекторов временных рядов (метод: {self.time_series_level_combination_method})")

    def _combine_scores_for_single_level(
        self,
        level_name: str,
        normalized_scores_list: List[np.ndarray],
        detector_names: List[str],
        detector_weights: Dict[str, float],
        combination_method: str
    ) -> np.ndarray:
        """
        Комбинирует нормализованные скоры от детекторов одного уровня.

        Args:
            level_name: Имя уровня (для логирования)
            normalized_scores_list: Список массивов нормализованных скоров от каждого детектора.
            detector_names: Список имен детекторов, соответствующих скорам.
            detector_weights: Словарь {имя_детектора: вес} для этого уровня.
            combination_method: Метод комбинирования ("average", "max", "weighted_average").

        Returns:
            Массив NumPy с комбинированными скорами.
        """
        if not normalized_scores_list:
            logger.warning(f"_combine_scores_for_single_level: Получен пустой список normalized_scores_list.")
            # Возвращаем массив NaN той же длины, что и ожидалось бы, если бы был хоть один результат.
            # Это сложный кейс, т.к. мы не знаем ожидаемую длину. 
            # Лучше, чтобы вызывающий код обрабатывал пустой results.
            # Однако, если мы должны вернуть что-то, то массив из одного NaN может быть вариантом,
            # или вызывающий код должен гарантировать, что normalized_scores_list не пуст.
            # Для текущей реализации, если normalized_scores_list пуст, значит, ни один детектор не отработал,
            # и соответствующий _detect_*_level метод должен вернуть None или DataFrame с NaN.
            # Здесь мы ожидаем, что normalized_scores_list не будет пустым, если функция вызвана.
            # Добавим явную проверку и выброс исключения или логирование.
            raise ValueError("Список normalized_scores_list не может быть пустым для комбинирования.")

        scores_matrix = np.array(normalized_scores_list)
        logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}): Матрица скоров (shape {scores_matrix.shape}), метод: {combination_method}")

        if combination_method == "average":
            combined_scores = np.nanmean(scores_matrix, axis=0)
            logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, метод: average): Распределение combined_scores: Min={np.nanmin(combined_scores):.4f}, Max={np.nanmax(combined_scores):.4f}, Mean={np.nanmean(combined_scores):.4f}")
            return combined_scores
        elif combination_method == "max":
            combined_scores = np.nanmax(scores_matrix, axis=0)
            logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, метод: max): Распределение combined_scores: Min={np.nanmin(combined_scores):.4f}, Max={np.nanmax(combined_scores):.4f}, Mean={np.nanmean(combined_scores):.4f}")
            return combined_scores
        elif combination_method == "weighted_average":
            use_simple_average = False
            if not detector_weights:
                logger.warning(f"Уровень '{level_name}': Метод 'weighted_average', но веса детекторов отсутствуют. Используется 'average'.")
                use_simple_average = True
            else:
                has_positive_weight = any(w > 0 for w in detector_weights.values() if w is not None)
                if not has_positive_weight:
                    logger.warning(f"Уровень '{level_name}': Метод 'weighted_average', но нет положительных весов. Используется 'average'.")
                    use_simple_average = True
            
            if use_simple_average:
                combined_scores = np.nanmean(scores_matrix, axis=0)
                logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, weighted_average -> average): Распределение combined_scores: Min={np.nanmin(combined_scores):.4f}, Max={np.nanmax(combined_scores):.4f}, Mean={np.nanmean(combined_scores):.4f}")
                return combined_scores

            # Если мы здесь, значит detector_weights не пуст и содержит хотя бы один положительный вес
            weights_array = np.array([detector_weights.get(name, 1.0) for name in detector_names])
            
            if np.sum(weights_array) == 0: # Эта проверка может остаться для случаев типа [1, -1]
                logger.warning(f"Уровень '{level_name}': Метод 'weighted_average', но сумма весов равна 0. Используется 'average'.")
                combined_scores = np.nanmean(scores_matrix, axis=0)
            else:
                masked_scores = np.ma.masked_invalid(scores_matrix)
                combined_scores = np.ma.average(masked_scores, axis=0, weights=weights_array).filled(np.nan)
            
            logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, метод: weighted_average): Используемые веса: {detector_weights}, массив весов: {weights_array}")
            logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, метод: weighted_average): Распределение combined_scores: Min={np.nanmin(combined_scores):.4f}, Max={np.nanmax(combined_scores):.4f}, Mean={np.nanmean(combined_scores):.4f}")
            return combined_scores
        else:
            logger.warning(f"Уровень '{level_name}': Неизвестный метод комбинирования: {combination_method}. Используется обычное среднее.")
            combined_scores = np.nanmean(scores_matrix, axis=0)
            logger.debug(f"_combine_scores_for_single_level (Уровень: {level_name}, unknown -> average): Распределение combined_scores: Min={np.nanmin(combined_scores):.4f}, Max={np.nanmax(combined_scores):.4f}, Mean={np.nanmean(combined_scores):.4f}")
            return combined_scores

    def _get_normalized_scores_from_detectors(
        self,
        data_to_process: pd.DataFrame,
        detectors: Dict[str, AnomalyDetector],
        level_name: str # для логирования
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Запускает детекторы и возвращает DataFrame с нормализованными скорами.

        Args:
            data_to_process: pd.DataFrame для передачи в метод detect каждого детектора.
            detectors: Словарь {имя_детектора: экземпляр_детектора}.
            level_name: Имя уровня для логирования.

        Returns:
            Кортеж (pd.DataFrame со столбцами нормализованных скоров, List[str] с именами детекторов, которые вернули скоры).
            DataFrame имеет тот же индекс, что и data_to_process.
            Если детектор не вернул 'anomaly_score' или произошла ошибка, его столбец будет состоять из NaN.
        """
        all_normalized_scores_df = pd.DataFrame(index=data_to_process.index)
        processed_detector_names = []

        if not detectors:
            logger.warning(f"Уровень '{level_name}': Нет детекторов для обработки.")
            return all_normalized_scores_df, processed_detector_names

        for detector_name, detector_instance in detectors.items():
            logger.info(f"Уровень '{level_name}': Запуск детектора '{detector_name}'...")
            try:
                # Убедимся, что детектор обучен, если он должен быть
                if not detector_instance.is_trained and not isinstance(detector_instance, (StatisticalDetector, GraphAnomalyDetector)): # Статистические и графовые могут работать без явного is_trained в некоторых сценариях, если их train только статистики собирает
                     # Некоторые детекторы (например, StatisticalDetector в PriceFreightRatioDetector) могут быть не is_trained,
                     # но их метод train вызывается внутри другого train, и они готовы к detect.
                     # Однако, для большинства ML моделей is_trained - это критично.
                     # Проверим, реализован ли метод train. Если да, то is_trained важен.
                     
                     # Проверяем, был ли метод train переопределен в классе экземпляра detector_instance
                     # по сравнению с базовым классом AnomalyDetector
                     train_method_of_instance_class = getattr(type(detector_instance), 'train', None)
                     train_method_of_base_class = getattr(AnomalyDetector, 'train', None)

                     is_train_overridden = False
                     if train_method_of_instance_class and train_method_of_base_class:
                         if train_method_of_instance_class is not train_method_of_base_class:
                             is_train_overridden = True

                     if is_train_overridden:
                         logger.warning(f"Детектор '{detector_name}' на уровне '{level_name}' не обучен. Пропускаем.")
                         all_normalized_scores_df[detector_name] = np.nan # Добавляем колонку NaN
                         continue

                detection_results_df = detector_instance.detect(data_to_process.copy())

                if 'anomaly_score' not in detection_results_df.columns:
                    logger.warning(f"Детектор '{detector_name}' на уровне '{level_name}' не вернул 'anomaly_score'. Пропускаем.")
                    all_normalized_scores_df[detector_name] = np.nan
                    continue
                
                raw_scores = detection_results_df['anomaly_score'].values.astype(float)

                # Нормализация скоров
                normalized_score_array = np.full_like(raw_scores, np.nan, dtype=float)
                if hasattr(detector_instance, 'min_score_') and hasattr(detector_instance, 'max_score_') and \
                   detector_instance.min_score_ is not None and detector_instance.max_score_ is not None:
                    # Убедимся, что min_score_ и max_score_ не NaN/inf перед использованием
                    if np.isfinite(detector_instance.min_score_) and np.isfinite(detector_instance.max_score_):
                        normalized_score_array = detector_instance.normalize_score(raw_scores.copy())
                    else:
                        logger.warning(f"Параметры нормализатора для '{detector_name}' некорректны (NaN/inf). Используем 0.5.")
                        valid_indices = ~np.isnan(raw_scores)
                        normalized_score_array[valid_indices] = 0.5
                else:
                    logger.warning(f"Нормализатор для '{detector_name}' не обучен или не имеет min/max. Попытка использовать скоры как есть или 0.5.")
                    if np.all(np.isnan(raw_scores)):
                        normalized_score_array = raw_scores # Оставляем NaN
                    else:
                        min_raw, max_raw = np.nanmin(raw_scores), np.nanmax(raw_scores)
                        if min_raw >= 0 and max_raw <= 1: # Предполагаем, что уже нормализован
                            normalized_score_array = raw_scores
                        else: # Скор не в диапазоне [0,1], используем 0.5 для не-NaN значений
                            valid_indices = ~np.isnan(raw_scores)
                            normalized_score_array[valid_indices] = 0.5
                
                all_normalized_scores_df[detector_name] = normalized_score_array
                processed_detector_names.append(detector_name)
                logger.info(f"Детектор '{detector_name}' обработан. Распределение нормализованных скоров: Min={np.nanmin(normalized_score_array):.4f}, Max={np.nanmax(normalized_score_array):.4f}, Mean={np.nanmean(normalized_score_array):.4f}")

            except Exception as e:
                logger.error(f"Ошибка при выполнении детектора '{detector_name}' на уровне '{level_name}': {e}", exc_info=True)
                all_normalized_scores_df[detector_name] = np.nan # Заполняем NaN в случае ошибки

        return all_normalized_scores_df, processed_detector_names

    def detect(self, data: pd.DataFrame, transaction_threshold: float = 0.6,
               behavior_threshold: float = 0.6, time_series_threshold: float = 0.6,
               final_threshold: float = 0.5) -> pd.DataFrame:
        """
        Выполняет обнаружение аномалий на всех уровнях и объединяет результаты.
        
        Args:
            data: Входные данные для анализа
            transaction_threshold: Порог для определения аномалий на транзакционном уровне
            behavior_threshold: Порог для определения аномалий на поведенческом уровне
            time_series_threshold: Порог для определения аномалий на временном уровне
            final_threshold: Порог для определения итоговых аномалий
            
        Returns:
            DataFrame с исходными данными и результатами обнаружения аномалий:
                - transaction_score: совокупный скор аномалий транзакционного уровня
                - behavior_score: совокупный скор аномалий поведенческого уровня
                - time_series_score: совокупный скор аномалий временного уровня
                - multilevel_score: итоговый взвешенный скор
                - is_anomaly: итоговый флаг аномалии
                - detailed_explanations_json: подробные объяснения для аномалий в формате JSON
        """
        result_df = data.copy()
        
        # Детекция на транзакционном уровне
        transaction_level_df = self._detect_transaction_level(data) # Возвращает data.copy() + 'transaction_score'
        if transaction_level_df is not None and 'transaction_score' in transaction_level_df.columns:
            result_df['transaction_level_score'] = transaction_level_df['transaction_score']
            result_df['transaction_is_anomaly'] = result_df['transaction_level_score'] > transaction_threshold
            
            # Копируем индивидуальные скоры от детекторов транзакционного уровня
            for col in transaction_level_df.columns:
                if col.startswith('trans_') and col.endswith('_score'):
                    result_df[col] = transaction_level_df[col]
        else:
            result_df['transaction_level_score'] = np.nan
            result_df['transaction_is_anomaly'] = False
            logger.info("Transaction level detection returned no results or no 'transaction_score' column.")
        
        # Детекция на поведенческом уровне
        # aggregated_behavior_data_with_scores это, например, seller_agg_df + 'behavior_score'
        aggregated_behavior_data_with_scores = self._detect_behavior_level(data) 
        if aggregated_behavior_data_with_scores is not None and \
           'seller_id' in aggregated_behavior_data_with_scores.columns and \
           'behavior_score' in aggregated_behavior_data_with_scores.columns and \
           'seller_id' in result_df.columns:
            
            behavior_scores_to_merge = aggregated_behavior_data_with_scores[['seller_id', 'behavior_score']].rename(
                columns={'behavior_score': 'behavior_level_score'}
            )
            result_df = result_df.merge(behavior_scores_to_merge, on='seller_id', how='left')
            result_df['behavior_is_anomaly'] = result_df['behavior_level_score'] > behavior_threshold
            
            # Копируем индивидуальные скоры от детекторов поведенческого уровня
            for col in aggregated_behavior_data_with_scores.columns:
                if col.startswith('behav_') and col.endswith('_score'):
                    # Мержим по seller_id
                    scores_df = aggregated_behavior_data_with_scores[['seller_id', col]]
                    result_df = result_df.merge(scores_df, on='seller_id', how='left')
        else:
            result_df['behavior_level_score'] = np.nan
            result_df['behavior_is_anomaly'] = False
            if aggregated_behavior_data_with_scores is None:
                logger.info("Behavior level detection returned no results.")
            else:
                logger.warning("Could not merge behavior scores. Check for 'seller_id' and 'behavior_score' in behavior results, and 'seller_id' in main transaction data.")
            
        # Детекция на временном уровне
        # aggregated_ts_data_with_scores это, например, daily_agg_df + 'time_series_score'
        aggregated_ts_data_with_scores = self._detect_time_series_level(data)
        if aggregated_ts_data_with_scores is not None and \
           'order_purchase_timestamp' in aggregated_ts_data_with_scores.columns and \
           'time_series_score' in aggregated_ts_data_with_scores.columns and \
           'order_purchase_timestamp' in result_df.columns:

            # Создаем колонки с датой (в формате datetime64[ns]) для корректного слияния
            result_df['merge_date_ts'] = pd.to_datetime(pd.to_datetime(result_df['order_purchase_timestamp']).dt.date)
            
            if 'date' in aggregated_ts_data_with_scores.columns and pd.api.types.is_datetime64_any_dtype(aggregated_ts_data_with_scores['date']):
                # Используем существующую колонку date если она datetime64
                aggregated_ts_data_with_scores['merge_date_ts'] = aggregated_ts_data_with_scores['date']
            else:
                # Создаем из order_purchase_timestamp
                aggregated_ts_data_with_scores['merge_date_ts'] = pd.to_datetime(pd.to_datetime(aggregated_ts_data_with_scores['order_purchase_timestamp']).dt.date)

            # Подготавливаем данные для слияния - только нужные колонки
            ts_scores_to_merge = aggregated_ts_data_with_scores[['merge_date_ts', 'time_series_score']].rename(
                columns={'time_series_score': 'time_series_level_score'}
            )
            
            # Копируем индивидуальные скоры от детекторов временного уровня
            for col in aggregated_ts_data_with_scores.columns:
                if col.startswith('ts_') and col.endswith('_score'):
                    ts_scores_to_merge[col] = aggregated_ts_data_with_scores[col]

            # Логируем статистику перед слиянием для диагностики
            logger.info(f"Слияние временных данных: result_df имеет {len(result_df)} строк, {result_df['merge_date_ts'].nunique()} уникальных дат")
            logger.info(f"Слияние временных данных: ts_scores_to_merge имеет {len(ts_scores_to_merge)} строк, {ts_scores_to_merge['merge_date_ts'].nunique()} уникальных дат")
            
            # Слияние по дате (без времени)
            result_df = result_df.merge(ts_scores_to_merge, on='merge_date_ts', how='left')
            
            # Заполняем пропуски после слияния средним значением вместо NaN
            if 'time_series_level_score' in result_df.columns and result_df['time_series_level_score'].isnull().any():
                mean_score = result_df['time_series_level_score'].mean()
                if pd.isna(mean_score): # На случай, если все значения NaN
                    mean_score = 0.5
                result_df['time_series_level_score'] = result_df['time_series_level_score'].fillna(mean_score)
                logger.info(f"Заполнено {result_df['time_series_level_score'].isnull().sum()} пропусков в time_series_level_score средним значением {mean_score:.4f}")
            
            result_df.drop(columns=['merge_date_ts'], inplace=True)  # Удаляем временную колонку
            
            result_df['time_series_is_anomaly'] = result_df['time_series_level_score'] > time_series_threshold
        
        if 'time_series_level_score' not in result_df.columns:
            result_df['time_series_level_score'] = np.nan
            result_df['time_series_is_anomaly'] = False
            if aggregated_ts_data_with_scores is None:
                logger.info("Time series level detection returned no results or failed pre-merge.")

        # --- Добавляем логирование распределения скоров уровней --- 
        logger.debug(f"Распределение transaction_level_score: Min={result_df['transaction_level_score'].min():.4f}, Max={result_df['transaction_level_score'].max():.4f}, Mean={result_df['transaction_level_score'].mean():.4f}")
        logger.debug(f"Распределение behavior_level_score: Min={result_df['behavior_level_score'].min():.4f}, Max={result_df['behavior_level_score'].max():.4f}, Mean={result_df['behavior_level_score'].mean():.4f}")
        logger.debug(f"Распределение time_series_level_score: Min={result_df['time_series_level_score'].min():.4f}, Max={result_df['time_series_level_score'].max():.4f}, Mean={result_df['time_series_level_score'].mean():.4f}")
        # ----------------------------------------------------------

        # Вычисляем финальный многоуровневый скор
        logger.info("Расчет итогового многоуровневого скора...")
        result_df['multilevel_score'] = (
            self.level_weights.get('transaction', 0.4) * result_df['transaction_level_score'].fillna(0) +
            self.level_weights.get('behavior', 0.5) * result_df['behavior_level_score'].fillna(0) +
            self.level_weights.get('time_series', 0.2) * result_df['time_series_level_score'].fillna(0)
        )
        
        # --- Логируем распределение итогового скора ---
        logger.debug(f"Распределение multilevel_score (перед порогом): Min={result_df['multilevel_score'].min():.4f}, Max={result_df['multilevel_score'].max():.4f}, Mean={result_df['multilevel_score'].mean():.4f}")
        # ---------------------------------------------

        # Определяем итоговые аномалии
        result_df['is_anomaly'] = result_df['multilevel_score'] > final_threshold
        
        # --- Генерируем объяснения для аномалий ---
        # Порог вклада детектора для включения в объяснение
        DETECTOR_CONTRIBUTION_THRESHOLD = 0.3  # Снижено с 0.4 до 0.2 для проверки гипотезы
        result_df['detailed_explanations_json'] = None
        
        # Получаем список всех колонок со скорами детекторов
        trans_detector_cols = [col for col in result_df.columns if col.startswith('trans_') and col.endswith('_score')]
        behav_detector_cols = [col for col in result_df.columns if col.startswith('behav_') and col.endswith('_score')]
        ts_detector_cols = [col for col in result_df.columns if col.startswith('ts_') and col.endswith('_score')]
        
        # Формируем объяснения только для обнаруженных аномалий
        for idx, row in result_df[result_df['is_anomaly'] == True].iterrows():
            # Инициализируем структуру объяснений
            explanations = {
                "transaction_level": [],
                "behavior_level": [],
                "time_series_level": []
            }
            
            # Анализируем транзакционный уровень
            if row['transaction_is_anomaly']:
                for col in trans_detector_cols:
                    detector_name = col[6:-6]  # Убираем 'trans_' и '_score'
                    # Добавляем логирование для проверки скоров и порога
                    score_value = row[col]
                    score_to_log_str = f"{score_value:.4f}" if pd.notna(score_value) else "NaN"
                    passes_threshold_str = str(score_value > DETECTOR_CONTRIBUTION_THRESHOLD) if pd.notna(score_value) else "Score is NaN"
                    
                    logger.debug(
                        f"Anomaly Explanation Check (idx: {idx}, order_id: {row.get('order_id', 'N/A')}, level: transaction): "
                        f"Detector Column: '{col}', Score: {score_to_log_str}, "
                        f"Threshold: {DETECTOR_CONTRIBUTION_THRESHOLD}, "
                        f"Passes Threshold: {passes_threshold_str}"
                    )
                    if pd.notna(row[col]) and row[col] > DETECTOR_CONTRIBUTION_THRESHOLD:
                        # Получаем детектор из словаря транзакционных детекторов
                        detector = self.transaction_detectors.get(detector_name)
                        logger.debug(f"    Detector instance for '{detector_name}': {type(detector).__name__ if detector else 'NOT FOUND'}")
                        if detector:
                            # Создаем DataFrame с одной строкой для получения объяснения
                            row_df = pd.DataFrame([row])
                            # Логируем данные, передаваемые в get_explanation_details
                            logger.debug(f"Data for get_explanation_details for transaction detector '{detector_name}':\n{row_df.head(1).to_string()}")
                            try:
                                # Получаем объяснение от детектора
                                detector_explanation = detector.get_explanation_details(row_df)
                                # Логируем результат вызова
                                import json
                                logger.debug(f"Explanation from transaction detector '{detector_name}': {json.dumps(detector_explanation, default=str, ensure_ascii=False)}")
                                if detector_explanation:
                                    logger.debug(f"    Explanation for '{detector_name}' IS NOT EMPTY. Appending to list.")
                                    explanation_entry = {
                                        "detector_name": detector_name,
                                        "detector_type": detector.__class__.__name__,
                                        "score": float(row[col]),
                                        "explanation": detector_explanation[0] if isinstance(detector_explanation, list) and detector_explanation else detector_explanation if isinstance(detector_explanation, dict) else {}
                                    }
                                    explanations["transaction_level"].append(explanation_entry)
                                    logger.debug(f"    Appended. Current explanations for transaction_level: {json.dumps(explanations['transaction_level'], default=str, ensure_ascii=False)}")
                                else:
                                    logger.warning(f"    Explanation for '{detector_name}' IS EMPTY or None. Not appending.")
                            except Exception as e:
                                logger.error(f"Ошибка при получении объяснения от детектора {detector_name}: {e}")
                        else:
                            logger.warning(f"    Детектор '{detector_name}' НЕ НАЙДЕН в словаре self.transaction_detectors!")
            
            # Анализируем поведенческий уровень
            if row['behavior_is_anomaly']:
                for col in behav_detector_cols:
                    detector_name = col[6:-6]  # Убираем 'behav_' и '_score'
                    # Добавляем логирование для проверки скоров и порога
                    score_value = row[col]
                    score_to_log_str = f"{score_value:.4f}" if pd.notna(score_value) else "NaN"
                    passes_threshold_str = str(score_value > DETECTOR_CONTRIBUTION_THRESHOLD) if pd.notna(score_value) else "Score is NaN"
                    
                    logger.debug(
                        f"Anomaly Explanation Check (idx: {idx}, order_id: {row.get('order_id', 'N/A')}, level: behavior): "
                        f"Detector Column: '{col}', Score: {score_to_log_str}, "
                        f"Threshold: {DETECTOR_CONTRIBUTION_THRESHOLD}, "
                        f"Passes Threshold: {passes_threshold_str}"
                    )
                    if pd.notna(row[col]) and row[col] > DETECTOR_CONTRIBUTION_THRESHOLD:
                        detector = self.behavior_detectors.get(detector_name)
                        logger.debug(f"    Detector instance for '{detector_name}': {type(detector).__name__ if detector else 'NOT FOUND'}")
                        if detector:
                            # Для поведенческого детектора нужно использовать агрегированные данные
                            if isinstance(detector, GraphAnomalyDetector):
                                # Для графового детектора используем исходные данные
                                row_df = pd.DataFrame([row])
                                logger.debug(f"    Using original row data for graph detector")
                            else:
                                # Для других поведенческих детекторов нужны агрегированные данные
                                if 'seller_id' in row and aggregated_behavior_data_with_scores is not None:
                                    seller_id = row['seller_id']
                                    logger.debug(f"    Looking for seller_id={seller_id} in aggregated_behavior_data_with_scores")
                                    agg_row = aggregated_behavior_data_with_scores[aggregated_behavior_data_with_scores['seller_id'] == seller_id]
                                    if not agg_row.empty:
                                        row_df = agg_row
                                        logger.debug(f"    Found matching seller_id row in aggregated data")
                                    else:
                                        logger.warning(f"    seller_id={seller_id} NOT FOUND in aggregated_behavior_data_with_scores!")
                                        continue
                                else:
                                    logger.warning(f"    Missing 'seller_id' column or aggregated_behavior_data_with_scores is None!")
                                    continue
                            
                            # Логируем данные, передаваемые в get_explanation_details
                            logger.debug(f"Data for get_explanation_details for behavior detector '{detector_name}':\n{row_df.head(1).to_string()}")
                            try:
                                detector_explanation = detector.get_explanation_details(row_df)
                                # Логируем результат вызова
                                import json
                                logger.debug(f"Explanation from behavior detector '{detector_name}': {json.dumps(detector_explanation, default=str, ensure_ascii=False)}")
                                if detector_explanation:
                                    logger.debug(f"    Explanation for '{detector_name}' IS NOT EMPTY. Appending to list.")
                                    explanation_entry = {
                                        "detector_name": detector_name,
                                        "detector_type": detector.__class__.__name__,
                                        "score": float(row[col]),
                                        "explanation": detector_explanation[0] if isinstance(detector_explanation, list) and detector_explanation else detector_explanation if isinstance(detector_explanation, dict) else {}
                                    }
                                    explanations["behavior_level"].append(explanation_entry)
                                    logger.debug(f"    Appended. Current explanations for behavior_level: {json.dumps(explanations['behavior_level'], default=str, ensure_ascii=False)}")
                                else:
                                    logger.warning(f"    Explanation for '{detector_name}' IS EMPTY or None. Not appending.")
                            except Exception as e:
                                logger.error(f"Ошибка при получении объяснения от детектора {detector_name}: {e}")
                        else:
                            logger.warning(f"    Детектор '{detector_name}' НЕ НАЙДЕН в словаре self.behavior_detectors!")
            
            # Анализируем временной уровень
            if row['time_series_is_anomaly']:
                for col in ts_detector_cols:
                    detector_name = col[3:-6]  # Убираем 'ts_' и '_score'
                    # Добавляем логирование для проверки скоров и порога
                    score_value = row[col]
                    score_to_log_str = f"{score_value:.4f}" if pd.notna(score_value) else "NaN"
                    passes_threshold_str = str(score_value > DETECTOR_CONTRIBUTION_THRESHOLD) if pd.notna(score_value) else "Score is NaN"
                    
                    logger.debug(
                        f"Anomaly Explanation Check (idx: {idx}, order_id: {row.get('order_id', 'N/A')}, level: time_series): "
                        f"Detector Column: '{col}', Score: {score_to_log_str}, "
                        f"Threshold: {DETECTOR_CONTRIBUTION_THRESHOLD}, "
                        f"Passes Threshold: {passes_threshold_str}"
                    )
                    if pd.notna(row[col]) and row[col] > DETECTOR_CONTRIBUTION_THRESHOLD:
                        detector = self.time_series_detectors.get(detector_name)
                        logger.debug(f"    Detector instance for '{detector_name}': {type(detector).__name__ if detector else 'NOT FOUND'}")
                        if detector:
                            # Для временного детектора нужно использовать временные данные
                            if 'order_purchase_timestamp' in row and aggregated_ts_data_with_scores is not None:
                                # Используем полночь даты для сопоставления (Timestamp формат)
                                current_date_for_lookup = pd.to_datetime(row['order_purchase_timestamp'].date())
                                
                                date_col_in_agg_ts = None
                                # Приоритет колонке 'date', если она datetime64
                                if 'date' in aggregated_ts_data_with_scores.columns and \
                                   pd.api.types.is_datetime64_any_dtype(aggregated_ts_data_with_scores['date']):
                                    agg_dates_for_lookup = aggregated_ts_data_with_scores['date'] # Это уже datetime64
                                    date_col_in_agg_ts = 'date'
                                # Запасной вариант, если вдруг осталась 'merge_date_ts' (хотя по идее ее быть не должно на этом этапе)
                                elif 'merge_date_ts' in aggregated_ts_data_with_scores.columns and \
                                     pd.api.types.is_datetime64_any_dtype(aggregated_ts_data_with_scores['merge_date_ts']):
                                    agg_dates_for_lookup = aggregated_ts_data_with_scores['merge_date_ts']
                                    date_col_in_agg_ts = 'merge_date_ts'
                                
                                if date_col_in_agg_ts:
                                    logger.debug(f"    Looking for date={current_date_for_lookup} in aggregated_ts_data_with_scores using column '{date_col_in_agg_ts}'")
                                    # Сравнение Timestamp с Timestamp (или серией Timestamp)
                                    ts_row = aggregated_ts_data_with_scores[agg_dates_for_lookup == current_date_for_lookup]
                                    if not ts_row.empty:
                                        row_df = ts_row.iloc[[0]] # Берем первую совпадающую строку
                                        logger.debug(f"    Found matching date row(s) in aggregated data for {current_date_for_lookup}")
                                    else:
                                        logger.warning(f"    date={current_date_for_lookup} NOT FOUND in aggregated_ts_data_with_scores (column: {date_col_in_agg_ts}) for time series detector '{detector_name}'!")
                                        continue
                                else:
                                    logger.warning(f"    Не найдена колонка с датой ('date' или 'merge_date_ts' с типом datetime64) в aggregated_ts_data_with_scores для time series detector '{detector_name}'!")
                                    continue
                            else:
                                logger.warning(f"    Missing 'order_purchase_timestamp' column or aggregated_ts_data_with_scores is None!")
                                continue
                            
                            # Логируем данные, передаваемые в get_explanation_details
                            logger.debug(f"Data for get_explanation_details for time_series detector '{detector_name}':\n{row_df.head(1).to_string()}")    
                            try:
                                detector_explanation = detector.get_explanation_details(row_df)
                                # Логируем результат вызова
                                import json
                                logger.debug(f"Explanation from time_series detector '{detector_name}': {json.dumps(detector_explanation, default=str, ensure_ascii=False)}")
                                if detector_explanation:
                                    logger.debug(f"    Explanation for '{detector_name}' IS NOT EMPTY. Appending to list.")
                                    explanation_entry = {
                                        "detector_name": detector_name,
                                        "detector_type": detector.__class__.__name__,
                                        "score": float(row[col]),
                                        "explanation": detector_explanation[0] if isinstance(detector_explanation, list) and detector_explanation else detector_explanation if isinstance(detector_explanation, dict) else {}
                                    }
                                    explanations["time_series_level"].append(explanation_entry)
                                    logger.debug(f"    Appended. Current explanations for time_series_level: {json.dumps(explanations['time_series_level'], default=str, ensure_ascii=False)}")
                                else:
                                    logger.warning(f"    Explanation for '{detector_name}' IS EMPTY or None. Not appending.")
                            except Exception as e:
                                logger.error(f"Ошибка при получении объяснения от детектора {detector_name}: {e}")
                        else:
                            logger.warning(f"    Детектор '{detector_name}' НЕ НАЙДЕН в словаре self.time_series_detectors!")
            
            # Сериализуем объяснения в JSON
            import json
            # Функция для преобразования неподходящих для JSON объектов
            def json_serializable(obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            try:
                result_df.at[idx, 'detailed_explanations_json'] = json.dumps(explanations, default=json_serializable)
            except Exception as e:
                logger.error(f"Ошибка при сериализации объяснений в JSON: {e}")
                result_df.at[idx, 'detailed_explanations_json'] = json.dumps({"error": str(e)})
                
        # Считаем количество аномалий с объяснениями
        anomalies_with_explanations = result_df['is_anomaly'] & result_df['detailed_explanations_json'].notna()
        logger.info(f"Сгенерированы объяснения для {anomalies_with_explanations.sum()} из {result_df['is_anomaly'].sum()} аномалий")

        # --- Переименовываем колонки для совместимости с сервисом сохранения ---
        rename_map = {
            'transaction_level_score': 'transaction_score',
            'behavior_level_score': 'behavior_score',
            'time_series_level_score': 'time_series_score',
            'multilevel_score': 'final_score' 
            # Дополнительно можно переименовать флаги is_anomaly уровней, если нужно
            # 'transaction_is_anomaly': 'transaction_anomaly',
            # 'behavior_is_anomaly': 'behavior_anomaly',
            # 'time_series_is_anomaly': 'time_series_anomaly'
        }
        result_df.rename(columns=rename_map, inplace=True)
        # ---------------------------------------------------------------------

        logger.info(f"Многоуровневая детекция завершена. Обнаружено {result_df['is_anomaly'].sum()} аномалий.")
        return result_df
        
    def _detect_transaction_level(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Выполняет обнаружение аномалий на транзакционном уровне.
        Возвращает DataFrame с результатами или None, если нет детекторов.
        """
        level_name = "transaction"
        logger.info(f"Запуск анализа на уровне: {level_name}...")

        if not self.transaction_detectors:
            logger.info(f"Нет детекторов для уровня '{level_name}'.")
            return None
            
        # Используем общий метод для получения нормализованных скоров всех детекторов
        normalized_scores_df, processed_detector_names = self._get_normalized_scores_from_detectors(
            data_to_process=data,
            detectors=self.transaction_detectors,
            level_name=level_name
        )

        if normalized_scores_df.empty or not processed_detector_names:
            logger.warning(f"Уровень '{level_name}': Не получено скоров от детекторов.")
            # Возвращаем исходные данные с колонкой transaction_score из NaN
            results_df = data.copy()
            results_df['transaction_score'] = np.nan
            return results_df

        # Собираем только те скоры, которые были успешно обработаны (не все NaN)
        valid_scores_list = []
        valid_detector_names_for_combination = []
        for name in processed_detector_names:
            if name in normalized_scores_df.columns and not normalized_scores_df[name].isnull().all():
                valid_scores_list.append(normalized_scores_df[name].values)
                valid_detector_names_for_combination.append(name)
            else:
                logger.warning(f"Уровень '{level_name}': Скоры от детектора '{name}' все NaN или отсутствуют, не используются в комбинировании.")

        if not valid_scores_list:
            logger.warning(f"Уровень '{level_name}': Нет валидных скоров для комбинирования.")
            results_df = data.copy()
            results_df['transaction_score'] = np.nan
            return results_df

        # Комбинируем скоры
        combined_level_scores = self._combine_scores_for_single_level(
            level_name=level_name,
            normalized_scores_list=valid_scores_list,
            detector_names=valid_detector_names_for_combination, # Передаем имена только тех детекторов, чьи скоры используются
            detector_weights=self.transaction_detector_weights,
            combination_method=self.transaction_level_combination_method
        )

        results_df = data.copy()
        results_df['transaction_score'] = combined_level_scores
        
        # Добавляем индивидуальные нормализованные скоры для детального анализа и объяснений
        for det_name in processed_detector_names:
            if det_name in normalized_scores_df.columns:
                results_df[f'trans_{det_name}_score'] = normalized_scores_df[det_name]

        logger.info(f"Анализ на уровне '{level_name}' завершен. Добавлен столбец 'transaction_score' и индивидуальные скоры детекторов.")
        return results_df
    
    def _detect_behavior_level(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Выполняет обнаружение аномалий на поведенческом уровне.
        Возвращает DataFrame с результатами или None.
        """
        level_name = "behavior"
        logger.info(f"Запуск анализа на уровне: {level_name}...")

        if not self.behavior_detectors:
            logger.info(f"Нет детекторов для уровня '{level_name}'.")
            return None

        # 1. Подготовка данных для поведенческого уровня
        data_for_level = self._prepare_behavior_data(data)
        if data_for_level is None or data_for_level.empty:
            logger.warning(f"Уровень '{level_name}': нет данных после подготовки.")
            return None # Возвращаем None, если нет данных для анализа

        # Создаем результирующий DataFrame для хранения скоров от всех детекторов
        all_behavior_scores_df = pd.DataFrame(index=data_for_level.index)
        processed_detector_names = []
        
        # Сохраняем результаты от каждого детектора отдельно для последующего анализа и объяснений
        detector_specific_results = {}
        
        # 2. Обрабатываем каждый детектор с соответствующими данными
        for detector_name, detector_instance in self.behavior_detectors.items():
            data_to_process = data_for_level  # По умолчанию используем агрегированные данные
            
            # Для GraphAnomalyDetector используем оригинальные данные, а не агрегированные
            if isinstance(detector_instance, GraphAnomalyDetector):
                logger.info(f"Детектор '{detector_name}' типа GraphAnomalyDetector получит оригинальные данные, а не агрегированные")
                data_to_process = data  # Используем исходные данные для графового детектора
            
            try:
                logger.info(f"Уровень '{level_name}': Запуск детектора '{detector_name}'...")
                
                # Check if the detector is trained if needed
                if not detector_instance.is_trained and not isinstance(detector_instance, (StatisticalDetector, GraphAnomalyDetector)):
                    # Проверяем, был ли метод train переопределен в классе экземпляра detector_instance
                    # по сравнению с базовым классом AnomalyDetector
                    train_method_of_instance_class = getattr(type(detector_instance), 'train', None)
                    train_method_of_base_class = getattr(AnomalyDetector, 'train', None)

                    is_train_overridden = False
                    if train_method_of_instance_class and train_method_of_base_class:
                        if train_method_of_instance_class is not train_method_of_base_class:
                            is_train_overridden = True

                    if is_train_overridden:
                        logger.warning(f"Детектор '{detector_name}' на уровне '{level_name}' не обучен. Пропускаем.")
                        all_behavior_scores_df[detector_name] = np.nan
                        continue
                
                # Запускаем детекцию с соответствующими данными
                detection_results_df = detector_instance.detect(data_to_process.copy())
                
                # Сохраняем результаты детектора для формирования объяснений позже
                detector_specific_results[detector_name] = detection_results_df.copy()
                
                if 'anomaly_score' not in detection_results_df.columns:
                    logger.warning(f"Детектор '{detector_name}' не вернул 'anomaly_score'. Пропускаем.")
                    all_behavior_scores_df[detector_name] = np.nan
                    continue
                
                # Для GraphAnomalyDetector нужно агрегировать скоры по seller_id
                if isinstance(detector_instance, GraphAnomalyDetector):
                    if 'seller_id' in detection_results_df.columns:
                        # Агрегируем скоры от графового детектора по seller_id
                        graph_scores_aggregated = detection_results_df.groupby('seller_id')['anomaly_score'].mean().reset_index()
                        # Делаем merge с data_for_level по seller_id
                        merged_df = pd.merge(data_for_level, graph_scores_aggregated, on='seller_id', how='left')
                        # Нормализуем скоры
                        raw_scores = merged_df['anomaly_score'].values.astype(float)
                    else:
                        logger.warning(f"GraphAnomalyDetector '{detector_name}' не вернул колонку 'seller_id'. Нельзя агрегировать скоры.")
                        all_behavior_scores_df[detector_name] = np.nan
                        continue
                else:
                    raw_scores = detection_results_df['anomaly_score'].values.astype(float)
                
                # Нормализация скоров
                if hasattr(detector_instance, 'normalize_score'):
                    normalized_scores = detector_instance.normalize_score(raw_scores)
                    all_behavior_scores_df[detector_name] = normalized_scores
                    processed_detector_names.append(detector_name)
                    logger.info(f"Детектор '{detector_name}' обработан. Распределение нормализованных скоров: Min={np.nanmin(normalized_scores):.4f}, Max={np.nanmax(normalized_scores):.4f}, Mean={np.nanmean(normalized_scores):.4f}")
                else:
                    logger.warning(f"Детектор '{detector_name}' не имеет метода normalize_score")
                    all_behavior_scores_df[detector_name] = np.nan
            
            except Exception as e:
                logger.error(f"Ошибка при выполнении детектора '{detector_name}' на уровне '{level_name}': {e}", exc_info=True)
                all_behavior_scores_df[detector_name] = np.nan
        
        if not processed_detector_names:
            logger.warning(f"Уровень '{level_name}': Не получено скоров от детекторов.")
            results_df = data_for_level.copy()
            results_df['behavior_score'] = np.nan
            return results_df

        # Собираем только те скоры, которые были успешно обработаны
        valid_scores_list = []
        valid_detector_names_for_combination = []
        for name in processed_detector_names:
            if name in all_behavior_scores_df.columns and not all_behavior_scores_df[name].isnull().all():
                valid_scores_list.append(all_behavior_scores_df[name].values)
                valid_detector_names_for_combination.append(name)
            else:
                logger.warning(f"Уровень '{level_name}': Скоры от детектора '{name}' все NaN или отсутствуют, не используются в комбинировании.")

        if not valid_scores_list:
            logger.warning(f"Уровень '{level_name}': Нет валидных скоров для комбинирования.")
            results_df = data_for_level.copy()
            results_df['behavior_score'] = np.nan
            return results_df

        # 3. Комбинируем скоры
        combined_level_scores = self._combine_scores_for_single_level(
            level_name=level_name,
            normalized_scores_list=valid_scores_list,
            detector_names=valid_detector_names_for_combination,
            detector_weights=self.behavior_detector_weights,
            combination_method=self.behavior_level_combination_method
        )
        
        # 4. Формируем DataFrame с результатами для этого уровня.
        results_df = data_for_level.copy()
        results_df['behavior_score'] = combined_level_scores
        
        # Добавляем индивидуальные нормализованные скоры для детального анализа и объяснений
        for det_name in processed_detector_names:
            if det_name in all_behavior_scores_df.columns:
                results_df[f'behav_{det_name}_score'] = all_behavior_scores_df[det_name]

        logger.info(f"Анализ на уровне '{level_name}' завершен. Добавлен столбец 'behavior_score' и индивидуальные скоры детекторов.")
        return results_df
    
    def _prepare_behavior_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Подготавливает данные для поведенческого уровня детекции.
        
        Агрегирует данные по продавцам/покупателям, вычисляя статистические 
        показатели их поведения (средние цены, стандартные отклонения, и т.д.)
        
        Args:
            data: Исходные данные транзакций
            
        Returns:
            Агрегированный DataFrame или None в случае ошибки
        """
        try:
            # Проверяем, есть ли нужные столбцы
            required_columns = ['seller_id', 'price', 'freight_value', 'product_category_name']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                logger.warning(f"Отсутствуют необходимые столбцы для поведенческой детекции: {missing}")
                return None
                
            # Агрегируем данные по продавцам
            seller_agg = data.groupby('seller_id').agg(
                mean_price=('price', 'mean'),
                std_price=('price', 'std'),
                count=('price', 'count'),
                min_price=('price', 'min'),
                max_price=('price', 'max'),
                mean_freight=('freight_value', 'mean'),
                std_freight=('freight_value', 'std'),
                categories=('product_category_name', lambda x: list(x.unique()))
            ).reset_index()
            
            # Вычисляем количество уникальных категорий
            seller_agg['category_count'] = seller_agg['categories'].apply(len)

            # Вычисляем дополнительные признаки для поведенческой аналитики
            seller_agg['price_volatility'] = seller_agg['std_price'] / seller_agg['mean_price'].replace(0, np.nan)
            seller_agg['price_range_ratio'] = seller_agg['max_price'] / seller_agg['min_price'].replace(0, np.nan)
            seller_agg['freight_to_price_ratio'] = seller_agg['mean_freight'] / seller_agg['mean_price'].replace(0, np.nan)
            
            # Заполняем NaN и бесконечности
            seller_agg = seller_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # --- ДОБАВЛЯЕМ ВЫЧИСЛЕНИЕ РАЗНООБРАЗИЯ КАТЕГОРИЙ --- 
            if 'product_category_name' in data.columns:
                # Агрегируем категории для каждого продавца
                seller_category = data.groupby(['seller_id', 'product_category_name']).size().reset_index(name='category_count')
                seller_unique_categories = seller_category.groupby('seller_id').size().reset_index(name='unique_categories')
                
                # Объединяем с основными агрегатами
                seller_agg = pd.merge(seller_agg, seller_unique_categories, on='seller_id', how='left')
                seller_agg['unique_categories'] = seller_agg['unique_categories'].fillna(0).astype(int)
                
                # Вычисляем индекс разнообразия категорий 
                seller_agg['category_diversity'] = seller_agg['unique_categories'] / seller_agg['count']
                
                # Вычисляем нормализованное разнообразие категорий
                max_categories = seller_agg['unique_categories'].max()
                if max_categories > 0:
                    seller_agg['normalized_diversity'] = np.log1p(seller_agg['unique_categories']) / np.log1p(max_categories)
                else:
                    seller_agg['normalized_diversity'] = 0.0
                    
                # Заполняем NaN, которые могли возникнуть при делении
                seller_agg[['category_diversity', 'normalized_diversity']] = seller_agg[['category_diversity', 'normalized_diversity']].fillna(0)
            else:
                 logger.warning("_prepare_behavior_data: Отсутствует 'product_category_name', признаки разнообразия не будут вычислены.")
                 seller_agg['category_diversity'] = 0.0
                 seller_agg['normalized_diversity'] = 0.0
            # ------------------------------------------------------
            
            return seller_agg
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для поведенческого уровня: {e}")
            return None
    
    def _detect_time_series_level(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Выполняет обнаружение аномалий во временных рядах.
        Возвращает DataFrame с результатами или None.
        """
        level_name = "time_series"
        logger.info(f"Запуск анализа на уровне: {level_name}...")

        if not self.time_series_detectors:
            logger.info(f"Нет детекторов для уровня '{level_name}'.")
            return None

        # 1. Подготовка данных для уровня временных рядов
        # Этот метод должен вернуть DataFrame, где каждая строка - это точка во времени,
        # а колонки - это временные ряды для разных сущностей (например, product_category_id)
        # или один агрегированный временной ряд.
        data_for_level = self._prepare_time_series_data(data)
        if data_for_level is None or data_for_level.empty:
            logger.warning(f"Уровень '{level_name}': нет данных после подготовки.")
            return None

        # 2. Используем общий метод для получения нормализованных скоров
        # data_to_process для детекторов временных рядов может быть разной.
        # Некоторые могут ожидать один ряд (pd.Series), другие - DataFrame с несколькими рядами.
        # _get_normalized_scores_from_detectors ожидает DataFrame. 
        # Детекторы временных рядов должны быть адаптированы для приема DataFrame и возврата DataFrame.
        normalized_scores_df, processed_detector_names = self._get_normalized_scores_from_detectors(
            data_to_process=data_for_level, 
            detectors=self.time_series_detectors,
            level_name=level_name
        )

        if normalized_scores_df.empty or not processed_detector_names:
            logger.warning(f"Уровень '{level_name}': Не получено скоров от детекторов.")
            results_df = data_for_level.copy() # Используем data_for_level, т.к. он имеет нужный индекс (обычно временной)
            results_df['time_series_score'] = np.nan
            return results_df

        # Собираем только те скоры, которые были успешно обработаны
        valid_scores_list = []
        valid_detector_names_for_combination = []
        for name in processed_detector_names:
            if name in normalized_scores_df.columns and not normalized_scores_df[name].isnull().all():
                valid_scores_list.append(normalized_scores_df[name].values)
                valid_detector_names_for_combination.append(name)
            else:
                logger.warning(f"Уровень '{level_name}': Скоры от детектора '{name}' все NaN или отсутствуют, не используются в комбинировании.")
        
        if not valid_scores_list:
            logger.warning(f"Уровень '{level_name}': Нет валидных скоров для комбинирования.")
            results_df = data_for_level.copy()
            results_df['time_series_score'] = np.nan
            return results_df

        # 3. Комбинируем скоры
        combined_level_scores = self._combine_scores_for_single_level(
            level_name=level_name,
            normalized_scores_list=valid_scores_list,
            detector_names=valid_detector_names_for_combination,
            detector_weights=self.time_series_detector_weights,
            combination_method=self.time_series_level_combination_method
        )
        
        # 4. Формируем DataFrame с результатами для этого уровня.
        # Индекс должен быть временным, как у data_for_level.
        results_df = data_for_level.copy() # Начинаем с data_for_level, чтобы сохранить его структуру/индекс
        
        # Добавляем явное создание колонки 'date' для консистентности при поиске объяснений
        if 'date' in results_df.columns and not pd.api.types.is_datetime64_any_dtype(results_df['date']):
            # Если 'date' уже есть, но это не datetime64 (например, object of datetime.date), конвертируем
            results_df['date'] = pd.to_datetime(results_df['date'])
        elif 'date' not in results_df.columns and 'order_purchase_timestamp' in results_df.columns:
            # Если 'date' нет, создаем из 'order_purchase_timestamp' как datetime64 (полночь)
            results_df['date'] = pd.to_datetime(pd.to_datetime(results_df['order_purchase_timestamp']).dt.date)
        
        results_df['time_series_score'] = combined_level_scores
        
        # Добавляем индивидуальные нормализованные скоры для детального анализа и объяснений
        for det_name in processed_detector_names:
            if det_name in normalized_scores_df.columns:
                results_df[f'ts_{det_name}_score'] = normalized_scores_df[det_name]

        logger.info(f"Анализ на уровне '{level_name}' завершен. Добавлен столбец 'time_series_score' и индивидуальные скоры детекторов.")
        return results_df # Этот DataFrame будет иметь временной индекс

    def _prepare_time_series_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Подготавливает данные для уровня временных рядов.
        
        Преобразует данные для анализа временных рядов, агрегируя по времени
        и вычисляя различные метрики (объем продаж, средние цены и т.д.).
        
        Args:
            data: Исходные данные транзакций
            
        Returns:
            Агрегированный DataFrame по времени или None в случае ошибки
        """
        try:
            # Проверяем, есть ли нужные столбцы
            if 'order_purchase_timestamp' not in data.columns:
                logger.warning("Отсутствует столбец 'order_purchase_timestamp' для анализа временных рядов")
                return None
                
            # Конвертируем временную метку в datetime, если нужно
            if not pd.api.types.is_datetime64_dtype(data['order_purchase_timestamp']):
                data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
            
            # Группируем данные по дням
            data['date'] = data['order_purchase_timestamp'].dt.date
            time_series = data.groupby('date').agg(
                order_count=('order_id', 'nunique'),
                total_sales=('price', 'sum'),
                avg_price=('price', 'mean'),
                unique_sellers=('seller_id', 'nunique'),
                unique_customers=('customer_id', 'nunique') if 'customer_id' in data.columns else ('order_id', 'count')
            ).reset_index()
            
            # Конвертируем обратно в datetime для последующего использования
            time_series['order_purchase_timestamp'] = pd.to_datetime(time_series['date'])
            
            # Вычисляем скользящие средние для выявления тренда
            time_series['sales_7d_avg'] = time_series['total_sales'].rolling(window=7, min_periods=1).mean()
            
            # Вычисляем отклонения от скользящего среднего
            time_series['sales_deviation'] = time_series['total_sales'] / time_series['sales_7d_avg'].replace(0, np.nan)
            
            # Заполняем пропуски (например, после resample)
            # Сначала ffill, чтобы заполнить промежутки, затем 0 для начальных NaN
            if time_series.empty:
                logger.warning(f"({self.model_name}) Агрегированный временной ряд пуст после подготовки и resampling.")
            else:
                # Сначала заменяем inf на NaN, чтобы ffill работал корректно
                time_series = time_series.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            
            return time_series
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для уровня временных рядов: {e}")
            return None
    
    def save(self, path: str) -> None:
        """
        Сохраняет конфигурацию MultilevelAnomalyDetector.
        
        Args:
            path: Путь для сохранения
        """
        try:
            save_dict = {
                'config': self.config,
                'model_base_path': self.model_base_path,
                'level_weights': self.level_weights
            }
            
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохраняем конфигурацию
            joblib.dump(save_dict, path)
            logger.info(f"Конфигурация MultilevelAnomalyDetector сохранена в {path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации MultilevelAnomalyDetector: {e}")
    
    def load(self, path: str) -> None:
        """
        Загружает конфигурацию MultilevelAnomalyDetector и переинициализирует детекторы.
        
        Args:
            path: Путь к сохраненному файлу
        """
        try:
            loaded_dict = joblib.load(path)
            
            self.config = loaded_dict.get('config', {})
            self.model_base_path = loaded_dict.get('model_base_path', "models")
            self.level_weights = loaded_dict.get('level_weights', {
                'transaction': 0.4,
                'behavior': 0.4, 
                'time_series': 0.2
            })
            
            # Перезагружаем детекторы
            self._initialize_detectors()
            
            logger.info(f"Конфигурация MultilevelAnomalyDetector загружена из {path}")
            
        except FileNotFoundError:
            logger.error(f"Файл конфигурации {path} не найден")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации MultilevelAnomalyDetector: {e}") 