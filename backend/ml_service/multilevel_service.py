# backend/ml_service/multilevel_service.py

import pandas as pd
import numpy as np
import os
import logging
import asyncio
import concurrent.futures
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime

from sqlalchemy.orm import Session

# --- Импорт настроек ---
from backend.config.config import get_settings
settings = get_settings()
# -----------------------

# --- Импорт для TaskManager ---
from backend.services.task_manager import task_manager
# --------------------------------

from .detector import AnomalyDetector
from .detector_factory import DetectorFactory
from .multilevel_detector import MultilevelAnomalyDetector
from .graph_detector import GraphAnomalyDetector
from .common import load_data_from_db, engineer_features
from .. import crud, models # Относительный импорт для доступа к корневым crud, models
from .. import schemas as root_schemas # Используем алиас для корневых схем
# НЕ импортируем SessionLocal напрямую на уровне модуля, чтобы избежать проблем с None при reload
# from ..database import SessionLocal

logger = logging.getLogger("backend.ml_service.multilevel_service")

class MultilevelDetectorService:
    def __init__(self, config_filename: str = "multilevel_config.joblib"):
        self.model_base_path = settings.common.model_base_path
        os.makedirs(self.model_base_path, exist_ok=True)
        logger.info(f"Базовый путь для моделей (MultilevelService): {self.model_base_path}")

        self.config_filename = config_filename

        self.multilevel_detector: Optional[MultilevelAnomalyDetector] = None
        self._load_or_create_detector()

    def _load_or_create_detector(self):
        # Старая логика загрузки всего объекта MultilevelAnomalyDetector из joblib файла удаляется.
        # Теперь детектор всегда создается на основе конфигурации из settings.yaml.
        logger.info(f"Создание/инициализация многоуровневой системы на основе настроек из config.yaml.")
        self._create_default_detector_from_settings()

    def _create_default_detector_from_settings(self):
        try:
            # multilevel_detector_default_config теперь содержит всю необходимую структуру,
            # включая *_level_combination_method и веса для каждого детектора.
            default_config_pydantic = settings.ml_service.multilevel_detector_default_config
            if default_config_pydantic is None:
                logger.error("Конфигурация 'ml_service.multilevel_detector_default_config' не найдена в settings.yaml.")
                self.multilevel_detector = None
                return
                
            default_config_dict = default_config_pydantic.model_dump(exclude_none=True)
            
            logger.debug(f"Используется следующая конфигурация для MultilevelAnomalyDetector: {json.dumps(default_config_dict, indent=2)}")
            
            self.multilevel_detector = MultilevelAnomalyDetector(
                config=default_config_dict, # Передаем всю структуру конфигурации
                model_base_path=self.model_base_path
            )
            logger.info(f"Многоуровневая система успешно создана на основе конфигурации из settings.yaml.")
        except AttributeError as ae:
             logger.error(f"Ошибка доступа к multilevel_detector_default_config в настройках: {ae}. "
                          "Убедитесь, что секция ml_service.multilevel_detector_default_config правильно определена в config.yaml и config.py.", exc_info=True)
             self.multilevel_detector = None
        except Exception as e:
            logger.error(f"Критическая ошибка при создании детектора на основе настроек: {e}", exc_info=True)
            self.multilevel_detector = None

    def train_task_wrapper(self,
                           task_id: str, 
                           data: Optional[pd.DataFrame] = None, 
                           load_data_params: Optional[Dict[str, Any]] = None):
        """Обертка для метода train для выполнения в фоновом режиме с обновлением статуса задачи."""
        logger.info(f"[TaskID: {task_id}] Обертка train_task_wrapper запущена.")
        task_manager.update_task_status(task_id, status="processing", details="Подготовка и обучение моделей...")
        try:
            success = self.train(data=data, load_data_params=load_data_params)
            if success:
                task_manager.update_task_status(task_id, status="completed", details="Обучение успешно завершено.")
                logger.info(f"[TaskID: {task_id}] Обучение успешно завершено.")
            else:
                task_manager.update_task_status(task_id, status="failed", details="Ошибка во время обучения. См. логи сервера.", error_type="TrainingError")
                logger.warning(f"[TaskID: {task_id}] Обучение завершилось с ошибками.")
        except Exception as e:
            logger.error(f"[TaskID: {task_id}] Критическая ошибка в train_task_wrapper: {e}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", details=f"Критическая ошибка сервера: {str(e)}", error_type=type(e).__name__)

    def train(self,
              data: Optional[pd.DataFrame] = None,
              load_data_params: Optional[Dict[str, Any]] = None) -> bool:
        logger.info("Начало обучения многоуровневой системы...")

        if self.multilevel_detector is None:
            logger.warning("Многоуровневый детектор не инициализирован. Попытка создать из настроек...")
            self._create_default_detector_from_settings() # Пересоздаем из настроек
            if self.multilevel_detector is None:
                logger.error("Не удалось создать детектор для обучения на основе настроек.")
                return False

        try:
            from backend.database import SessionLocal as GlobalSessionLocal
            if GlobalSessionLocal is None:
                logger.error("GlobalSessionLocal не инициализирован! Невозможно создать сессию БД.")
                return False
            current_session_local = GlobalSessionLocal
        except ImportError:
            logger.error("Не удалось импортировать GlobalSessionLocal.")
            return False

        db_for_load: Optional[Session] = None
        try:
            training_data = data
            if training_data is None:
                logger.info("Загрузка данных из БД для обучения multilevel...")
                load_params = load_data_params if load_data_params is not None else {}
                load_params['load_associations'] = True
                db_for_load = current_session_local()
                training_data = load_data_from_db(db=db_for_load, **load_params)

            if training_data is None or training_data.empty:
                logger.error("Невозможно обучить: пустой DataFrame.")
                return False

            logger.info(f"Данные для обучения multilevel ({len(training_data)} строк) подготовлены. Инженерия признаков...")
            training_data_with_features = engineer_features(training_data)
            logger.info("Инженерия признаков завершена.")

            all_detectors_trained_successfully = True

            # 1. Транзакционный уровень
            if self.multilevel_detector.transaction_detectors:
                logger.info("Обучение детекторов транзакционного уровня...")
                for name, detector_instance in self.multilevel_detector.transaction_detectors.items():
                    try:
                        logger.info(f"  Обучение транзакционного детектора: {name}")
                        # Предполагается, что detector_instance.config содержит 'model_filename'
                        detector_config = self.multilevel_detector.config.get('transaction_level', [])
                        current_detector_conf = next((c for c in detector_config if DetectorFactory._generate_detector_name(c['type'],c) == name), None)
                        if not current_detector_conf:
                            logger.error(f"Не найдена конфигурация для детектора {name} на транзакционном уровне.")
                            all_detectors_trained_successfully = False
                            continue

                        detector_instance.train(training_data_with_features.copy()) # Обучаем
                        model_filename = current_detector_conf.get("model_filename")
                        if model_filename and detector_instance.is_trained:
                            model_path = os.path.join(self.model_base_path, model_filename)
                            detector_instance.save_model(model_path)
                            logger.info(f"  Детектор {name} обучен и сохранен в {model_path}")
                        elif not model_filename:
                            logger.warning(f"  Детектор {name} обучен, но model_filename не указан в конфигурации. Модель не сохранена.")
                        elif not detector_instance.is_trained:
                            logger.warning(f"  Детектор {name} не был помечен как обученный после вызова train(). Модель не сохранена.")
                            all_detectors_trained_successfully = False # Считаем это неудачей

                    except Exception as e_train_trans:
                        logger.error(f"Ошибка при обучении/сохранении транзакционного детектора {name}: {e_train_trans}", exc_info=True)
                        all_detectors_trained_successfully = False
            else:
                logger.info("Нет детекторов для обучения на транзакционном уровне.")

            # 2. Поведенческий уровень
            if self.multilevel_detector.behavior_detectors:
                logger.info("Обучение детекторов поведенческого уровня...")
                # Данные для "классических" поведенческих детекторов, агрегированные по продавцам
                behavior_input_data_aggregated = self.multilevel_detector._prepare_behavior_data(training_data_with_features)
                
                for name, detector_instance in self.multilevel_detector.behavior_detectors.items():
                    try:
                        logger.info(f"  Обучение поведенческого детектора: {name} (тип: {type(detector_instance).__name__})")
                        detector_config = self.multilevel_detector.config.get('behavior_level', [])
                        current_detector_conf = next((c for c in detector_config if DetectorFactory._generate_detector_name(c['type'],c) == name), None)
                        if not current_detector_conf:
                            logger.error(f"Не найдена конфигурация для детектора {name} на поведенческом уровне.")
                            all_detectors_trained_successfully = False
                            continue
                        
                        data_for_this_detector_train: Optional[pd.DataFrame] = None

                        if isinstance(detector_instance, GraphAnomalyDetector):
                            # GraphAnomalyDetector требует исходные транзакционные данные
                            data_for_this_detector_train = training_data_with_features.copy() 
                            logger.info(f"  Детектор {name} (GraphAnomalyDetector) будет использовать исходные транзакционные данные для обучения ({data_for_this_detector_train.shape}).")
                        elif behavior_input_data_aggregated is not None and not behavior_input_data_aggregated.empty:
                            # Другие поведенческие детекторы используют агрегированные данные
                            data_for_this_detector_train = behavior_input_data_aggregated.copy()
                            logger.info(f"  Детектор {name} будет использовать агрегированные поведенческие данные для обучения ({data_for_this_detector_train.shape}).")
                        else:
                            logger.warning(f"  Не удалось подготовить данные для обучения поведенческого детектора {name} (агрегированные данные пусты или None). Пропускаем.")
                            all_detectors_trained_successfully = False
                            continue

                        if data_for_this_detector_train is None or data_for_this_detector_train.empty: # Дополнительная проверка
                            logger.warning(f"  Финальные данные для обучения детектора {name} пусты. Пропускаем обучение.")
                            all_detectors_trained_successfully = False
                            continue
                        
                        detector_instance.train(data_for_this_detector_train) 
                        
                        # Сохранение модели детектора
                        model_filename = current_detector_conf.get("model_filename")
                        if model_filename and detector_instance.is_trained:
                            model_path = os.path.join(self.model_base_path, model_filename)
                            detector_instance.save_model(model_path)
                            logger.info(f"  Детектор {name} обучен и сохранен в {model_path}")
                        elif not model_filename:
                            logger.warning(f"  Детектор {name} обучен, но model_filename не указан. Модель не сохранена.")
                        elif not detector_instance.is_trained:
                            logger.warning(f"  Детектор {name} не обучен. Модель не сохранена.")
                            all_detectors_trained_successfully = False

                    except Exception as e_train_behav:
                        logger.error(f"Ошибка при обучении/сохранении поведенческого детектора {name}: {e_train_behav}", exc_info=True)
                        all_detectors_trained_successfully = False
            else:
                logger.info("Нет детекторов для обучения на поведенческом уровне.")

            # 3. Временной уровень
            if self.multilevel_detector.time_series_detectors:
                logger.info("Обучение детекторов временного уровня...")
                time_series_input_data = self.multilevel_detector._prepare_time_series_data(training_data_with_features)
                if time_series_input_data is not None and not time_series_input_data.empty:
                    for name, detector_instance in self.multilevel_detector.time_series_detectors.items():
                        try:
                            logger.info(f"  Обучение детектора временных рядов: {name}")
                            detector_config = self.multilevel_detector.config.get('time_series_level', [])
                            current_detector_conf = next((c for c in detector_config if DetectorFactory._generate_detector_name(c['type'],c) == name), None)
                            if not current_detector_conf:
                                logger.error(f"Не найдена конфигурация для детектора {name} на временном уровне.")
                                all_detectors_trained_successfully = False
                                continue

                            detector_instance.train(time_series_input_data.copy())
                            model_filename = current_detector_conf.get("model_filename")
                            if model_filename and detector_instance.is_trained:
                                model_path = os.path.join(self.model_base_path, model_filename)
                                detector_instance.save_model(model_path)
                                logger.info(f"  Детектор {name} обучен и сохранен в {model_path}")
                            elif not model_filename:
                                logger.warning(f"  Детектор {name} обучен, но model_filename не указан. Модель не сохранена.")
                            elif not detector_instance.is_trained:
                                logger.warning(f"  Детектор {name} не обучен. Модель не сохранена.")
                                all_detectors_trained_successfully = False
                                
                        except Exception as e_train_ts:
                            logger.error(f"Ошибка при обучении/сохранении детектора временных рядов {name}: {e_train_ts}", exc_info=True)
                            all_detectors_trained_successfully = False
                else:
                    logger.warning("Не удалось подготовить данные для обучения детекторов временных рядов (time_series_input_data пуст или None).")
            else:
                logger.info("Нет детекторов для обучения на временном уровне.")
            
            # Сохранение конфигурации MultilevelAnomalyDetector (self.config_path) больше не требуется,
            # т.к. все теперь управляется через config.yaml.
            # self.multilevel_detector.save(self.config_path) 

            if all_detectors_trained_successfully:
                logger.info("Обучение всех применимых детекторов в многоуровневой системе завершено.")
            else:
                logger.warning("Обучение многоуровневой системы завершено, но были ошибки при обучении/сохранении некоторых детекторов.")
            return all_detectors_trained_successfully

        except Exception as e:
            logger.error(f"Критическая ошибка при обучении многоуровневой системы: {e}", exc_info=True)
            return False

    def detect(
        self,
        data: Optional[pd.DataFrame] = None,
        load_data_params: Optional[Dict[str, Any]] = None,
        transaction_threshold: float = 0.6,
        behavior_threshold: float = 0.6,
        time_series_threshold: float = 0.6,
        final_threshold: float = 0.5
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        logger.info("Начало обнаружения аномалий (multilevel) с сохранением в БД...")
        save_stats = {
            "total_detected_anomalies_before_save": 0,
            "newly_saved_anomalies_count": 0,
            "newly_saved_anomaly_ids": [],
            "skipped_duplicates_count": 0,
            "errors_on_save": 0
        }

        if self.multilevel_detector is None:
            logger.warning("Многоуровневый детектор не инициализирован. Попытка загрузить/создать...")
            self._load_or_create_detector()
            if self.multilevel_detector is None:
                logger.error("Не удалось создать/загрузить детектор для обнаружения.")
                return None, save_stats

        # --- ИСПРАВЛЕНИЕ ДЛЯ SessionLocal ---
        try:
            from backend.database import SessionLocal as GlobalSessionLocal
            if GlobalSessionLocal is None:
                logger.error("GlobalSessionLocal не инициализирован в database.py! Невозможно создать сессию БД для detect.")
                return None, save_stats # Или пустой DataFrame
            current_session_local_for_detect = GlobalSessionLocal
        except ImportError:
            logger.error("Не удалось импортировать GlobalSessionLocal из backend.database для detect.")
            return None, save_stats
        # ------------------------------------

        db_for_crud: Optional[Session] = None
        db_for_load_detect: Optional[Session] = None

        try:
            detection_input_data = data
            if detection_input_data is None:
                logger.info("Загрузка данных из БД для multilevel detect...")
                load_params_detect = load_data_params if load_data_params is not None else {}
                load_params_detect['load_associations'] = True

                db_for_load_detect = current_session_local_for_detect() # Используем полученный SessionLocal
                detection_input_data = load_data_from_db(db=db_for_load_detect, **load_params_detect)

            if detection_input_data is None or detection_input_data.empty:
                logger.error("Невозможно выполнить multilevel detect: пустой DataFrame после загрузки/передачи.")
                return pd.DataFrame(), save_stats

            logger.info(f"Данные для multilevel detect ({len(detection_input_data)} строк) подготовлены. Применение инженерии признаков...")
            detection_data_with_features = engineer_features(detection_input_data)
            logger.info("Инженерия признаков для детекции завершена.")

            if detection_data_with_features.empty:
                logger.warning("DataFrame пуст после инженерии признаков для детекции.")
                return pd.DataFrame(), save_stats

            logger.info("Запуск multilevel_detector.detect()...")
            result_df = self.multilevel_detector.detect(
                detection_data_with_features,
                transaction_threshold=transaction_threshold,
                behavior_threshold=behavior_threshold,
                time_series_threshold=time_series_threshold,
                final_threshold=final_threshold
            )

            if result_df is None or result_df.empty:
                 logger.warning("multilevel_detector.detect() вернул пустой или None результат.")
                 return pd.DataFrame(), save_stats

            anomalies_df = result_df[result_df['is_anomaly']].copy()
            save_stats["total_detected_anomalies_before_save"] = len(anomalies_df)
            logger.info(f"Обнаружено {save_stats['total_detected_anomalies_before_save']} multilevel аномалий до сохранения.")

            if save_stats["total_detected_anomalies_before_save"] > 0:
                db_for_crud = current_session_local_for_detect() # Используем тот же SessionLocal
                logger.info("Начало сохранения multilevel аномалий в БД...")
                score_cols = ['transaction_score', 'behavior_score', 'time_series_score', 'final_score']

                for index, row in anomalies_df.iterrows():
                    order_id_val = row.get('order_id')
                    order_item_id_val = row.get('order_item_id')

                    if pd.isna(order_id_val) or pd.isna(order_item_id_val):
                        logger.warning(f"Пропуск сохранения multilevel аномалии (индекс {index}): отсутствует order_id или order_item_id.")
                        save_stats["errors_on_save"] += 1
                        continue

                    order_id_str = str(order_id_val)
                    try:
                        order_item_id_int = int(order_item_id_val)
                    except ValueError:
                        logger.warning(f"Не удалось преобразовать order_item_id '{order_item_id_val}' в int для multilevel аномалии (индекс {index}). Пропуск.")
                        save_stats["errors_on_save"] += 1
                        continue

                    existing_anomaly = db_for_crud.query(models.Anomaly).filter(
                        models.Anomaly.order_id == order_id_str,
                        models.Anomaly.order_item_id == order_item_id_int,
                        models.Anomaly.detector_type == "multilevel"
                    ).first()

                    if existing_anomaly:
                        save_stats["skipped_duplicates_count"] += 1
                        continue

                    anomaly_details = {
                        "product_id": str(row.get('product_id')) if pd.notna(row.get('product_id')) else None,
                        "seller_id": str(row.get('seller_id')) if pd.notna(row.get('seller_id')) else None,
                        "final_threshold_used": final_threshold,
                        "level_scores": {}
                    }
                    for col_name in score_cols:
                        if col_name in row and pd.notna(row[col_name]):
                            anomaly_details["level_scores"][col_name] = float(row[col_name])

                    anomaly_details["level_thresholds"] = {
                         "transaction": transaction_threshold,
                         "behavior": behavior_threshold,
                         "time_series": time_series_threshold
                    }
                    
                    # Добавляем подробные объяснения, если они есть
                    if 'detailed_explanations_json' in row and pd.notna(row['detailed_explanations_json']):
                        try:
                            # Если detailed_explanations_json уже является строкой JSON, парсим его
                            if isinstance(row['detailed_explanations_json'], str):
                                explanations = json.loads(row['detailed_explanations_json'])
                                anomaly_details['contributing_detectors_explanations'] = explanations
                            else:
                                # Если это не строка, пробуем использовать напрямую
                                anomaly_details['contributing_detectors_explanations'] = row['detailed_explanations_json']
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"Не удалось декодировать detailed_explanations_json как JSON: {json_err}. Сохраняем как строку.")
                            anomaly_details['contributing_detectors_explanations_raw'] = str(row['detailed_explanations_json'])
                        except Exception as e:
                            logger.warning(f"Ошибка при обработке detailed_explanations_json: {e}")
                    
                    # Безопасная сериализация в JSON
                    def safe_json_convert(obj):
                        if isinstance(obj, (np.integer, np.int_)): return int(obj)
                        elif isinstance(obj, (np.floating, np.float64)): return float(obj)
                        elif isinstance(obj, np.ndarray): return obj.tolist()
                        elif isinstance(obj, dict): return {k: safe_json_convert(v) for k, v in obj.items()}
                        elif isinstance(obj, list): return [safe_json_convert(i) for i in obj]
                        elif pd.isna(obj): return None
                        return obj
                    details_str = json.dumps(safe_json_convert(anomaly_details))


                    anomaly_score_val = row.get('final_score')
                    anomaly_score_to_save = float(anomaly_score_val) if pd.notna(anomaly_score_val) else 0.0

                    anomaly_data = root_schemas.AnomalyCreate(
                        order_id=order_id_str,
                        order_item_id=order_item_id_int,
                        detection_date=datetime.utcnow(),
                        anomaly_score=anomaly_score_to_save,
                        detector_type="multilevel",
                        details=details_str
                    )
                    try:
                        created_anomaly = crud.create_anomaly(db=db_for_crud, anomaly=anomaly_data)
                        save_stats["newly_saved_anomalies_count"] += 1
                        if created_anomaly and hasattr(created_anomaly, 'id'):
                             save_stats["newly_saved_anomaly_ids"].append(created_anomaly.id)
                    except Exception as create_exc:
                        logger.error(f"Ошибка при сохранении multilevel аномалии {order_id_str}-{order_item_id_int} в БД: {create_exc}", exc_info=True)
                        save_stats["errors_on_save"] +=1
                logger.info(f"Статистика сохранения multilevel аномалий: {save_stats}")

            return result_df, save_stats

        except Exception as e:
            logger.error(f"Критическая ошибка при обнаружении/сохранении multilevel аномалий: {e}", exc_info=True)
            return pd.DataFrame() if data is None or data.empty else data.copy(), save_stats
        finally:
            if db_for_crud:
                db_for_crud.close()
                logger.debug("Сессия БД (db_for_crud) для multilevel detect закрыта.")
            if db_for_load_detect:
                db_for_load_detect.close()
                logger.debug("Сессия БД (db_for_load_detect) для multilevel detect закрыта.")

    async def detect_async(
        self,
        data: Optional[pd.DataFrame] = None,
        load_data_params: Optional[Dict[str, Any]] = None,
        transaction_threshold: float = 0.6,
        behavior_threshold: float = 0.6,
        time_series_threshold: float = 0.6,
        final_threshold: float = 0.5
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        # Эта функция является оберткой над self.detect для асинхронного выполнения.
        # Основная логика остается в self.detect.
        # Убедимся, что все параметры правильно передаются.
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result_df, save_stats = await loop.run_in_executor(
                pool, 
                self.detect, # Вызываем синхронный метод detect
                data, 
                load_data_params, 
                transaction_threshold, 
                behavior_threshold, 
                time_series_threshold, 
                final_threshold
            )
        return result_df, save_stats

    def update_config(self, new_config_dict: Dict[str, Any]) -> bool:
        """
        Обновляет конфигурацию MultilevelAnomalyDetector и пересоздает его.
        Эта функция ПЕРЕЗАПИШЕТ текущую конфигурацию в settings.py/config.yaml (если бы была такая логика).
        В текущей реализации, это просто пересоздаст детектор с новой конфигурацией в памяти.
        Для сохранения изменений в config.yaml потребуется отдельный механизм.
        """
        logger.info("Попытка обновить конфигурацию MultilevelDetectorService...")
        try:
            # Валидация новой конфигурации (базовая)
            if not isinstance(new_config_dict, dict):
                logger.error("Ошибка обновления: новая конфигурация должна быть словарем.")
                return False
            # Здесь можно добавить более сложную валидацию схемы новой конфигурации
            
            self.multilevel_detector = MultilevelAnomalyDetector(
                config=new_config_dict, 
                model_base_path=self.model_base_path
            )
            # Логика сохранения этой new_config_dict в settings.ml_service.multilevel_detector_default_config
            # и последующего сохранения в config.yaml здесь не реализована.
            # Это потребует изменения pydantic модели Settings и записи в YAML файл.
            # settings.ml_service.multilevel_detector_default_config = ... # Нужна Pydantic модель
            # save_settings_to_yaml(settings) # Нужна функция сохранения
            logger.info("Конфигурация MultilevelAnomalyDetector в памяти обновлена. "
                        "Для постоянного сохранения необходимо обновить config.yaml вручную или реализовать механизм сохранения.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обновлении и пересоздании MultilevelAnomalyDetector: {e}", exc_info=True)
            # Восстанавливаем предыдущий детектор из текущих настроек, если обновление не удалось
            logger.info("Попытка восстановить детектор из текущих настроек config.yaml...")
            self._create_default_detector_from_settings()
            return False

    def get_config(self) -> Dict[str, Any]:
        """Возвращает текущую конфигурацию многоуровневой системы."""
        if self.multilevel_detector and self.multilevel_detector.config:
            return self.multilevel_detector.config
        logger.warning("Конфигурация для MultilevelAnomalyDetector не найдена или детектор не инициализирован.")
        # Можно вернуть конфигурацию из settings.yaml как запасной вариант
        try:
            default_config_pydantic = settings.ml_service.multilevel_detector_default_config
            if default_config_pydantic:
                return default_config_pydantic.model_dump(exclude_none=True)
        except Exception as e:
            logger.error(f"Не удалось получить конфигурацию из settings.ml_service.multilevel_detector_default_config: {e}")
        return {}

    def _get_single_detector_status_entry(
        self,
        detector_name: str,
        detector_instance: Any, # AnomalyDetector, но Any для гибкости
        level_name: str, # Для логирования
        level_configs_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Собирает информацию о статусе для одного детектора."""
        # from .detector_factory import DetectorFactory # Локальный импорт

        status_entry = {
            "detector_name": detector_name,
            "detector_type": "Unknown",
            "model_filename": "N/A",
            "expected_path": "N/A",
            "exists": False,
            "is_trained": detector_instance.is_trained if hasattr(detector_instance, 'is_trained') else False,
            "can_load": False,
            "error_message": None,
            "params_from_config": {},
            "internal_params": {}
        }

        original_config_for_detector = None
        for conf in level_configs_list:
            generated_name_from_conf = DetectorFactory._generate_detector_name(conf.get('type'), conf)
            if generated_name_from_conf == detector_name:
                original_config_for_detector = conf
                break
        
        if original_config_for_detector:
            status_entry["detector_type"] = original_config_for_detector.get('type', "Unknown")
            status_entry["model_filename"] = original_config_for_detector.get('model_filename', "N/A")
            # Копируем все параметры, кроме зарезервированных, которые уже есть или не нужны как "параметры"
            status_entry["params_from_config"] = {
                k: v for k, v in original_config_for_detector.items() 
                if k not in ['type', 'model_filename', 'weight']
            }
            if 'weight' in original_config_for_detector:
                status_entry["params_from_config"]['weight_for_level_combination'] = original_config_for_detector['weight']

            if status_entry["model_filename"] != "N/A":
                model_path = os.path.join(self.model_base_path, status_entry["model_filename"])
                status_entry["expected_path"] = model_path
                status_entry["exists"] = os.path.exists(model_path)
                
                if status_entry["exists"] and status_entry["is_trained"]:
                    status_entry["can_load"] = True
                elif status_entry["exists"] and not status_entry["is_trained"]:
                    status_entry["error_message"] = "Файл модели существует, но детектор не помечен как обученный (возможно, ошибка загрузки или модель не обучена)."
                elif not status_entry["exists"] and status_entry["is_trained"]:
                    # Это странный случай: файла нет, но is_trained=True. Возможно, модель не была сохранена или путь неверный.
                    status_entry["error_message"] = "Детектор помечен как обученный, но файл модели по указанному пути не найден."
        else:
            status_entry["error_message"] = f"Не удалось найти исходную конфигурацию для детектора '{detector_name}' (тип: {type(detector_instance).__name__}) на уровне '{level_name}'. Статус может быть неполным."
            logger.warning(status_entry["error_message"])
            # Попытаемся получить тип из самого инстанса, если конфиг не нашелся
            if hasattr(detector_instance, 'detector_type_name_for_factory'): # Если мы добавим такой атрибут в детекторы
                 status_entry["detector_type"] = detector_instance.detector_type_name_for_factory
            elif hasattr(detector_instance, '_model_type_for_factory_ref'): # Как в StatisticalDetector
                 status_entry["detector_type"] = detector_instance._model_type_for_factory_ref
            else:
                 status_entry["detector_type"] = type(detector_instance).__name__ # крайний случай

        try:
            if hasattr(detector_instance, 'get_params') and callable(detector_instance.get_params):
                params_to_log = detector_instance.get_params()
                # Конвертируем numpy типы, если они есть, для JSON сериализации
                status_entry["internal_params"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in params_to_log.items()} 
            elif hasattr(detector_instance, 'params') and isinstance(detector_instance.params, dict):
                status_entry["internal_params"] = detector_instance.params
        except Exception as e_int_params:
            logger.debug(f"Не удалось получить внутренние параметры для {detector_name}: {e_int_params}")
        
        return status_entry

    def get_detector_status(self) -> Dict[str, Any]:
        """Возвращает статус всех детекторов в многоуровневой системе."""
        if self.multilevel_detector is None:
            logger.warning("Попытка получить статус, когда multilevel_detector не инициализирован.")
            return {"error": "Multilevel detector is not initialized."}

        status_report: Dict[str, Any] = {
            "transaction_level": {},
            "behavior_level": {},
            "time_series_level": {}
        }
        
        # Получаем конфигурации из объекта MultilevelAnomalyDetector
        # multilevel_config должен быть объектом MultilevelConfigPydantic из config.py
        multilevel_config_pydantic = settings.ml_service.multilevel_detector_default_config
        
        if multilevel_config_pydantic is None:
            logger.error("multilevel_detector_default_config не загружена из settings.yaml")
            # Возвращаем статус с ошибкой для каждого уровня, если конфигурация отсутствует
            for level_name_key in status_report.keys():
                status_report[level_name_key] = {
                    "_level_info_": { # Используем специальный ключ для информации об уровне
                        "is_trained": False,
                        "detector_type": "level_summary",
                        "error_message": "Конфигурация multilevel_detector_default_config не найдена в settings.yaml.",
                        "model_filename": None,
                        "expected_path": None,
                        "exists": None,
                        "can_load": None,
                        "params_from_config": None,
                        "internal_params": None
                    }
                }
            return status_report

        # Конвертируем в словарь для удобства
        # Важно: model_dump может не сохранять структуры default_factory, если они пустые по умолчанию.
        # Но DetectorFactory._generate_detector_name ожидает словарь.
        effective_config = multilevel_config_pydantic.model_dump(exclude_none=False) # exclude_none=False чтобы сохранить пустые списки
        
        # Уровни и их детекторы/конфигурации
        levels_map = {
            "transaction_level": (self.multilevel_detector.transaction_detectors, effective_config.get("transaction_level", [])),
            "behavior_level": (self.multilevel_detector.behavior_detectors, effective_config.get("behavior_level", [])),
            "time_series_level": (self.multilevel_detector.time_series_detectors, effective_config.get("time_series_level", []))
        }

        for level_name_key, (detectors_on_level, level_configs_list) in levels_map.items():
            level_status_dict: Dict[str, Dict[str, Any]] = {} # Для Pydantic это будет Dict[str, DetectorStatus]
            
            config_exists_for_level = bool(level_configs_list)
            
            if not detectors_on_level:
                error_msg = f"Нет активных детекторов для уровня '{level_name_key}'."
                if config_exists_for_level:
                    error_msg += " В config.yaml есть конфигурации для этого уровня, но ни один детектор не был успешно инициализирован."
                else:
                    error_msg += " В config.yaml нет конфигураций для этого уровня."
                
                level_status_dict["_level_info_"] = {
                    "is_trained": False,
                    "detector_type": "level_summary",
                    "error_message": error_msg,
                    "model_filename": None, "expected_path": None, "exists": None, 
                    "can_load": None, "params_from_config": None, "internal_params": None
                }
            else:
                for name, detector_instance in detectors_on_level.items():
                    level_status_dict[name] = self._get_single_detector_status_entry(
                        detector_name=name,
                        detector_instance=detector_instance,
                        level_name=level_name_key,
                        level_configs_list=level_configs_list # передаем список конфигов для текущего уровня
                    )
            status_report[level_name_key] = level_status_dict
            
        return status_report

    async def detect_async_task_wrapper(self,
                                task_id: str,
                                data: Optional[pd.DataFrame] = None,
                                load_data_params: Optional[Dict[str, Any]] = None,
                                transaction_threshold: float = 0.6,
                                behavior_threshold: float = 0.6,
                                time_series_threshold: float = 0.6,
                                final_threshold: float = 0.5):
        """Обертка для метода detect_async для выполнения в фоновом режиме с обновлением статуса задачи."""
        logger.info(f"[TaskID: {task_id}] Обертка detect_async_task_wrapper запущена.")
        task_manager.update_task_status(task_id, status="processing", details="Обнаружение аномалий и сохранение результатов...")
        
        try:
            results_df, save_stats = await self.detect_async(
                data=data,
                load_data_params=load_data_params,
                transaction_threshold=transaction_threshold,
                behavior_threshold=behavior_threshold,
                time_series_threshold=time_series_threshold,
                final_threshold=final_threshold
            )
            
            # Подготовка результата для TaskManager
            task_result = {
                "detection_summary": f"Обработано {len(results_df) if results_df is not None else 0} записей.",
                "saved_anomalies_count": save_stats.get("newly_saved_anomalies_count", 0),
                "skipped_duplicates_count": save_stats.get("skipped_duplicates_count", 0),
                "errors_on_save": save_stats.get("errors_on_save", 0)
            }
            if results_df is None or results_df.empty:
                task_manager.update_task_status(task_id, status="completed_no_data", 
                                                details="Детекция завершена, но нет данных для обработки или не найдено аномалий.",
                                                result=task_result)
                logger.info(f"[TaskID: {task_id}] Детекция завершена, нет данных/аномалий.")                                
            elif save_stats.get("newly_saved_anomalies_count", 0) == 0 and save_stats.get("errors_on_save", 0) == 0:
                task_manager.update_task_status(task_id, status="completed_no_anomalies_found", 
                                                details="Детекция завершена, новые аномалии не обнаружены/не сохранены.",
                                                result=task_result)
                logger.info(f"[TaskID: {task_id}] Детекция завершена, новые аномалии не найдены.")
            else:
                task_manager.update_task_status(task_id, status="completed", 
                                                details="Детекция и сохранение аномалий успешно завершены.",
                                                result=task_result)
                logger.info(f"[TaskID: {task_id}] Детекция и сохранение аномалий успешно завершены.")

        except Exception as e:
            logger.error(f"[TaskID: {task_id}] Критическая ошибка в detect_async_task_wrapper: {e}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", 
                                            details=f"Критическая ошибка сервера при детекции: {str(e)}", 
                                            error_type=type(e).__name__)

# TODO: Добавить методы для управления жизненным циклом отдельных детекторов внутри multilevel, если это необходимо.
# Например, train_single_detector_in_multilevel, get_single_detector_status_in_multilevel.
# Но это усложнит API, возможно, текущего уровня достаточно.