# backend/api/anomalies.py

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any, Tuple
import json
import pandas as pd
from datetime import datetime
import os
import numpy as np
import logging
import re

from backend.config.config import get_settings
settings = get_settings()

from .. import crud, models
from ..ml_service import schemas as ml_schemas
from ..database import get_db
from ..ml_service import common
from ..ml_service.protocols import AnomalyDetectorProtocol
from ..ml_service.graph_detector import GraphAnomalyDetector
from ..ml_service.detector import StatisticalDetector

# Импорт для LLM
from backend.ml_service.llm_explainer import LLMExplainer
from pydantic import BaseModel, Field, field_validator

# --- Импорты для TaskManager ---
from backend.services.task_manager import task_manager, TaskCreationResponse, TaskStatus
# --------------------------------

router = APIRouter()

llm_explainer_default_model = settings.common.llm_model_name 
llm_explainer_ollama_url = str(settings.common.ollama_base_url) # Преобразуем HttpUrl в строку

llm_explainer = LLMExplainer(ollama_base_url=llm_explainer_ollama_url, model_name=llm_explainer_default_model)

logger = logging.getLogger("backend.api.anomalies")

def _initialize_detector(
    detector_type: str,
    model_config: Dict[str, Any],
    model_base_path: str,
    load_trained_model_if_path_exists: bool = False
) -> Tuple[AnomalyDetectorProtocol, str, List[str]]:
    """
    Централизованно инициализирует детектор, определяет путь к модели и необходимые базовые признаки.

    Возвращает:
        Tuple[AnomalyDetectorProtocol, str, List[str]]: (экземпляр детектора, путь к файлу модели, список базовых признаков)
    """
    task_name_prefix = f"_initialize_detector ({detector_type}, {model_config.get('model_filename', 'N/A')})"
    logger.info(f"[{task_name_prefix}] Начало инициализации детектора.")

    model_filename = model_config.get("model_filename")
    if not model_filename:
        logger.error(f"[{task_name_prefix}] ОШИБКА: model_filename отсутствует в model_config.")
        raise ValueError("model_filename отсутствует в конфигурации модели")

    actual_model_path = os.path.join(model_base_path, model_filename)
    logger.info(f"[{task_name_prefix}] Путь к файлу модели: {actual_model_path}")

    from backend.ml_service.detector_factory import DetectorFactory
    detector_class = DetectorFactory.get_detector_class(detector_type)
    if not detector_class:
        logger.error(f"[{task_name_prefix}] ОШИБКА: Неизвестный тип детектора через DetectorFactory: {detector_type}.")
        raise ValueError(f"Неизвестный тип детектора: {detector_type}")
    logger.info(f"[{task_name_prefix}] Класс детектора получен из DetectorFactory: {detector_class.__name__}")

    init_params = {k: v for k, v in model_config.items() if k in detector_class.__init__.__code__.co_varnames}
    init_params.pop('type', None)
    init_params.pop('model_filename', None)
    init_params.pop('weight', None)
    
    base_features_needed: List[str] = []

    if 'features' in init_params and init_params['features']:
        base_features_needed = init_params['features']
    elif detector_type == 'statistical':
        if 'feature' not in init_params: init_params['feature'] = 'price'
        base_features_needed = [init_params['feature']]
    elif detector_type in ['isolation_forest', 'autoencoder', 'vae'] and not base_features_needed:
        if 'features' not in init_params:
            if detector_type == 'isolation_forest':
                 init_params['features'] = ['price', 'price_deviation_from_category_mean']
            else: # autoencoder, vae
                 init_params['features'] = ['price', 'freight_value']
        base_features_needed = init_params['features']

    logger.info(f"[{task_name_prefix}] Итоговые параметры для инициализации детектора: {init_params}")
    logger.info(f"[{task_name_prefix}] Определены базовые признаки (если применимо): {base_features_needed}")

    try:
        detector_instance: AnomalyDetectorProtocol = detector_class(**init_params)
        logger.info(f"[{task_name_prefix}] Экземпляр детектора {detector_instance.model_name} создан.")
    except Exception as e_init:
        logger.error(f"[{task_name_prefix}] ОШИБКА при инициализации детектора: {e_init}", exc_info=True)
        raise

    if load_trained_model_if_path_exists:
        if os.path.exists(actual_model_path):
            logger.info(f"[{task_name_prefix}] Загрузка обученной модели из {actual_model_path}...")
            try:
                detector_instance.load_model(actual_model_path)
                logger.info(f"[{task_name_prefix}] Модель {detector_instance.model_name} загружена. is_trained: {detector_instance.is_trained}")
            except Exception as e_load:
                logger.error(f"[{task_name_prefix}] ОШИБКА при загрузке модели {detector_instance.model_name} из {actual_model_path}: {e_load}", exc_info=True)
        else:
            logger.warning(f"[{task_name_prefix}] Файл модели {actual_model_path} не найден. Модель не будет загружена.")

    return detector_instance, actual_model_path, base_features_needed

def train_model_task(
    task_id: str,
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None, 
    detector_type: str = "isolation_forest", 
    model_config: Optional[Dict[str, Any]] = None
):
    if model_config is None:
        model_config = {}
        
    task_name = f"train_model_task ({detector_type}, {model_config.get('model_filename', 'N/A')}, task_id={task_id})"
    logger.info(f"НАЧАЛО ЗАДАЧИ: {task_name}")
    task_manager.update_task_status(task_id, status="processing", details=f"Инициализация и подготовка данных для обучения детектора {detector_type}...")
    
    db: Optional[Session] = None
    try:
        db = next(get_db())
        logger.info(f"[{task_name}] Сессия БД создана для задачи.")

        logger.info(f"[{task_name}] Параметры: start_date={start_date}, end_date={end_date}, detector_type={detector_type}")

        try:
            detector, model_path, base_features_needed = _initialize_detector(
                detector_type=detector_type,
                model_config=model_config,
                model_base_path=settings.common.model_base_path,
                load_trained_model_if_path_exists=False
            )
        except ValueError as e_val:
            logger.error(f"[{task_name}] ОШИБКА конфигурации при инициализации детектора: {e_val}")
            task_manager.update_task_status(task_id, status="failed", details=f"Ошибка конфигурации: {e_val}", error_type="ConfigurationError")
            return
        logger.info(f"[{task_name}] Путь для сохранения модели (из _initialize_detector): {model_path}")

        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info(f"[{task_name}] Директория для модели создана/существует: {os.path.dirname(model_path)}")
        except Exception as e_mkdir:
            logger.error(f"[{task_name}] ОШИBKA при создании директории для модели: {e_mkdir}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", details=f"Ошибка файловой системы: {e_mkdir}", error_type="FileSystemError")
            return

        logger.info(f"[{task_name}] Загрузка данных из БД...")
        task_manager.update_task_status(task_id, status="processing", details="Загрузка данных из БД...")
        items_df = common.load_data_from_db(db, start_date=start_date, end_date=end_date,
                                          load_associations=True)
        
        if items_df is None:
            logger.error(f"[{task_name}] ОШИБКА: common.load_data_from_db вернул None. Данные не загружены.")
            task_manager.update_task_status(task_id, status="failed", details="Ошибка загрузки данных из БД.", error_type="DataLoadError")
            return
        if items_df.empty:
            logger.warning(f"[{task_name}] Нет данных для обучения после загрузки из БД.")
            task_manager.update_task_status(task_id, status="completed_no_data", details="Нет данных для обучения.")
            return
        logger.info(f"[{task_name}] Загружено {len(items_df)} записей из БД.")
        task_manager.update_task_status(task_id, status="processing", details=f"Загружено {len(items_df)} записей. Инженерия признаков...")

        detector_uses_basic_features = False
        actual_features_for_check = base_features_needed
        if not actual_features_for_check and hasattr(detector, 'features') and detector.features:
            actual_features_for_check = detector.features
            logger.info(f"[{task_name}] Используем detector.features для проверки: {actual_features_for_check}")

        if actual_features_for_check and all(f in items_df.columns for f in actual_features_for_check):
            detector_uses_basic_features = True
            logger.info(f"[{task_name}] Используются признаки: {actual_features_for_check}. Feature engineering не применяется.")
        else:
            missing_features_str = str(actual_features_for_check) if actual_features_for_check else "(не определены или не требуются)"
            logger.info(f"[{task_name}] Не все признаки {missing_features_str} доступны или требуются. Будет применено feature engineering (если применимо).")

        if detector_type != 'graph' and not detector_uses_basic_features:
            logger.info(f"[{task_name}] Применение feature engineering...")
            try:
                items_df = common.engineer_features(items_df)
                logger.info(f"[{task_name}] Feature engineering завершен. Колонки: {items_df.columns.tolist()}")
            except Exception as e_fe:
                logger.error(f"[{task_name}] ОШИБКА при выполнении feature engineering: {e_fe}", exc_info=True)
                task_manager.update_task_status(task_id, status="failed", details=f"Ошибка feature engineering: {e_fe}", error_type="FeatureEngineeringError")
                return
        
        # Инициализация экземпляра детектора и его параметры уже выполнены в _initialize_detector
        logger.info(f"[{task_name}] Экземпляр детектора {detector.model_name} уже создан функцией _initialize_detector.")
        task_manager.update_task_status(task_id, status="processing", details=f"Обучение модели {detector.model_name}...")

        logger.info(f"[{task_name}] Начало обучения модели {detector.model_name}...")
        try:
            detector.train(items_df)
            logger.info(f"[{task_name}] Обучение модели {detector.model_name} завершено. is_trained: {detector.is_trained}")
        except Exception as e_train:
            logger.error(f"[{task_name}] ОШИБКА при обучении модели {detector.model_name}: {e_train}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", details=f"Ошибка обучения модели: {e_train}", error_type="TrainingError")
            return

        if detector.is_trained:
            logger.info(f"[{task_name}] Начало сохранения модели {detector.model_name} в {model_path}...")
            task_manager.update_task_status(task_id, status="processing", details=f"Сохранение модели {detector.model_name}...")
            try:
                detector.save_model(model_path)
                logger.info(f"[{task_name}] Модель/параметры {detector.model_name} успешно сохранены в {model_path}.")
                task_manager.update_task_status(task_id, status="completed", details=f"Модель {detector.model_name} успешно обучена и сохранена.", result={"model_path": model_path})
            except Exception as e_save:
                logger.error(f"[{task_name}] ОШИБКА при сохранении модели {detector.model_name}: {e_save}", exc_info=True)
                task_manager.update_task_status(task_id, status="failed", details=f"Ошибка сохранения модели: {e_save}", error_type="ModelSavingError")
                return
        else:
            logger.warning(f"[{task_name}] Модель {detector.model_name} не была помечена как обученная (is_trained=False). Сохранение не будет выполнено.")
            task_manager.update_task_status(task_id, status="failed", details=f"Модель {detector.model_name} не обучена после вызова train().", error_type="TrainingFailedError")

    except Exception as e_task:
        logger.error(f"[{task_name}] НЕПРЕДВИДЕННАЯ ОШИБКА в задаче обучения: {e_task}", exc_info=True)
        task_manager.update_task_status(task_id, status="failed", details=f"Непредвиденная ошибка сервера: {e_task}", error_type=type(e_task).__name__)
    finally:
        if db:
            db.close()
            logger.info(f"[{task_name}] Сессия БД закрыта.")
        logger.info(f"ЗАВЕРШЕНИЕ ЗАДАЧИ: {task_name}")

def detect_anomalies_task(
    task_id: str,
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None, 
    detector_type: str = "isolation_forest", 
    model_config: Optional[Dict[str, Any]] = None
):
    if model_config is None: model_config = {}
    task_name = f"detect_anomalies_task ({detector_type}, {model_config.get('model_filename', 'N/A')}, task_id={task_id})"
    logger.info(f"НАЧАЛО ЗАДАЧИ: {task_name}")
    
    actual_model_path_for_logging = "N/A"
    db: Optional[Session] = None
    try:
        db = next(get_db())
        logger.info(f"[{task_name}] Сессия БД создана для задачи.")
        
        task_manager.update_task_status(task_id, status="running", details="Инициализация детектора и загрузка модели...")

        try:
            detector, model_path, base_features_needed = _initialize_detector(
                detector_type=detector_type,
                model_config=model_config,
                model_base_path=settings.common.model_base_path,
                load_trained_model_if_path_exists=True # Важно: загружаем модель для детекции
            )
            actual_model_path_for_logging = model_path
        except ValueError as e_val: # Ошибки конфигурации от _initialize_detector
            logger.error(f"[{task_name}] ОШИБКА конфигурации при инициализации детектора: {e_val}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", details=f"Ошибка конфигурации: {e_val}", error_type="ConfigurationError")
            return 

        if not detector.is_trained:
            logger.warning(f"[{task_name}] Модель {detector.model_name} из {actual_model_path_for_logging} не обучена или не смогла загрузиться. Детекция невозможна.")
            details_msg = f"Модель {detector.model_name} из {actual_model_path_for_logging} не обучена."
            if not os.path.exists(actual_model_path_for_logging):
                 details_msg = f"Файл модели {actual_model_path_for_logging} не найден."
            task_manager.update_task_status(task_id, status="failed", details=details_msg, error_type="ModelNotReadyError")
            return
        
        task_manager.update_task_status(task_id, status="running", details="Загрузка и подготовка данных для детекции...")

        logger.info(f"[{task_name}] Загрузка данных из БД для детекции...")
        items_df = common.load_data_from_db(db, start_date=start_date, end_date=end_date, load_associations=True)
        
        if items_df is None:
            logger.error(f"[{task_name}] ОШИБКА: common.load_data_from_db вернул None. Данные для детекции не загружены.")
            # Обновляем статус задачи, если есть task_id
            task_manager.update_task_status(task_id, status="failed", details="Ошибка загрузки данных из БД.", error_type="DataLoadError")
            return 
        if items_df.empty:
            logger.warning(f"[{task_name}] Нет данных для детекции после загрузки из БД.")
            # Обновляем статус задачи
            task_manager.update_task_status(task_id, status="completed_no_data", details="Нет данных для детекции.")
            return
        
        logger.info(f"[{task_name}] Загружено {len(items_df)} записей для детекции.")
        task_manager.update_task_status(task_id, status="running", details=f"Загружено {len(items_df)} записей. Feature engineering (если нужно)..." )
        
        detector_uses_basic_features = False
        actual_features_for_check_detect = base_features_needed
        if not actual_features_for_check_detect and hasattr(detector, 'features') and detector.features:
            actual_features_for_check_detect = detector.features
            logger.info(f"[{task_name}] Используем detector.features для проверки в detect_anomalies_task: {actual_features_for_check_detect}")

        if actual_features_for_check_detect and all(f in items_df.columns for f in actual_features_for_check_detect):
            detector_uses_basic_features = True
            logger.info(f"[{task_name}] Используются признаки: {actual_features_for_check_detect}. Feature engineering не применяется.")
        else:
            logger.info(f"[{task_name}] Не все признаки {actual_features_for_check_detect if actual_features_for_check_detect else '(не определены)'} доступны. Будет применено feature engineering.")
        
        if detector_type != 'graph' and not detector_uses_basic_features:
            logger.info(f"[{task_name}] Применение feature engineering...")
            try:
                items_df = common.engineer_features(items_df)
                logger.info(f"[{task_name}] Feature engineering для детекции завершен.")
            except Exception as e_fe:
                logger.error(f"[{task_name}] ОШИБКА при выполнении feature engineering для детекции: {e_fe}", exc_info=True)
                task_manager.update_task_status(task_id, status="failed", details=f"Ошибка feature engineering: {e_fe}", error_type="FeatureEngineeringError")
                return

        logger.info(f"[{task_name}] Начало детекции аномалий моделью {detector.model_name}...")
        task_manager.update_task_status(task_id, status="running", details=f"Детекция аномалий моделью {detector.model_name} ({len(items_df)} записей)..." )
        try:
            results_df = detector.detect(items_df)
            logger.info(f"[{task_name}] Детекция аномалий моделью {detector.model_name} завершена.")
        except Exception as e_detect:
            logger.error(f"[{task_name}] ОШИБКА при детекции аномалий моделью {detector.model_name}: {e_detect}", exc_info=True)
            task_manager.update_task_status(task_id, status="failed", details=f"Ошибка во время детекции: {e_detect}", error_type="DetectionError")
            return

        if results_df is None or results_df.empty:
            logger.warning(f"[{task_name}] Результаты детекции пусты.")
            task_manager.update_task_status(task_id, status="completed_no_data", details="Результаты детекции пусты.")
            return
        
        anomalies_df = results_df[results_df['is_anomaly'] == True]
        logger.info(f"[{task_name}] Обнаружено {len(anomalies_df)} аномалий (is_anomaly == True). Сохранение в БД...")
        task_manager.update_task_status(task_id, status="running", 
                                        details=f"Обнаружено {len(anomalies_df)} аномалий. Сохранение в БД..." )

        saved_count = 0
        skipped_duplicates_count = 0 # Для будущего расширения, если будет проверка дубликатов
        errors_on_save = 0
        newly_saved_anomaly_ids = []

        if not anomalies_df.empty:
            logger.info(f"[{task_name}] Начало сохранения {len(anomalies_df)} найденных аномалий...")
            task_manager.update_task_status(task_id, status="processing", details=f"Найдено {len(anomalies_df)} аномалий. Сохранение...")
            
            for index, anomaly_row in anomalies_df.iterrows():
                try:
                    details_for_json = {}
                    shap_values_dict = None

                    # --- Генерация SHAP объяснений ---
                    if hasattr(detector, 'get_shap_explanations') and callable(detector.get_shap_explanations):
                        try:
                            if hasattr(detector, 'features') and detector.features:
                                single_raw_data_df = pd.DataFrame([items_df.loc[anomaly_row.name]])
                                
                                explanations_list = detector.get_shap_explanations(single_raw_data_df)

                                if explanations_list and isinstance(explanations_list, list) and len(explanations_list) > 0:
                                    shap_values_dict = explanations_list[0]
                                    logger.debug(f"[{task_name}] SHAP объяснения сгенерированы для anomaly_row index {index}")
                            else:
                                logger.warning(f"[{task_name}] Детектор {detector.model_name} не имеет атрибута 'features' или он пуст. SHAP объяснения не будут сгенерированы.")
                        except KeyError as e_key:
                            logger.error(f"[{task_name}] Ошибка KeyError при доступе к items_df.loc[{anomaly_row.name}] для SHAP: {e_key}. Возможно, индекс не найден. Индекс: {anomaly_row.name}", exc_info=True)
                        except Exception as e_shap:
                            logger.error(f"[{task_name}] Ошибка при генерации SHAP объяснений для anomaly_row index {index}: {e_shap}", exc_info=True)
                    # --- Конец генерации SHAP объяснений ---

                    excluded_cols_for_details = ['order_id', 'order_item_id', 'is_anomaly', 'anomaly_score', 'anomaly_score_normalized']
                    for col_name, value in anomaly_row.items():
                        if col_name not in excluded_cols_for_details:
                            if isinstance(value, (np.integer, np.int_)): value = int(value)
                            elif isinstance(value, (np.floating, np.float64)): value = float(value) # Fixed: np.float_ to np.float64 for broader compatibility
                            elif isinstance(value, np.bool_): value = bool(value)
                            elif isinstance(value, pd.Timestamp): value = value.isoformat() 
                            if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                                details_for_json[col_name] = value
                    
                    if shap_values_dict:
                        details_for_json['shap_values'] = shap_values_dict
                    
                    details_str = json.dumps(details_for_json) if details_for_json else None

                    # Используем нормализованный скор, если он есть, иначе сырой
                    current_anomaly_score = float(anomaly_row.get('anomaly_score_normalized', anomaly_row['anomaly_score']))

                    anomaly_to_save = RootAnomalyCreateSchema(
                        order_id=str(anomaly_row['order_id']),
                        order_item_id=int(anomaly_row['order_item_id']),
                        detection_date=datetime.utcnow(),
                        anomaly_score=current_anomaly_score, # MODIFIED: Используем нормализованный или сырой скор
                        detector_type=detector_type,
                        details=details_str
                    )
                    
                    created_anomaly = crud.create_anomaly(db, anomaly_to_save)
                    saved_count += 1
                    if created_anomaly.id is not None:
                         newly_saved_anomaly_ids.append(created_anomaly.id)
                    logger.debug(f"[{task_name}] Аномалия сохранена: id={created_anomaly.id}, order_id={anomaly_to_save.order_id}, item_id={anomaly_to_save.order_item_id}")

                except Exception as e_save_item:
                    logger.error(f"[{task_name}] Ошибка при подготовке или сохранении отдельной аномалии ({anomaly_row.get('order_id', 'N/A')}, {anomaly_row.get('order_item_id', 'N/A')}): {e_save_item}", exc_info=True)
            
            final_details = f"Обнаружено {len(anomalies_df)} аномалий. Сохранено {saved_count} новых. Пропущено дубликатов: {skipped_duplicates_count}. Ошибок сохранения: {errors_on_save}."
            final_status = "completed_no_anomalies_found" if saved_count == 0 and errors_on_save == 0 else "completed"
            if errors_on_save > 0:
                final_status = "completed_with_errors"
            
            task_manager.update_task_status(task_id, status=final_status, details=final_details,
                                            result={"detected_count": len(anomalies_df), 
                                                    "saved_count": saved_count, 
                                                    "skipped_duplicates": skipped_duplicates_count,
                                                    "save_errors": errors_on_save,
                                                    "detector_type": detector_type,
                                                    "model_filename": model_config.get('model_filename')
                                                    })
        else: # Если anomalies_df пустой
            logger.info(f"[{task_name}] Новых аномалий для сохранения не найдено.")
            task_manager.update_task_status(task_id, status="completed_no_anomalies_found", details="Детекция завершена. Аномалий не обнаружено.")

    except Exception as e_task:
        logger.error(f"[{task_name}] НЕПРЕДВИДЕННАЯ ОШИБКА в задаче детекции: {e_task}", exc_info=True)
        task_manager.update_task_status(task_id, status="failed", details=f"Непредвиденная ошибка сервера: {e_task}", error_type=type(e_task).__name__)
    finally:
        if db:
            db.close()
            logger.info(f"[{task_name}] Сессия БД закрыта.")
        logger.info(f"ЗАВЕРШЕНИЕ ЗАДАЧИ: {task_name}")

# --- Pydantic модели для эндпоинтов --- 
class TrainModelRequest(BaseModel):
    start_date: Optional[datetime] = Field(default=datetime(2000, 1, 1), description="Начальная дата для фильтрации данных при обучении. По умолчанию 2000-01-01 для захвата всех исторических данных.")
    end_date: Optional[datetime] = None
    detector_type: str = Field(default="isolation_forest", description="Тип детектора для обучения")
    detector_config_payload: Dict[str, Any] = Field(default_factory=dict, description="Конфигурация модели для инициализации и обучения")

class DetectAnomaliesRequest(BaseModel):
    start_date: Optional[datetime] = Field(default=datetime(2000, 1, 1), description="Начальная дата для фильтрации данных. По умолчанию 2000-01-01 для захвата всех исторических данных.")
    end_date: Optional[datetime] = None
    detector_type: str = Field(default="isolation_forest", description="Тип детектора для обнаружения")
    detector_config_payload: Dict[str, Any] = Field(default_factory=dict, description="Конфигурация модели для инициализации и обнаружения")

# Для Root ответа при чтении списка аномалий
class RootAnomalySchema(ml_schemas.AnomalyBase):
    id: int
    details: Optional[Dict[str, Any]] = None

    model_config = {
        "from_attributes": True
    }

    @field_validator('details', mode='before')
    @classmethod
    def parse_details_json(cls, v: Any) -> Optional[Dict[str, Any]]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from 'details' field: {v}")
                return None
        if isinstance(v, dict) or v is None:
            return v
        logger.warning(f"Unexpected type for 'details' field ({type(v)}), expected str, dict, or None.")
        return None

class RootAnomalyCreateSchema(ml_schemas.AnomalyCreate):
    pass

class RootPaginatedResponse(ml_schemas.PaginatedResponse[RootAnomalySchema]):
    pass

# ==============================================================================
# API эндпоинты
# ==============================================================================

@router.post("/train_model", 
            response_model=TaskCreationResponse,
            status_code=status.HTTP_202_ACCEPTED, 
            summary="Train anomaly detection model",
            tags=["Anomaly Model Training & Detection"])
async def train_model_endpoint(
    request_body: TrainModelRequest,
    background_tasks: BackgroundTasks
):
    """
    Запускает обучение модели обнаружения аномалий в фоновом режиме.
    """
    task_id = task_manager.create_task(description=f"Обучение детектора {request_body.detector_type}")
    logger.info(f"Эндпоинт /train_model вызван. Task ID: {task_id} для детектора {request_body.detector_type}")

    await task_manager.run_task_in_background(
        background_tasks,
        task_id,
        train_model_task, # Передаем саму функцию
        start_date=request_body.start_date,
        end_date=request_body.end_date,
        detector_type=request_body.detector_type,
        model_config=request_body.detector_config_payload
    )
    
    initial_status_obj = task_manager.get_task_status(task_id)
    if initial_status_obj is None: # На случай, если что-то пошло не так при создании
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать задачу.")

    return TaskCreationResponse(
        task_id=task_id,
        message=f"Задача обучения для детектора {request_body.detector_type} запущена.",
        status_endpoint=f"/api/tasks/task_status/{task_id}", # Общий эндпоинт
        initial_status=initial_status_obj.status
    )

@router.post("/detect_anomalies", 
            response_model=TaskCreationResponse,
            status_code=status.HTTP_202_ACCEPTED, 
            summary="Detect anomalies with a model",
            tags=["Anomaly Model Training & Detection"])
async def detect_anomalies_endpoint(
    request_body: DetectAnomaliesRequest,
    background_tasks: BackgroundTasks
):
    """
    Запускает обнаружение аномалий с использованием указанной модели в фоновом режиме.
    """
    task_id = task_manager.create_task(description=f"Детекция аномалий детектором {request_body.detector_type}")
    logger.info(f"Эндпоинт /detect_anomalies вызван. Task ID: {task_id} для детектора {request_body.detector_type}")

    await task_manager.run_task_in_background(
        background_tasks,
        task_id,
        detect_anomalies_task, # Передаем саму функцию
        start_date=request_body.start_date,
        end_date=request_body.end_date,
        detector_type=request_body.detector_type,
        model_config=request_body.detector_config_payload
    )
    
    initial_status_obj = task_manager.get_task_status(task_id)
    if initial_status_obj is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать задачу.")

    return TaskCreationResponse(
        task_id=task_id,
        message=f"Задача детекции для детектора {request_body.detector_type} запущена.",
        status_endpoint=f"/api/tasks/task_status/{task_id}", # Общий эндпоинт
        initial_status=initial_status_obj.status
    )

# --- Эндпоинты для CRUD операций с аномалиями ---
@router.get("/", response_model=RootPaginatedResponse, tags=["Anomalies Records CRUD"])
def read_anomalies(
    skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db),
    start_date: Optional[datetime] = Query(None), end_date: Optional[datetime] = Query(None),
    min_score: Optional[float] = Query(None), max_score: Optional[float] = Query(None),
    detector_type: Optional[str] = Query(None)
):
    anomalies_db = crud.get_anomalies(db, skip=skip, limit=limit, start_date=start_date, end_date=end_date, min_score=min_score, max_score=max_score, detector_type=detector_type)
    
    count_stmt = select(func.count()).select_from(models.Anomaly)
    if start_date:
        count_stmt = count_stmt.where(models.Anomaly.detection_date >= start_date)
    if end_date:
        count_stmt = count_stmt.where(models.Anomaly.detection_date < end_date)
    if min_score is not None:
        count_stmt = count_stmt.where(models.Anomaly.anomaly_score >= min_score)
    if max_score is not None:
        count_stmt = count_stmt.where(models.Anomaly.anomaly_score <= max_score)
    if detector_type:
        count_stmt = count_stmt.where(models.Anomaly.detector_type == detector_type)
    
    total_count = db.execute(count_stmt).scalar_one_or_none() or 0 # Используем scalar_one_or_none() и fallback на 0

    items_response = [RootAnomalySchema.model_validate(anomaly) for anomaly in anomalies_db]
    return RootPaginatedResponse(total=total_count, items=items_response)

@router.get("/{anomaly_id}", response_model=RootAnomalySchema, tags=["Anomalies Records CRUD"])
def read_anomaly(anomaly_id: int, db: Session = Depends(get_db)):
    db_anomaly = crud.get_anomaly(db, anomaly_id=anomaly_id)
    if db_anomaly is None:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return RootAnomalySchema.model_validate(db_anomaly)

@router.delete("/{anomaly_id}", response_model=RootAnomalySchema, summary="Delete Anomaly by ID", tags=["Anomalies Records CRUD"])
def delete_anomaly_endpoint(anomaly_id: int, db: Session = Depends(get_db)):
    db_anomaly = crud.delete_anomaly(db, anomaly_id=anomaly_id)
    if db_anomaly is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    return RootAnomalySchema.model_validate(db_anomaly)

@router.delete("/", status_code=status.HTTP_200_OK, summary="Delete All Anomalies", tags=["Anomalies Records CRUD"])
def delete_all_anomalies_endpoint(db: Session = Depends(get_db)):
    try:
        num_deleted = crud.delete_all_anomalies(db)
        logging.info(f"Удалено {num_deleted} записей об аномалиях.")
        return {"message": f"Все аномалии ({num_deleted}) успешно удалены."}
    except Exception as e:
        logging.error(f"Ошибка при удалении всех аномалий: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Произошла ошибка при удалении аномалий.")

class LLMExplanationResponse(BaseModel):
    anomaly_id: int
    original_details: RootAnomalySchema
    llm_explanation: str

# --- Вспомогательная функция для извлечения деталей для LLM ---
def _extract_llm_relevant_details(db_anomaly: models.Anomaly, anomaly_id: int) -> tuple:
    details_dict_for_llm = {}
    shap_values_for_llm = None
    detector_specific_info_for_llm = None
    technical_detector_name = "Unknown"
    anomaly_score_for_llm = db_anomaly.anomaly_score  # По умолчанию из основного поля
    contributing_detectors_explanations_for_llm = None  # Новое поле для объяснений от детекторов

    try:
        if db_anomaly.details:
            parsed_details = json.loads(db_anomaly.details)
            details_dict_for_llm = parsed_details  # Используем распарсенный dict для LLM
            
            shap_values_for_llm = details_dict_for_llm.get("shap_values")
            detector_specific_info_for_llm = details_dict_for_llm.get("detector_specific_info")
            technical_detector_name = details_dict_for_llm.get("detector_type", "Unknown")
            
            # Извлекаем подробные объяснения от детекторов, добавленные многоуровневым детектором
            contributing_detectors_explanations_for_llm = details_dict_for_llm.get("contributing_detectors_explanations")
            
            if 'anomaly_score' in details_dict_for_llm:  # Приоритет из details
                anomaly_score_for_llm = details_dict_for_llm['anomaly_score']
            elif db_anomaly.anomaly_score is None:  # Если в details нет, и в основном поле None
                anomaly_score_for_llm = 0.0  # Устанавливаем дефолт
            # Иначе остается db_anomaly.anomaly_score

            logger.info(f"Подготовлены данные для LLM: SHAP keys: {list(shap_values_for_llm.keys()) if shap_values_for_llm else 'None'}, " 
                        f"Detector: {technical_detector_name}, "
                        f"Contributing detectors: {bool(contributing_detectors_explanations_for_llm)}, "
                        f"Anomaly ID: {anomaly_id}")

    except json.JSONDecodeError:
        logger.warning(f"Не удалось распарсить JSON из details для anomaly_id {anomaly_id}. LLM объяснение будет ограничено.")
        if anomaly_score_for_llm is None: anomaly_score_for_llm = 0.0

    except Exception as e_parse: 
        logger.warning(f"Ошибка при обработке details для LLM для anomaly_id {anomaly_id}: {e_parse}. LLM объяснение будет ограничено.")
        if anomaly_score_for_llm is None: anomaly_score_for_llm = 0.0
    
    return (details_dict_for_llm, shap_values_for_llm, detector_specific_info_for_llm, 
            technical_detector_name, anomaly_score_for_llm, contributing_detectors_explanations_for_llm)

@router.get("/{anomaly_id}/explain-llm", 
            response_model=LLMExplanationResponse, 
            summary="Get LLM-generated explanation for an anomaly",
            tags=["Anomaly Explanations (LLM)"])
async def get_anomaly_llm_explanation(anomaly_id: int, db: Session = Depends(get_db)):
    db_anomaly = crud.get_anomaly(db, anomaly_id)
    if not db_anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    (details_dict_for_llm, 
     shap_values_for_llm, 
     detector_specific_info_for_llm, 
     technical_detector_name, 
     anomaly_score_for_llm,
     contributing_detectors_explanations_for_llm) = _extract_llm_relevant_details(db_anomaly, anomaly_id)

    try:
        current_score_float = float(anomaly_score_for_llm)
    except (ValueError, TypeError):
        logger.warning(f"Не удалось преобразовать anomaly_score '{anomaly_score_for_llm}' в float для LLM. Используется 0.0.")
        current_score_float = 0.0

    try:
        validated_original_anomaly = RootAnomalySchema.model_validate(db_anomaly)
    except Exception as e_val_orig:
        logger.error(f"Критическая ошибка: не удалось валидировать db_anomaly ({anomaly_id}) в RootAnomalySchema: {e_val_orig}. Формируем fallback.")
        validated_original_anomaly = RootAnomalySchema(
            id=db_anomaly.id,
            order_id=db_anomaly.order_id,
            timestamp=db_anomaly.detection_date, 
            anomaly_score=db_anomaly.anomaly_score if db_anomaly.anomaly_score is not None else 0.0,
            detector_type=db_anomaly.detector_type if db_anomaly.detector_type else "Unknown",
            details=db_anomaly.details if isinstance(db_anomaly.details, (str, type(None))) else None 
        )

    explanation_text = llm_explainer.generate_explanation(
        anomaly_data=details_dict_for_llm, 
        shap_values=shap_values_for_llm, 
        detector_specific_info=detector_specific_info_for_llm, 
        anomaly_score=current_score_float
    )

    return LLMExplanationResponse(
        anomaly_id=anomaly_id,
        llm_explanation=explanation_text,
        original_details=validated_original_anomaly 
    )
