# backend/api/multilevel.py

import os
import time
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from fastapi import APIRouter, HTTPException, Query, Body, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime

from ..ml_service.multilevel_service import MultilevelDetectorService
from ..services.task_manager import task_manager, TaskCreationResponse, TaskStatus

logger = logging.getLogger("backend.api.multilevel")

try:
    multilevel_service = MultilevelDetectorService()
except Exception as e:
    logger.critical(f"Не удалось инициализировать MultilevelDetectorService при старте модуля api.multilevel: {e}", exc_info=True)
    multilevel_service = None

router = APIRouter()

# --- Pydantic модели ---
class DetectionParams(BaseModel):
    transaction_threshold: float = Field(0.6)
    behavior_threshold: float = Field(0.6)
    time_series_threshold: float = Field(0.6)
    final_threshold: float = Field(0.5)
    filter_period_days: Optional[int] = Field(10000, description="Количество дней назад для фильтрации данных. По умолчанию 10000 дней (~ 27 лет) для захвата всех исторических данных.")

class DetectorConfigEntry(BaseModel):
    type: str
    model_filename: Optional[str] = None
    weight: Optional[float] = 1.0
    class Config:
        extra = 'allow'

class MultilevelConfig(BaseModel):
    transaction_level: List[DetectorConfigEntry] = Field(default_factory=list)
    behavior_level: List[DetectorConfigEntry] = Field(default_factory=list)
    time_series_level: List[DetectorConfigEntry] = Field(default_factory=list)
    combination_weights: Dict[str, float] = Field(
        default_factory=lambda: {"transaction": 0.4, "behavior": 0.4, "time_series": 0.2}
    )

class SaveStatistics(BaseModel):
    total_detected_anomalies_before_save: int
    newly_saved_anomalies_count: int
    newly_saved_anomaly_ids: List[int]
    skipped_duplicates_count: int
    errors_on_save: int

class DetectResponse(BaseModel):
    status: str
    message: str
    elapsed_time_seconds: float
    save_statistics: SaveStatistics

class DetectorStatus(BaseModel):
    is_trained: bool
    detector_type: str
    model_filename: Optional[str] = None
    expected_path: Optional[str] = None
    exists: Optional[bool] = None
    can_load: Optional[bool] = None
    error_message: Optional[str] = None
    params_from_config: Optional[Dict[str, Any]] = None
    internal_params: Optional[Dict[str, Any]] = None

class MultilevelStatus(BaseModel):
    transaction_level: Dict[str, DetectorStatus] = Field(default_factory=dict)
    behavior_level: Dict[str, DetectorStatus] = Field(default_factory=dict)
    time_series_level: Dict[str, DetectorStatus] = Field(default_factory=dict)

class TrainResponse(BaseModel):
    status: str
    message: str

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    # ... (остальная часть функции convert_numpy_types)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    return obj
    
def _get_checked_service() -> MultilevelDetectorService:
    """Проверяет инициализацию сервиса и возвращает его, или вызывает HTTPException."""
    if multilevel_service is None:
        logger.warning("_get_checked_service вызван, когда multilevel_service is None")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MultilevelDetectorService не инициализирован. Проверьте логи сервера."
        )
    return multilevel_service

@router.get("/status", response_model=MultilevelStatus)
async def get_multilevel_status_endpoint():
    service = _get_checked_service()
    raw_status = service.get_detector_status()
    if "error" in raw_status:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=raw_status["error"])
    return MultilevelStatus(**raw_status)

@router.get("/config", response_model=MultilevelConfig)
async def get_multilevel_config_endpoint():
    service = _get_checked_service()
    config_dict = service.get_config()
    if not config_dict:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Конфигурация не найдена.")
    return MultilevelConfig(**config_dict)

@router.post("/config", response_model=bool)
async def update_multilevel_config_endpoint(config: MultilevelConfig):
    service = _get_checked_service()
    success = service.update_config(config.model_dump(exclude_none=True))
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось обновить конфигурацию.")
    return True

@router.post("/train", response_model=TaskCreationResponse)
async def train_multilevel_system_endpoint(
    background_tasks: BackgroundTasks
):
    service = _get_checked_service()
    task_id = task_manager.create_task(description="Обучение многоуровневой системы")
    
    # Используем пустой словарь для параметров загрузки данных, 
    # что означает загрузку всех данных из БД без фильтрации
    load_data_params = {}
    logger.info("Запуск обучения на всех данных из БД (без фильтрации по дате)")
    
    await task_manager.run_task_in_background(
        background_tasks,
        task_id,
        service.train_task_wrapper, # Предполагаем, что такая обертка будет в сервисе
        load_data_params=load_data_params # Передаем пустые параметры для загрузки всех данных
    )
    
    initial_status_obj = task_manager.get_task_status(task_id)
    if initial_status_obj is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать или получить статус задачи.")
    
    return TaskCreationResponse(
        task_id=task_id,
        message="Задача обучения многоуровневой системы запущена в фоновом режиме.",
        status_endpoint=f"/api/tasks/task_status/{task_id}",
        initial_status=initial_status_obj.status
    )

@router.post("/detect", response_model=TaskCreationResponse)
async def detect_anomalies_multilevel_endpoint(
    params: DetectionParams,
    background_tasks: BackgroundTasks
):
    service = _get_checked_service()
    task_id = task_manager.create_task(description="Детекция аномалий многоуровневой системой")

    load_params: Dict[str, Any] = {}
    # Используем очень давнюю дату как начальную точку, если параметр задан
    # Это позволит захватить все исторические данные
    if params.filter_period_days is not None:
        start_date_filter = datetime.now() - pd.Timedelta(days=params.filter_period_days)
        load_params['start_date'] = start_date_filter
        logger.info(f"Установлена начальная дата фильтра: {start_date_filter} (за последние {params.filter_period_days} дней)")
    else:
        # Если filter_period_days не задан, не устанавливаем start_date вообще
        # что позволит захватить все данные из БД
        logger.info("Фильтр по дате не установлен, будут обработаны все данные из БД")
    
    await task_manager.run_task_in_background(
        background_tasks,
        task_id,
        service.detect_async_task_wrapper, # Предполагаем, что такая обертка будет в сервисе
        load_data_params=load_params,
        transaction_threshold=params.transaction_threshold,
        behavior_threshold=params.behavior_threshold,
        time_series_threshold=params.time_series_threshold,
        final_threshold=params.final_threshold
    )
    
    initial_status_obj = task_manager.get_task_status(task_id)
    if initial_status_obj is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать или получить статус задачи.")
    
    return TaskCreationResponse(
        task_id=task_id,
        message="Задача детекции аномалий многоуровневой системой запущена.",
        status_endpoint=f"/api/tasks/task_status/{task_id}",
        initial_status=initial_status_obj.status
    )

@router.get("/available-detectors", response_model=Dict[str, List[str]])
async def get_available_detectors_endpoint():
    try:
        from ..ml_service.detector_factory import DetectorFactory
        all_detectors = DetectorFactory.get_available_detector_types()
        transaction_types = ['statistical', 'isolation_forest', 'autoencoder', 'vae', 'price_freight_ratio', 'category_price_outlier', 'transaction_isolation_forest', 'transaction_vae']
        behavior_types = ['seller_pricing_behavior', 'seller_category_mix', 'behavior_isolation_forest', 'graph']
        time_series_types = ['seasonal_deviation', 'moving_average_volatility', 'cumulative_sum']
        return {
            "transaction_level": [d for d in all_detectors if d in transaction_types],
            "behavior_level": [d for d in all_detectors if d in behavior_types],
            "time_series_level": [d for d in all_detectors if d in time_series_types]
        }
    except ImportError:
        logger.error("Не удалось импортировать DetectorFactory для /available-detectors")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка конфигурации сервера.")
    except Exception as e:
        logger.error(f"Ошибка в /available-detectors: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Внутренняя ошибка сервера.")