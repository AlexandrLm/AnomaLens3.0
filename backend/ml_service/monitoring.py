"""
monitoring.py - Мониторинг моделей и данных
================================================
Модуль содержит функции для мониторинга производительности моделей,
отслеживания дрифта данных и модели, а также визуализации метрик.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from datetime import datetime
import json
import os
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from .metrics import compute_binary_metrics, plot_confusion_matrix, plot_roc_curve, model_drift_monitor

logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Класс для мониторинга производительности моделей и отслеживания дрифта данных.
    
    Собирает и хранит метрики производительности моделей, а также
    отслеживает изменения в распределении данных и скоров модели.
    """
    def __init__(self, model_name: str, metrics_dir: str = "monitoring/metrics"):
        """
        Инициализирует монитор для конкретной модели.
        
        Args:
            model_name: Имя модели для мониторинга
            metrics_dir: Директория для сохранения метрик
        """
        self.model_name = model_name
        self.metrics_dir = metrics_dir
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_scores: Optional[np.ndarray] = None
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Создаем директорию для метрик, если она не существует
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Путь к файлу с историей метрик
        self.metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.json")
        
        # Загружаем историю метрик, если она существует
        self._load_metrics_history()
        
    def set_reference_data(self, data: pd.DataFrame, scores: np.ndarray):
        """
        Устанавливает эталонные данные и скоры для мониторинга дрифта.
        
        Args:
            data: DataFrame с эталонными данными
            scores: Массив скоров аномальности на эталонных данных
        """
        self.reference_data = data.copy()
        self.reference_scores = scores.copy()
        logger.info(f"Установлены эталонные данные для модели {self.model_name}: {len(data)} записей")
        
    def track_metrics(self, 
                      predictions: np.ndarray, 
                      true_labels: np.ndarray, 
                      scores: np.ndarray,
                      additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Отслеживает метрики производительности модели.
        
        Args:
            predictions: Предсказанные метки (бинарные)
            true_labels: Истинные метки
            scores: Скоры аномальности
            additional_info: Дополнительная информация для сохранения
            
        Returns:
            Словарь с вычисленными метриками
        """
        timestamp = datetime.now().isoformat()
        
        # Вычисляем метрики
        metrics = compute_binary_metrics(true_labels, predictions, scores)
        
        # Сохраняем метрики в историю
        metrics_entry = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        # Добавляем дополнительную информацию, если она предоставлена
        if additional_info:
            metrics_entry["additional_info"] = additional_info
            
        self.metrics_history.append(metrics_entry)
        
        # Сохраняем обновленную историю
        self._save_metrics_history()
        
        return metrics
        
    def track_data_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Отслеживает дрифт данных, сравнивая текущие данные с эталонными.
        
        Args:
            current_data: DataFrame с текущими данными
            
        Returns:
            Словарь с метриками дрифта для каждой числовой колонки
        """
        if self.reference_data is None:
            logger.warning(f"Эталонные данные не установлены для модели {self.model_name}")
            return {}
            
        drift_metrics = {}
        
        # Проверяем дрифт только для числовых колонок
        numeric_columns = self.reference_data.select_dtypes(include=['number']).columns
        
        for col in numeric_columns:
            if col not in current_data.columns:
                logger.warning(f"Колонка {col} отсутствует в текущих данных")
                continue
                
            # Получаем распределения
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                logger.warning(f"Недостаточно данных для анализа дрифта колонки {col}")
                drift_metrics[col] = {"drift_detected": False, "reason": "insufficient_data"}
                continue
                
            try:
                # Вычисляем средние и стандартные отклонения
                ref_mean, ref_std = np.mean(ref_values), np.std(ref_values)
                cur_mean, cur_std = np.mean(cur_values), np.std(cur_values)
                
                # Рассчитываем процентные изменения
                mean_change_pct = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-10) * 100
                std_change_pct = abs(cur_std - ref_std) / max(abs(ref_std), 1e-10) * 100
                
                # Выполняем KS-тест для сравнения распределений
                ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
                
                # Определяем наличие дрифта (p-value < 0.05 означает статистически значимые различия)
                drift_detected = ks_pvalue < 0.05 or mean_change_pct > 20 or std_change_pct > 30
                
                col_metrics = {
                    "ref_mean": float(ref_mean),
                    "cur_mean": float(cur_mean),
                    "mean_change_pct": float(mean_change_pct),
                    "ref_std": float(ref_std),
                    "cur_std": float(cur_std),
                    "std_change_pct": float(std_change_pct),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "drift_detected": drift_detected
                }
                
                drift_metrics[col] = col_metrics
                
                if drift_detected:
                    logger.warning(f"Обнаружен дрифт данных для колонки {col} модели {self.model_name}: mean_change={mean_change_pct:.2f}%, std_change={std_change_pct:.2f}%, ks_pvalue={ks_pvalue:.6f}")
                
            except Exception as e:
                logger.error(f"Ошибка при анализе дрифта колонки {col}: {e}")
                drift_metrics[col] = {"error": str(e)}
                
        # Общая метрика дрифта данных
        num_drifted_columns = sum(col_data.get("drift_detected", False) for col_data in drift_metrics.values())
        total_drift_ratio = num_drifted_columns / max(len(drift_metrics), 1)
        
        drift_metrics["__summary__"] = {
            "timestamp": datetime.now().isoformat(),
            "drifted_columns_count": num_drifted_columns,
            "total_columns": len(drift_metrics),
            "drift_ratio": total_drift_ratio,
            "significant_drift": total_drift_ratio > 0.3  # Если более 30% колонок имеют дрифт
        }
        
        # Сохраняем метрики дрифта в историю
        drift_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "data_drift",
            "drift_metrics": drift_metrics
        }
        
        self.metrics_history.append(drift_entry)
        self._save_metrics_history()
        
        return drift_metrics
        
    def track_model_drift(self, current_scores: np.ndarray) -> Dict[str, float]:
        """
        Отслеживает дрифт модели, сравнивая распределения текущих скоров с эталонными.
        
        Args:
            current_scores: Массив текущих скоров аномальности
            
        Returns:
            Словарь с метриками дрифта модели
        """
        if self.reference_scores is None:
            logger.warning(f"Эталонные скоры не установлены для модели {self.model_name}")
            return {}
            
        try:
            # Используем функцию model_drift_monitor из metrics.py
            drift_metrics = model_drift_monitor(self.reference_scores, current_scores)
            
            if drift_metrics.get("is_significant_drift"):
                logger.warning(f"Обнаружен значительный дрифт модели {self.model_name}: "
                              f"KS={drift_metrics['ks_statistic']:.4f}, p-value={drift_metrics['ks_pvalue']:.6f}")
            
            # Сохраняем метрики дрифта в историю
            drift_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "model_drift",
                "drift_metrics": drift_metrics
            }
            
            self.metrics_history.append(drift_entry)
            self._save_metrics_history()
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Ошибка при анализе дрифта модели {self.model_name}: {e}")
            return {"error": str(e)}
            
    def visualize_metrics_history(self, metric_name: str = "f1") -> plt.Figure:
        """
        Визуализирует историю указанной метрики.
        
        Args:
            metric_name: Имя метрики для визуализации ('precision', 'recall', 'f1', 'auc')
            
        Returns:
            Matplotlib Figure с графиком истории метрики
        """
        # Извлекаем значения метрик и временные метки
        timestamps = []
        metric_values = []
        
        for entry in self.metrics_history:
            if "metrics" in entry and metric_name in entry["metrics"]:
                timestamps.append(datetime.fromisoformat(entry["timestamp"]))
                metric_values.append(entry["metrics"][metric_name])
        
        if not timestamps:
            logger.warning(f"Нет данных для визуализации метрики {metric_name}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Нет данных для метрики {metric_name}", 
                    horizontalalignment='center', verticalalignment='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, metric_values, 'o-')
        ax.set_xlabel('Время')
        ax.set_ylabel(f'Метрика {metric_name}')
        ax.set_title(f'История метрики {metric_name} для модели {self.model_name}')
        ax.grid(True)
        
        # Определяем тренд (улучшение/ухудшение)
        if len(metric_values) > 1:
            # Простая линейная регрессия для определения тренда
            slope, _, _, _, _ = stats.linregress(range(len(metric_values)), metric_values)
            trend_text = "Улучшение" if slope > 0 else "Ухудшение" if slope < 0 else "Стабильно"
            ax.text(0.02, 0.02, f"Тренд: {trend_text}", transform=ax.transAxes)
            
        plt.tight_layout()
        return fig
    
    def _load_metrics_history(self):
        """Загружает историю метрик из файла."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
                logger.info(f"Загружена история метрик для модели {self.model_name}: {len(self.metrics_history)} записей")
            except Exception as e:
                logger.error(f"Ошибка при загрузке истории метрик для модели {self.model_name}: {e}")
                self.metrics_history = []
        else:
            logger.info(f"Файл истории метрик не найден для модели {self.model_name}. Создана новая история.")
            self.metrics_history = []
    
    def _save_metrics_history(self):
        """Сохраняет историю метрик в файл."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.debug(f"Сохранена история метрик для модели {self.model_name}: {len(self.metrics_history)} записей")
        except Exception as e:
            logger.error(f"Ошибка при сохранении истории метрик для модели {self.model_name}: {e}")

# Функция для создания мониторов для всех детекторов ансамбля
def create_monitors_for_ensemble(ensemble_service, metrics_dir: str = "monitoring/metrics") -> Dict[str, ModelMonitor]:
    """
    Создает мониторы для всех детекторов в ансамбле.
    
    Args:
        ensemble_service: Экземпляр EnsembleDetectorService
        metrics_dir: Директория для сохранения метрик
        
    Returns:
        Словарь {имя_детектора: монитор}
    """
    monitors = {}
    
    for detector_name in ensemble_service.get_loaded_detectors():
        monitors[detector_name] = ModelMonitor(detector_name, metrics_dir)
        logger.info(f"Создан монитор для детектора {detector_name}")
    
    # Также создаем монитор для ансамбля в целом
    monitors["ensemble"] = ModelMonitor("ensemble", metrics_dir)
    logger.info("Создан монитор для ансамбля в целом")
    
    return monitors 