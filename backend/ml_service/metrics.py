"""
metrics.py - Метрики производительности моделей
===============================================
Модуль содержит функции для оценки качества моделей обнаружения аномалий.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, 
    confusion_matrix, roc_curve
)
from typing import Dict, Tuple, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def compute_binary_metrics(
    true_labels: np.ndarray, 
    predictions: np.ndarray, 
    scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Вычисляет метрики для бинарной классификации аномалий.
    
    Args:
        true_labels: Истинные метки (1 для аномалий, 0 для нормальных)
        predictions: Предсказанные метки (1 для аномалий, 0 для нормальных)
        scores: Сырые скоры аномальности (необязательно)
        
    Returns:
        Словарь с метриками: precision, recall, f1, auc (если переданы scores)
    """
    # Проверяем наличие данных
    if len(true_labels) == 0 or len(predictions) == 0:
        logger.warning("Пустые массивы меток для вычисления метрик")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }

    # Вычисляем основные метрики
    metrics = {
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'accuracy': (true_labels == predictions).mean()
    }
    
    # Добавляем AUC, если предоставлены скоры
    if scores is not None:
        try:
            metrics['auc'] = roc_auc_score(true_labels, scores)
        except Exception as e:
            logger.warning(f"Не удалось вычислить AUC: {e}")
            metrics['auc'] = 0.0
    
    return metrics

def plot_confusion_matrix(
    true_labels: np.ndarray, 
    predictions: np.ndarray,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Создает матрицу ошибок для визуализации результатов.
    
    Args:
        true_labels: Истинные метки
        predictions: Предсказанные метки
        title: Заголовок графика
        
    Returns:
        Matplotlib Figure с визуализацией
    """
    cm = confusion_matrix(true_labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im)
    
    classes = ["Нормальный", "Аномалия"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Добавляем значения в ячейки
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('Истинная метка')
    ax.set_xlabel('Предсказанная метка')
    plt.tight_layout()
    
    return fig

def plot_roc_curve(
    true_labels: np.ndarray, 
    scores: np.ndarray,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Создает ROC-кривую для визуализации качества модели.
    
    Args:
        true_labels: Истинные метки
        scores: Скоры аномальности
        title: Заголовок графика
        
    Returns:
        Matplotlib Figure с ROC-кривой
    """
    fpr, tpr, _ = roc_curve(true_labels, scores)
    auc = roc_auc_score(true_labels, scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    return fig

def evaluate_threshold_performance(
    true_labels: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Оценивает производительность модели при разных порогах.
    
    Args:
        true_labels: Истинные метки
        scores: Скоры аномальности
        thresholds: Список порогов для оценки (если None, генерируются автоматически)
        
    Returns:
        DataFrame с метриками для каждого порога
    """
    if thresholds is None:
        # Если пороги не указаны, генерируем равномерно распределенные значения
        thresholds = np.linspace(0, 1, 20)
    
    results = []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        metrics = compute_binary_metrics(true_labels, predictions)
        
        result_row = {
            'threshold': threshold,
            **metrics
        }
        results.append(result_row)
    
    return pd.DataFrame(results)

def find_optimal_threshold(
    true_labels: np.ndarray,
    scores: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Находит оптимальный порог для заданной метрики.
    
    Args:
        true_labels: Истинные метки
        scores: Скоры аномальности
        metric: Метрика для оптимизации ('f1', 'precision', 'recall')
        
    Returns:
        Кортеж (оптимальный_порог, метрики_при_оптимальном_пороге)
    """
    # Оцениваем на множестве порогов
    results_df = evaluate_threshold_performance(true_labels, scores)
    
    # Находим строку с максимальным значением заданной метрики
    best_idx = results_df[metric].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    # Получаем все метрики при оптимальном пороге
    best_metrics = results_df.loc[best_idx].to_dict()
    
    return best_threshold, best_metrics

def model_drift_monitor(
    reference_scores: np.ndarray,
    current_scores: np.ndarray
) -> Dict[str, float]:
    """
    Отслеживает дрифт модели, сравнивая распределения скоров.
    
    Args:
        reference_scores: Скоры аномальности на референсных данных
        current_scores: Скоры аномальности на текущих данных
        
    Returns:
        Словарь с метриками дрифта
    """
    # Базовая статистика распределений
    ref_mean, ref_std = np.mean(reference_scores), np.std(reference_scores)
    cur_mean, cur_std = np.mean(current_scores), np.std(current_scores)
    
    # Отклонения в процентах
    mean_drift_pct = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-10) * 100
    std_drift_pct = abs(cur_std - ref_std) / max(abs(ref_std), 1e-10) * 100
    
    # KS-тест для сравнения распределений
    from scipy.stats import ks_2samp
    ks_stat, ks_pvalue = ks_2samp(reference_scores, current_scores)
    
    return {
        'mean_drift_pct': mean_drift_pct,
        'std_drift_pct': std_drift_pct,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'is_significant_drift': ks_pvalue < 0.05
    } 