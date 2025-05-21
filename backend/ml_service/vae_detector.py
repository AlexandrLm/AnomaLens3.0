"""
vae_detector.py - Детектор аномалий на основе вариационного автоэнкодера
==========================================================================
Модуль реализует детектор аномалий, использующий вариационный автоэнкодер (VAE)
для обнаружения аномалий в многомерных данных.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
import shap # Добавим импорт SHAP

from .detector import AnomalyDetector  # Импортируем базовый класс

logger = logging.getLogger(__name__) # Убедимся, что логгер модуля определен

# =============================================================================
# Архитектура вариационного автоэнкодера
# =============================================================================

class VAE(nn.Module):
    """
    Улучшенная архитектура вариационного автоэнкодера (VAE).
    
    VAE обучается реконструировать входные данные через сжатое латентное представление,
    при этом накладывая регуляризацию на латентное пространство через KL-дивергенцию
    с нормальным распределением N(0, 1).
    
    Улучшения:
    - Добавлены слои BatchNorm для стабилизации обучения
    - Добавлен Dropout для регуляризации
    - Расширены промежуточные слои
    - Добавлены skip-connections
    """
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64, dropout_rate: float = 0.2):
        """
        Инициализирует архитектуру VAE.
        
        Args:
            input_dim: Размерность входного вектора признаков
            latent_dim: Размерность латентного пространства
            hidden_dim: Размерность скрытых слоев
            dropout_rate: Вероятность dropout (0 = отключен)
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # ----- Энкодер -----
        # Сжимает входные данные до параметров латентного распределения
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # Входной слой -> Скрытый слой 1
            nn.BatchNorm1d(hidden_dim),          # Нормализация батча для стабилизации
            nn.ReLU(),                           # Функция активации ReLU
            nn.Dropout(dropout_rate),            # Dropout для регуляризации
            
            nn.Linear(hidden_dim, hidden_dim),   # Дополнительный скрытый слой той же размерности
            nn.BatchNorm1d(hidden_dim),          # Нормализация батча
            nn.ReLU(),                           # Функция активации ReLU
            nn.Dropout(dropout_rate),            # Dropout для регуляризации
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # Скрытый слой 1 -> Скрытый слой 2 (уменьшенный)
            nn.BatchNorm1d(hidden_dim // 2),     # Нормализация батча
            nn.ReLU(),                           # Функция активации ReLU
        )
        
        # Отдельные слои для параметров латентного распределения
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)         # Слой для среднего значения
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)    # Слой для логарифма дисперсии

        # ----- Декодер -----
        # Восстанавливает входные данные из латентного представления
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),  # Латентное пространство -> Скрытый слой 1
            nn.BatchNorm1d(hidden_dim // 2),     # Нормализация батча
            nn.ReLU(),                          # Функция активации ReLU
            nn.Dropout(dropout_rate),           # Dropout для регуляризации
        )
        
        self.decoder_hidden = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),  # Скрытый слой 1 -> Скрытый слой 2 (увеличенный)
            nn.BatchNorm1d(hidden_dim),          # Нормализация батча
            nn.ReLU(),                          # Функция активации ReLU
            nn.Dropout(dropout_rate),           # Dropout для регуляризации
            
            nn.Linear(hidden_dim, hidden_dim),   # Дополнительный скрытый слой той же размерности
            nn.BatchNorm1d(hidden_dim),          # Нормализация батча
            nn.ReLU(),                          # Функция активации ReLU
        )
        
        # Выходной слой
        self.decoder_output = nn.Linear(hidden_dim, input_dim)  # Скрытый слой -> Выходной слой
        
        # Skip-connection весовые матрицы для "пропуска" части информации напрямую
        self.skip_h1_to_out = nn.Linear(hidden_dim // 2, input_dim)  # Пропуск от первого скрытого слоя декодера

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирует входные данные в параметры латентного распределения.
        
        Args:
            x: Входной тензор данных
            
        Returns:
            Кортеж (mu, log_var): среднее и логарифм дисперсии латентного распределения
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Выполняет трюк репараметризации для дифференцируемого сэмплирования.
        
        Вместо прямого сэмплирования из распределения (что недифференцируемо),
        используется параметризация z = mu + epsilon * sigma, где epsilon ~ N(0, 1).
        
        Args:
            mu: Среднее значение латентного распределения
            log_var: Логарифм дисперсии латентного распределения
            
        Returns:
            Точка z из латентного пространства
        """
        std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)      # eps ~ N(0, 1)
        return mu + eps * std            # z = mu + epsilon * sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Декодирует точку из латентного пространства обратно в пространство признаков.
        
        Args:
            z: Точка из латентного пространства
            
        Returns:
            Реконструированный входной вектор
        """
        h1 = self.decoder_input(z)
        h2 = self.decoder_hidden(h1)
        
        # Основной выход декодера
        main_output = self.decoder_output(h2)
        
        # Skip-connection от первого скрытого слоя декодера
        skip_output = self.skip_h1_to_out(h1)
        
        # Объединяем основной выход и skip-connection
        return main_output + skip_output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход через модель: кодирование -> репараметризация -> декодирование.
        
        Args:
            x: Входной тензор данных
            
        Returns:
            Кортеж (reconstruction, mu, log_var):
            - reconstruction: Реконструированный входной вектор
            - mu: Среднее значение латентного распределения
            - log_var: Логарифм дисперсии латентного распределения
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

# =============================================================================
# Детектор аномалий на основе VAE
# =============================================================================

class VAEDetector(AnomalyDetector):
    """
    Детектор аномалий на основе вариационного автоэнкодера (VAE).
    
    Обнаруживает аномалии путем измерения ошибки реконструкции входных данных.
    Аномальные экземпляры данных обычно имеют более высокую ошибку реконструкции,
    так как VAE обучается восстанавливать нормальные паттерны в данных.
    """
    class _VAEModelWrapperForSHAP(nn.Module):
        def __init__(self, vae_model: VAE):
            super().__init__()
            self.vae_model = vae_model
            self.vae_model.eval() 

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_reconstructed, _, _ = self.vae_model(x)
            reconstruction_error_per_sample = torch.mean((x - x_reconstructed)**2, dim=tuple(range(1, x.ndim))) # Усредняем по всем измерениям кроме батча
            return reconstruction_error_per_sample.unsqueeze(1)

    def __init__(self, 
                 features: List[str], 
                 latent_dim: int = 16, 
                 hidden_dim: int = 64,
                 epochs: int = 10,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 kld_weight: float = 0.5,
                 dropout_rate: float = 0.2,
                 model_name: str = "vae",
                 device: Optional[str] = None,
                 shap_background_samples: int = 100):
        """
        Инициализация базового VAE детектора.

        Args:
            features: Список названий признаков для обучения модели.
            latent_dim: Размерность скрытого (латентного) пространства.
            hidden_dim: Размерность основного скрытого слоя в VAE.
            epochs: Количество эпох обучения.
            batch_size: Размер пакета для обучения.
            learning_rate: Скорость обучения.
            kld_weight: Вес дивергенции Кульбака-Лейблера в функции потерь.
            dropout_rate: Коэффициент Dropout для скрытых слоев VAE.
            model_name: Уникальное имя модели.
            device: Устройство для обучения ('cpu' или 'cuda'). Если None, определяется автоматически.
            shap_background_samples: Количество фоновых выборок для SHAP.
        """
        super().__init__(model_name)
        self.features = features
        if not self.features:
            raise ValueError("Список признаков 'features' не может быть пустым.")
        
        self.input_dim = len(features) # Определяем input_dim здесь
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim 
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kld_weight = kld_weight
        self.dropout_rate = dropout_rate # <--- СОХРАНИТЬ
        self.model: Optional[VAE] = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.shap_background_samples = shap_background_samples
        self.is_fitted = False
        logger.info(f"({self.model_name}) VAEDetector инициализирован. Устройство: {self.device}, Входная размерность: {self.input_dim}, Признаки: {self.features}")
        self._reset_state() # Гарантируем сброс при инициализации

    def _reset_state(self):
        super()._reset_state() # Вызов метода базового класса
        # self.model, self.scaler, self.is_trained, self.min_score_, self.max_score_ сброшены в super()
        self.threshold_ = None
        self.explainer = None 
        self.background_data_for_shap = None 
        # VAEDetector создает StandardScaler в __init__ и preprocess, поэтому super()._reset_state() уже установил self.scaler = None
        # Если VAEDetector должен всегда иметь экземпляр scaler после reset, то:
        # self.scaler = StandardScaler() # Но это может быть избыточным, если __init__ или train его пересоздадут.
        # Пока оставляем как есть, так как scaler управляется в __init__ и preprocess.
        logger.info(f"({self.model_name}) Специфичное состояние VAEDetector (порог, SHAP) сброшено.")

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> Optional[pd.DataFrame]:
        """
        Выбирает признаки, обрабатывает пропуски, масштабирует.
        Если fit_scaler=True, обучает новый StandardScaler.
        Возвращает Optional[pd.DataFrame], None если данных недостаточно или ошибка.
        """
        if not all(f in data.columns for f in self.features):
            missing = [f for f in self.features if f not in data.columns]
            logger.error(f"({self.model_name}) Не найдены признаки в DataFrame: {missing}")
            return None

        processed_data_df = data[self.features].copy()
        
        # Обработка пропусков (заполняем средним)
        for col in self.features:
            if processed_data_df[col].isnull().any():
                mean_val = processed_data_df[col].mean()
                if pd.isna(mean_val): # Если вся колонка NaN
                    mean_val = 0 
                processed_data_df[col] = processed_data_df[col].fillna(mean_val)

        if processed_data_df.empty:
            logger.warning(f"({self.model_name}) Нет данных после выбора признаков.")
            return pd.DataFrame(columns=self.features)
        
        # Масштабирование
        data_scaled_array: Optional[np.ndarray] = None
        if fit_scaler:
            logger.info(f"({self.model_name}) Обучение StandardScaler на {len(processed_data_df)} сэмплах.")
            self.scaler = StandardScaler()
            data_scaled_array = self.scaler.fit_transform(processed_data_df.values)  # Явно передаем numpy array
            logger.info(f"({self.model_name}) StandardScaler обучен.")
        else:
            # Диагностическое логирование для отладки
            logger.debug(f"({self.model_name}) [Detect_Preprocess] Scaler type: {type(self.scaler)}")
            if self.scaler:
                logger.debug(f"({self.model_name}) [Detect_Preprocess] Scaler has mean_: {hasattr(self.scaler, 'mean_')}")
                if hasattr(self.scaler, 'mean_'):
                    logger.debug(f"({self.model_name}) [Detect_Preprocess] Scaler mean_ shape: {self.scaler.mean_.shape}")
                logger.debug(f"({self.model_name}) [Detect_Preprocess] Scaler params: {self.scaler.get_params()}")
            
            if self.scaler is None:
                logger.error(f"({self.model_name}) StandardScaler для {self.model_name} не инициализирован. Для VAE масштабирование критично.")
                return None
            
            # Проверка, обучен ли scaler - теперь проверяем наличие mean_
            if not hasattr(self.scaler, 'mean_'):
                logger.error(f"({self.model_name}) StandardScaler не обучен (отсутствует mean_). Для VAE масштабирование критично.")
                return None
                
            try:
                logger.info(f"({self.model_name}) StandardScaler обучен (mean_ присутствует). Выполняется transform.")
                # Явно передаем numpy array, а не DataFrame, для избежания UserWarning
                data_scaled_array = self.scaler.transform(processed_data_df.values)
            except Exception as e:
                logger.error(f"({self.model_name}) Ошибка при масштабировании данных: {str(e)}. Для VAE масштабирование критично.")
                return None
            
        if data_scaled_array is None:
             logger.error(f"({self.model_name}) Ошибка масштабирования, data_scaled_array is None.")
             return None

        # Проверка на NaN после масштабирования
        if np.isnan(data_scaled_array).any():
             logger.warning(f"({self.model_name}) Обнаружены NaN после масштабирования. Замена на 0.")
             data_scaled_array = np.nan_to_num(data_scaled_array, nan=0.0)

        # Преобразование в DataFrame с сохранением индекса и колонок
        final_processed_df = pd.DataFrame(data_scaled_array, columns=self.features, index=processed_data_df.index)
        return final_processed_df

    def train(self, data: pd.DataFrame):
        """Обучает модель VAE и нормализатор скоров."""
        logger.info(f"({self.model_name}) Начало обучения...")
        start_total_time = time.time()
        
        self._reset_state() # Сбрасываем состояние перед обучением, включая модель и SHAP

        if not self.features:
            logger.error(f"({self.model_name}) Атрибут 'features' не установлен. Обучение невозможно.")
            self.is_trained = False
            return

        logger.info(f"({self.model_name}) Предобработка данных для обучения...")
        processed_data_df = self.preprocess(data.copy(), fit_scaler=True)

        if processed_data_df is None or processed_data_df.empty:
            logger.warning(f"({self.model_name}) Нет данных после предобработки. Обучение не будет выполнено.")
            self.is_trained = False
            return
        
        if self.scaler is None: # Дополнительная проверка после preprocess
            logger.error(f"({self.model_name}) Scaler не был инициализирован после preprocess. Обучение прервано.")
            self.is_trained = False
            return

        # Конвертация данных в тензоры PyTorch
        try:
            tensor_data = torch.tensor(processed_data_df.values, dtype=torch.float32).to(self.device)
            dataset = TensorDataset(tensor_data) # VAE ожидает только X
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        except Exception as e_tensor:
            logger.error(f"({self.model_name}) Ошибка при конвертации данных в тензоры или создании DataLoader: {e_tensor}", exc_info=True)
            self.is_trained = False
            return
            
        logger.info(f"({self.model_name}) Данные ({tensor_data.shape[0]} сэмплов) готовы для обучения VAE.")

        # Инициализация модели VAE
        try:
            self.model = VAE(
                input_dim=self.input_dim, 
                latent_dim=self.latent_dim, 
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            logger.info(f"({self.model_name}) Модель VAE создана и перемещена на {self.device}.")
        except Exception as e_model_init:
            logger.error(f"({self.model_name}) Ошибка при инициализации модели VAE: {e_model_init}", exc_info=True)
            self.model = None
            self.is_trained = False
            return

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Функция потерь VAE
        def loss_function(recon_x, x, mu, log_var, kld_weight_param):
            # Ошибка реконструкции (например, MSE)
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
            
            # KL-дивергенция
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            return recon_loss + kld_weight_param * kld_loss

        # Цикл обучения
        self.model.train() # Переводим модель в режим обучения
        min_loss = float('inf')
        epochs_no_improve = 0
        patience = 5 # Для ранней остановки

        logger.info(f"({self.model_name}) Начало цикла обучения VAE...")
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            total_loss = 0
            for batch_data_list in dataloader:
                batch_data = batch_data_list[0] # DataLoader возвращает список тензоров
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(batch_data)
                loss = loss_function(recon_batch, batch_data, mu, log_var, self.kld_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader.dataset)
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"({self.model_name}) Эпоха {epoch+1}/{self.epochs}, Потери: {avg_loss:.4f}, Время: {epoch_duration:.2f} сек.")

            if avg_loss < min_loss:
                min_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"({self.model_name}) Ранняя остановка на эпохе {epoch+1} из-за отсутствия улучшения потерь.")
                    break
        
        self.model.eval() # Переводим модель в режим оценки после обучения

        # Вычисление порога аномальности на обучающих данных
        logger.info(f"({self.model_name}) Вычисление ошибок реконструкции на обучающих данных для определения порога...")
        all_reconstruction_errors = []
        with torch.no_grad():
            for batch_data_list in dataloader: # Используем тот же dataloader, но без shuffle было бы лучше для консистентности, но для порога это не критично
                batch_data = batch_data_list[0]
                recon_batch, _, _ = self.model(batch_data)
                errors = torch.mean((batch_data - recon_batch)**2, dim=tuple(range(1, batch_data.ndim))) # MSE per sample
                all_reconstruction_errors.extend(errors.cpu().numpy())
        
        if not all_reconstruction_errors:
            logger.error(f"({self.model_name}) Не удалось вычислить ошибки реконструкции на обучающих данных. Порог не будет установлен.")
            self.is_trained = False
            return

        # Установка порога как перцентиля ошибок реконструкции (например, 95-й или 99-й)
        # Это более робастно, чем просто mean + k*std, особенно если есть выбросы в самих ошибках
        self.threshold_ = float(np.percentile(all_reconstruction_errors, 99)) 
        logger.info(f"({self.model_name}) Порог аномальности установлен: {self.threshold_:.6f} (99-й перцентиль ошибок реконструкции).")
        
        # Фиттинг нормализатора для anomaly_score
        # Используем все ошибки реконструкции для обучения нормализатора
        # Это позволит корректно масштабировать скоры и для аномалий, и для нормальных данных
        self.fit_normalizer(np.array(all_reconstruction_errors))
        if self.min_score_ is not None and self.max_score_ is not None:
             logger.info(f"({self.model_name}) Нормализатор anomaly_score обучен: min={self.min_score_:.4f}, max={self.max_score_:.4f}")
        else:
             logger.warning(f"({self.model_name}) Нормализатор anomaly_score не был корректно обучен (min_score_ или max_score_ is None).")


        self.is_trained = True # Модель обучена, порог и нормализатор установлены
        total_duration = time.time() - start_total_time
        logger.info(f"({self.model_name}) Обучение завершено. Общее время: {total_duration:.2f} сек.")

        # --- SHAP Background Data and Explainer Initialization ---
        self.explainer = None # Сброс перед новой инициализацией
        self.background_data_for_shap = None

        if shap and self.is_trained and self.model:
            # 1. Подготовка фоновых данных (tensor_data - это предобработанные обучающие данные на self.device)
            num_samples_shap = min(self.shap_background_samples, tensor_data.shape[0])
            if num_samples_shap > 0:
                indices = np.random.choice(tensor_data.shape[0], num_samples_shap, replace=False)
                self.background_data_for_shap = tensor_data[indices] 
                logger.info(f"({self.model_name}) Сохранен фоновый набор данных SHAP ({self.background_data_for_shap.shape[0]} экземпляров) на устройстве {self.background_data_for_shap.device}.")

                # 2. Создание ModelWrapperForSHAP для VAE
                wrapped_model_for_shap = self._VAEModelWrapperForSHAP(self.model) # Используем внутренний класс
                # wrapped_model_for_shap.eval() # _VAEModelWrapperForSHAP уже делает eval() в __init__

                # 3. Инициализация GradientExplainer
                try:
                    # GradientExplainer(model, data_tensor_or_list_of_tensors)
                    self.explainer = shap.GradientExplainer(wrapped_model_for_shap, self.background_data_for_shap)
                    logger.info(f"({self.model_name}) SHAP GradientExplainer инициализирован.")
                except Exception as e_shap_init:
                    logger.error(f"({self.model_name}) Ошибка при инициализации SHAP GradientExplainer: {e_shap_init}", exc_info=True)
                    self.explainer = None
                    self.background_data_for_shap = None # Сбрасываем, если эксплейнер не создался
            else:
                logger.warning(f"({self.model_name}) Недостаточно данных ({tensor_data.shape[0]} сэмплов) для создания фонового набора SHAP (требуется {self.shap_background_samples}).")
                self.explainer = None
                self.background_data_for_shap = None
        else: # Если условия для инициализации SHAP не выполнены
            if not shap: 
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: библиотека SHAP не доступна.")
            elif not self.model: 
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: модель VAE отсутствует.")
            elif self.background_data_for_shap is None:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: фоновые данные SHAP отсутствуют (None).")
            elif self.background_data_for_shap.shape[0] == 0:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: фоновые данные SHAP пусты (пустой тензор).")
            else:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: неизвестная причина.")
            
            self.explainer = None
            self.background_data_for_shap = None
        
        # Проверка деталей загруженных фоновых данных SHAP (вынос за пределы блока условий)
        if self.background_data_for_shap is not None:
            logger.info(f"({self.model_name}) Статус SHAP после загрузки: background_data shape={self.background_data_for_shap.shape}, device={self.background_data_for_shap.device}, explainer={'создан' if self.explainer is not None else 'отсутствует'}")
        else:
            logger.warning(f"({self.model_name}) Фоновые данные SHAP не были загружены или None. Объяснения будут недоступны.")

        # Устанавливаем итоговый статус обучения детектора
        # Теперь учитываем наличие threshold_ при установке is_trained
        if self.is_trained and self.model:
            self.is_trained = True
            logger.info(f"({self.model_name}) Детектор VAE (is_trained=True) успешно загружен.")
        else:
            self.is_trained = False
            missing_components = []
            if not self.model:
                missing_components.append("модель")
            if self.threshold_ is None:
                missing_components.append("порог аномальности")
            logger.warning(f"({self.model_name}) Детектор VAE не полностью загружен. Отсутствуют компоненты: {', '.join(missing_components)}")

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаруживает аномалии, вычисляя ошибку реконструкции VAE.
        Возвращает DataFrame с СЫРЫМИ скорами (ошибка реконструкции).
        """
        if not self.is_trained or self.model is None:
            logger.error(f"({self.model_name}) Детектор должен быть обучен перед использованием метода detect.")
            # Возвращаем исходный DataFrame с пустыми колонками аномалий
            result_df_error = data.copy()
            result_df_error['anomaly_score'] = np.nan
            result_df_error['is_anomaly'] = False
            return result_df_error

        logger.info(f"({self.model_name}) Запуск детекции...")
        start_time = time.time()
        
        # 1. Предобработка данных (возвращает DataFrame)
        # Метод preprocess теперь может вернуть немасштабированные данные, если скейлер не обучен
        processed_df = self.preprocess(data, fit_scaler=False)
        
        result_df = data.copy() 
        
        if processed_df is None or processed_df.empty:
            logger.error(f"({self.model_name}) Нет данных для детекции после предобработки или ошибка предобработки.")
            result_df['anomaly_score'] = np.nan
            result_df['is_anomaly'] = False
            return result_df
        
        # Проверяем, были ли данные масштабированы
        # Меняем проверку на более надежную, используя наличие атрибута mean_
        was_scaled = self.scaler is not None and hasattr(self.scaler, 'mean_')
        if not was_scaled:
            logger.warning(f"({self.model_name}) Данные не были масштабированы (скейлер не обучен). Результаты могут быть некорректными.")
        
        # 1.1 Конвертация в тензор
        tensor_data = torch.tensor(processed_df.values, dtype=torch.float32).to(self.device)
        if tensor_data.shape[0] == 0:
            logger.error(f"({self.model_name}) Нет данных для детекции: тензор пуст после конвертации.")
            result_df['anomaly_score'] = np.nan
            result_df['is_anomaly'] = False
            return result_df
            
        # 2. Получение скоров аномалий (ошибка реконструкции)
        self.model.eval() # Режим оценки
        all_scores = []
        dataset = TensorDataset(tensor_data)
        # Уменьшаем кол-во воркеров для Windows
        num_workers = 0 if os.name == 'nt' else 2 
        # Используем batch_size как при обучении или меньше, если данных мало
        effective_batch_size = min(self.batch_size, len(tensor_data)) if len(tensor_data) > 0 else 1
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=num_workers)

        with torch.no_grad():
            for batch_data in dataloader:
                inputs = batch_data[0].to(self.device)
                recon_batch, _, _ = self.model(inputs)
                # Скор = MSE ошибка реконструкции для каждого сэмпла
                batch_scores = nn.functional.mse_loss(recon_batch, inputs, reduction='none').mean(dim=1)
                all_scores.append(batch_scores.cpu().numpy())

        if not all_scores:
             anomaly_scores = np.full(len(data), np.nan)
        else:
             anomaly_scores = np.concatenate(all_scores)
             # Убедимся, что длина совпадает (на случай если dataloader что-то отбросил)
             if len(anomaly_scores) != len(data):
                 logger.warning(f"({self.model_name}) Длина скоров ({len(anomaly_scores)}) не совпадает с длиной данных ({len(data)}). Заполняем NaN.")
                 # Попробуем выровнять по индексу, если preprocess вернул тензор
                 # Но проще вернуть NaN для всех, если есть несовпадение
                 full_scores = np.full(len(data), np.nan)
                 # Пытаемся сопоставить, если индексы доступны (сложно без передачи индекса)
                 # Пока оставим как есть или заполним NaN
                 # Если tensor_data создавался из data[self.features], индексы должны совпадать
                 if len(anomaly_scores) == len(tensor_data): 
                      # anomaly_scores должен иметь тот же порядок, что и processed_df.index
                      result_df['anomaly_score'] = pd.Series(anomaly_scores, index=processed_df.index).reindex(result_df.index).values
                 else:
                      result_df['anomaly_score'] = np.nan 
             else:
                  result_df['anomaly_score'] = anomaly_scores

        # Заполняем NaN там, где скор не посчитался
        result_df['anomaly_score'] = result_df['anomaly_score'].fillna(np.nan) 
        if self.threshold_ is not None:
             # Сравниваем с порогом, обрабатывая NaN в скорах (они не будут аномалиями)
             result_df['is_anomaly'] = (result_df['anomaly_score'] >= self.threshold_).fillna(False)
        else:
             # Если порог не загружен (не должно происходить при is_trained=True)
             result_df['is_anomaly'] = False
             logger.warning(f"({self.model_name}) Порог аномальности не определен. is_anomaly установлен в False.")

        duration = time.time() - start_time
        logger.info(f"({self.model_name}) Детекция завершена. Время: {duration:.2f} сек.")
        return result_df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает словарь атрибутов для сохранения."""
        attrs = {
            'features': self.features,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'kld_weight': self.kld_weight,
            'dropout_rate': self.dropout_rate,
            'device': self.device,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'shap_background_samples': self.shap_background_samples,
            'model_name': self.model_name,
            'threshold_': self.threshold_,  # Добавляем порог аномальности
        }
        
        # Сохраняем background_data_for_shap, если он существует
        if self.background_data_for_shap is not None:
            try:
                # Конвертируем в numpy.array и сохраняем
                attrs['background_data_for_shap'] = self.background_data_for_shap.cpu().numpy() if torch.is_tensor(self.background_data_for_shap) else self.background_data_for_shap
                logger.info(f"({self.model_name}) Сохранены фоновые данные SHAP размерностью {attrs['background_data_for_shap'].shape}")
            except Exception as e:
                logger.warning(f"({self.model_name}) Не удалось сохранить фоновые данные SHAP: {e}")
                attrs['background_data_for_shap'] = None  # Явно устанавливаем None в случае ошибки
                logger.warning(f"({self.model_name}) Установлен background_data_for_shap=None после ошибки сохранения")
        else:
            attrs['background_data_for_shap'] = None
            logger.info(f"({self.model_name}) Фоновые данные SHAP отсутствуют при сохранении модели")
        
        # Расширенное сохранение параметров скейлера для последующей корректной инициализации
        if self.scaler is not None and hasattr(self.scaler, 'mean_'):
            attrs['scaler_params'] = {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'var': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
                'n_samples_seen': self.scaler.n_samples_seen_ if hasattr(self.scaler, 'n_samples_seen_') else None,
                # Важный атрибут для sklearn 0.24+
                'n_features_in_': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(self.features) if self.features else self.input_dim
            }
            attrs['is_fitted_scaler'] = True
        else:
            # Для моделей без обученного скейлера - четко указываем, что его нет
            attrs['scaler_params'] = None
            attrs['is_fitted_scaler'] = False
            logger.warning(f"({self.model_name}) При сохранении модели обнаружен необученный скейлер или скейлер отсутствует. Параметры скейлера не будут сохранены.")
            
        return attrs

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает дополнительные атрибуты в экземпляр."""
        self.features = loaded_data.get('features', [])
        self.input_dim = loaded_data.get('input_dim', len(self.features) if self.features else 0)
        self.latent_dim = loaded_data.get('latent_dim', 16) # Default or error
        self.hidden_dim = loaded_data.get('hidden_dim', 64) # Default or error
        self.epochs = loaded_data.get('epochs', 10)
        self.batch_size = loaded_data.get('batch_size', 64)
        self.learning_rate = loaded_data.get('learning_rate', 1e-3)
        self.kld_weight = loaded_data.get('kld_weight', 0.5) 
        self.dropout_rate = loaded_data.get('dropout_rate', 0.2)
        self.device = loaded_data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.shap_background_samples = loaded_data.get('shap_background_samples', 100)
        self.threshold_ = loaded_data.get('threshold_')  # Явно загружаем порог аномальности
        
        # Если порог не был сохранен, логируем предупреждение
        if self.threshold_ is None:
            logger.warning(f"({self.model_name}) Не удалось загрузить порог аномальности (threshold_)")
        else:
            logger.info(f"({self.model_name}) Загружен порог аномальности: {self.threshold_:.6f}")
        
        # Проверяем наличие флага is_fitted_scaler для обратной совместимости
        is_fitted_scaler = loaded_data.get('is_fitted_scaler', False)
        
        scaler_params = loaded_data.get('scaler_params')
        is_scaler_successfully_loaded = False  # Новый флаг успешности загрузки скейлера
        if scaler_params:
            self.scaler = StandardScaler()
            
            # Установка необходимых атрибутов для скейлера
            if scaler_params.get('mean') is not None: 
                self.scaler.mean_ = np.array(scaler_params['mean'])
                if scaler_params.get('scale') is not None: 
                    self.scaler.scale_ = np.array(scaler_params['scale'])
                    
                    if 'var' in scaler_params and scaler_params.get('var') is not None: 
                        self.scaler.var_ = np.array(scaler_params['var'])
                    else:
                        logger.warning(f"({self.model_name}) Отсутствует var_ в scaler_params, вычисляю из scale_.")
                        self.scaler.var_ = np.square(self.scaler.scale_)
                        
                    if 'n_samples_seen' in scaler_params and scaler_params.get('n_samples_seen') is not None:
                        self.scaler.n_samples_seen_ = scaler_params['n_samples_seen']
                    else:
                        logger.warning(f"({self.model_name}) Отсутствует n_samples_seen_ в scaler_params, использую 100.")
                        self.scaler.n_samples_seen_ = 100
                    
                    # Установка n_features_in_ - обязательного атрибута для sklearn 0.24+
                    self.scaler.n_features_in_ = scaler_params.get('n_features_in_', self.input_dim)
                    
                    # Устанавливаем флаг is_fitted_, чтобы метод transform работал без ошибок
                    # Этот флаг проверяется в sklearn.utils.validation.check_is_fitted
                    if hasattr(self.scaler, 'is_fitted_'):
                        self.scaler.is_fitted_ = True
                    
                    is_scaler_successfully_loaded = True
                    logger.info(f"({self.model_name}) StandardScaler успешно инициализирован с параметрами: n_features_in_={self.scaler.n_features_in_}, mean_={self.scaler.mean_.shape if hasattr(self.scaler, 'mean_') else None}, scale_={self.scaler.scale_.shape if hasattr(self.scaler, 'scale_') else None}")
                else:
                    logger.warning(f"({self.model_name}) Отсутствует scale_ в scaler_params. StandardScaler не будет полностью восстановлен.")
            else:
                logger.warning(f"({self.model_name}) Отсутствует mean_ в scaler_params. StandardScaler не будет полностью восстановлен.")
        else:
            logger.warning(f"({self.model_name}) Отсутствуют параметры StandardScaler в загруженной модели. Создан пустой экземпляр.")
            self.scaler = StandardScaler()
            # Если из scaler_params ничего не загружено, не считаем скейлер обученным
            is_scaler_successfully_loaded = False

        # Загрузка background_data_for_shap
        self.background_data_for_shap = None
        if 'background_data_for_shap' in loaded_data and loaded_data['background_data_for_shap'] is not None:
            try:
                # Загружаем фоновые данные и преобразуем в тензор PyTorch
                loaded_bg_numpy = loaded_data['background_data_for_shap']
                self.background_data_for_shap = torch.from_numpy(loaded_bg_numpy).float().to(self.device)
                logger.info(f"({self.model_name}) Фоновые данные SHAP загружены ({self.background_data_for_shap.shape})")
            except Exception as e:
                logger.warning(f"({self.model_name}) Не удалось загрузить фоновые данные SHAP: {e}")
                self.background_data_for_shap = None

        # Загружаем модель, если есть состояние
        model_loaded = False
        if loaded_data.get('model_state_dict'):
            try:
                # Убедимся, что input_dim корректно установлен перед созданием модели
                if not self.input_dim and self.features:
                    self.input_dim = len(self.features)
                elif not self.features: # Если нет признаков, не можем создать модель
                    logger.warning(f"({self.model_name}) Невозможно восстановить модель VAE: список признаков пуст.")
                    self.model = None
                    self.is_trained = False # Сбрасываем флаг, так как модель не восстановлена
                    return

                if not self.input_dim: # Если input_dim все еще 0
                    logger.warning(f"({self.model_name}) Невозможно восстановить модель VAE: input_dim равен 0.")
                    self.model = None
                    self.is_trained = False # Сбрасываем флаг
                    return

                current_vae_model = VAE(
                    input_dim=self.input_dim, 
                    latent_dim=self.latent_dim, 
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate
                )
                current_vae_model.load_state_dict(loaded_data['model_state_dict'])
                current_vae_model.to(self.device)
                self.model = current_vae_model
                model_loaded = True
                logger.info(f"({self.model_name}) Модель VAE успешно загружена и перемещена на {self.device}.")
            except Exception as e:
                logger.error(f"({self.model_name}) Ошибка при загрузке состояния модели VAE: {e}", exc_info=True)
                self.model = None # В случае ошибки, модель не будет загружена
        
        # Пересоздаем SHAP explainer при загрузке модели
        self.explainer = None # Сначала сбрасываем
        if shap and self.model and self.background_data_for_shap is not None and self.background_data_for_shap.shape[0] > 0:
            logger.info(f"({self.model_name}) Попытка пересоздания SHAP GradientExplainer после загрузки...")
            try:
                wrapped_model_cpu = self._VAEModelWrapperForSHAP(self.model.to('cpu')) # Обертка и модель на CPU
                background_data_cpu = self.background_data_for_shap.to('cpu')         # Фоновые данные на CPU
                self.explainer = shap.GradientExplainer(wrapped_model_cpu, background_data_cpu)
                logger.info(f"({self.model_name}) SHAP GradientExplainer успешно ПЕРЕСОЗДАН при загрузке модели. Device=cpu, background_data.shape={background_data_cpu.shape}")
                # Возвращаем модель на исходное устройство после создания эксплейнера
                self.model = self.model.to(self.device)
            except Exception as e_shap_reload:
                logger.error(f"({self.model_name}) Ошибка пересоздания SHAP GradientExplainer после загрузки: {e_shap_reload}", exc_info=True)
                self.explainer = None
                # Возвращаем модель на исходное устройство в случае ошибки
                self.model = self.model.to(self.device)
        else:
            # Логируем причину, если эксплейнер не может быть создан
            if not shap: 
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: библиотека SHAP не доступна.")
            elif not self.model: 
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: модель VAE отсутствует.")
            elif self.background_data_for_shap is None:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: фоновые данные SHAP отсутствуют (None).")
            elif self.background_data_for_shap.shape[0] == 0:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: фоновые данные SHAP пусты (пустой тензор).")
            else:
                logger.warning(f"({self.model_name}) SHAP Explainer не будет пересоздан: неизвестная причина.")
        
        # Проверка деталей загруженных фоновых данных SHAP (вынос за пределы блока условий)
        if self.background_data_for_shap is not None:
            logger.info(f"({self.model_name}) Статус SHAP после загрузки: background_data shape={self.background_data_for_shap.shape}, device={self.background_data_for_shap.device}, explainer={'создан' if self.explainer is not None else 'отсутствует'}")
        else:
            logger.warning(f"({self.model_name}) Фоновые данные SHAP не были загружены или None. Объяснения будут недоступны.")

        # Устанавливаем итоговый статус обучения детектора
        # Теперь учитываем наличие threshold_ при установке is_trained
        if model_loaded and is_scaler_successfully_loaded and self.threshold_ is not None:
            self.is_trained = True
            logger.info(f"({self.model_name}) Детектор VAE (is_trained=True) успешно загружен.")
        else:
            self.is_trained = False
            missing_components = []
            if not model_loaded:
                missing_components.append("модель")
            if not is_scaler_successfully_loaded:
                missing_components.append("скейлер")
            if self.threshold_ is None:
                missing_components.append("порог аномальности")
            logger.warning(f"({self.model_name}) Детектор VAE не полностью загружен. Отсутствуют компоненты: {', '.join(missing_components)}")

    def get_shap_explanations(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, float]]]:
        """
        Генерирует SHAP объяснения для предоставленных данных.

        Args:
            data_for_explanation_raw: DataFrame с "сырыми" данными (перед масштабированием),
                                      содержащий признаки, указанные в self.features.

        Returns:
            Список словарей, где каждый словарь представляет SHAP значения для одного экземпляра данных,
            или None, если объяснения не могут быть сгенерированы.
        """
        if not shap:
            logger.warning(f"({self.model_name}) Библиотека SHAP не установлена или не импортирована. Объяснения не будут сгенерированы.")
            return None
        if not self.is_trained or self.model is None:
            logger.warning(f"({self.model_name}) Модель не обучена или отсутствует. SHAP объяснения не будут сгенерированы.")
            return None
        
        current_explainer_to_use = self.explainer 

        if current_explainer_to_use is None:
            logger.warning(f"({self.model_name}) SHAP explainer не инициализирован.")
            # Попытка ленивой инициализации, если есть фоновые данные и модель
            if self.background_data_for_shap is not None and self.background_data_for_shap.shape[0] > 0 and self.model is not None:
                logger.info(f"({self.model_name}) Попытка ленивой инициализации GradientExplainer для SHAP...")
                try:
                    wrapped_model = self._VAEModelWrapperForSHAP(self.model)
                    # Попытка создать объяснитель на CPU для большей стабильности
                    background_data_cpu = self.background_data_for_shap.cpu()
                    model_cpu = wrapped_model.cpu() if hasattr(wrapped_model, 'cpu') else wrapped_model
                    current_explainer_to_use = shap.GradientExplainer(model_cpu, background_data_cpu)
                    # Сохраняем лениво созданный эксплейнер для будущего использования
                    self.explainer = current_explainer_to_use 
                    logger.info(f"({self.model_name}) GradientExplainer успешно создан (лениво) и сохранен.")
                except Exception as e_lazy_shap:
                    logger.error(f"({self.model_name}) Ошибка ленивой инициализации GradientExplainer: {e_lazy_shap}", exc_info=True)
                    return None # Не удалось создать эксплейнер
            else:
                logger.warning(f"({self.model_name}) Нет фоновых данных или модели для ленивой инициализации эксплейнера SHAP.")
                return None # Нет эксплейнера и не из чего создать

        if data_for_explanation_raw.empty:
            logger.warning(f"({self.model_name}) DataFrame для SHAP объяснений пуст.")
            return [] # Возвращаем пустой список, если нет данных для объяснения
        
        if not self.features: 
            logger.warning(f"({self.model_name}) Атрибут 'features' не установлен. Невозможно сгенерировать SHAP.")
            return None
        
        missing_features_in_input = [f for f in self.features if f not in data_for_explanation_raw.columns]
        if missing_features_in_input:
            logger.error(f"({self.model_name}) Отсутствуют необходимые признаки {missing_features_in_input} в data_for_explanation_raw для SHAP.")
            return None

        try:
            data_to_preprocess = data_for_explanation_raw[self.features].copy()
            processed_df_for_shap = self.preprocess(data_to_preprocess, fit_scaler=False) 
            
            if processed_df_for_shap is None or processed_df_for_shap.empty:
                logger.warning(f"({self.model_name}) SHAP: Нет данных после предобработки.")
                return [] 
            
            processed_tensor_for_shap = torch.tensor(processed_df_for_shap.values, dtype=torch.float32).cpu()

            if processed_tensor_for_shap.shape[0] == 0: 
                logger.warning(f"({self.model_name}) SHAP: Тензор для объяснения пуст после конвертации.")
                return []

            logger.info(f"({self.model_name}) Вычисление SHAP values для {processed_tensor_for_shap.shape[0]} экземпляров...")
            
            # Вместо nsamples='auto', используем явное количество сэмплов
            n_samples_for_shap = 50
            if self.background_data_for_shap is not None:
                n_samples_for_shap = min(50, self.background_data_for_shap.shape[0])
            logger.info(f"({self.model_name}) Используется nsamples={n_samples_for_shap} для SHAP")
            
            # Всегда перемещаем модель и данные на CPU для вычисления SHAP values
            # Это избегает проблем с torch.zeros() и device
            original_device = None
            if hasattr(self.model, 'to') and hasattr(next(self.model.parameters(), None), 'device'):
                original_device = next(self.model.parameters()).device
                self.model = self.model.cpu()
                
            # Создаем обертку для модели на CPU
            wrapped_model_cpu = self._VAEModelWrapperForSHAP(self.model)
                
            # Диагностическое логирование запроса SHAP объяснений
            background_data_exists_log = self.background_data_for_shap is not None and self.background_data_for_shap.shape[0] > 0 if torch.is_tensor(self.background_data_for_shap) else self.background_data_for_shap is not None
            logger.debug(f"({self.model_name}) Запрос SHAP объяснений: explainer={current_explainer_to_use is not None}, background_data_exists={background_data_exists_log}, model={self.model is not None}")
            
            # Используем CPU версию данных для фона
            background_data_cpu = self.background_data_for_shap.cpu() if self.background_data_for_shap is not None else None
            
            if background_data_cpu is None or background_data_cpu.shape[0] == 0:
                # Если нет фоновых данных, используем случайную выборку из текущих данных
                sample_size = min(50, processed_tensor_for_shap.shape[0])
                if sample_size > 0:
                    indices = np.random.choice(processed_tensor_for_shap.shape[0], sample_size, replace=False)
                    background_data_cpu = processed_tensor_for_shap[indices]
                    logger.info(f"({self.model_name}) Используем {sample_size} случайных образцов из текущих данных как фоновые")
                else:
                    logger.warning(f"({self.model_name}) Недостаточно данных для создания фоновой выборки")
                    if original_device is not None:
                        self.model = self.model.to(original_device)
                    return None
            
            # Используем GradientExplainer вместо DeepExplainer для совместимости с нашей оберткой
            try:
                logger.info(f"({self.model_name}) Создание GradientExplainer с размерами данных: {background_data_cpu.shape}")
                temp_explainer = shap.GradientExplainer(wrapped_model_cpu, background_data_cpu)
                shap_values_raw = temp_explainer.shap_values(processed_tensor_for_shap)
                
                # Добавляем детальное логирование формы SHAP значений
                logger.debug(f"({self.model_name}) SHAP values raw type: {type(shap_values_raw)}")
                if isinstance(shap_values_raw, np.ndarray):
                    logger.debug(f"({self.model_name}) SHAP values raw (ndarray) shape: {shap_values_raw.shape}")
                elif isinstance(shap_values_raw, list):
                    logger.debug(f"({self.model_name}) SHAP values raw is a list of length: {len(shap_values_raw)}")
                    for i, item in enumerate(shap_values_raw):
                        if isinstance(item, np.ndarray):
                            logger.debug(f"({self.model_name})   Item {i} shape: {item.shape}")
                        else:
                            logger.debug(f"({self.model_name})   Item {i} type: {type(item)}")
                
                logger.info(f"({self.model_name}) SHAP GradientExplainer успешно создан и вычислены SHAP-значения")
            except Exception as e:
                logger.error(f"({self.model_name}) Ошибка при использовании SHAP GradientExplainer: {e}", exc_info=True)
                # Возвращаем модель на исходное устройство
                if original_device is not None:
                    self.model = self.model.to(original_device)
                return None
            
            # Обработка результатов от GradientExplainer
            if isinstance(shap_values_raw, list):
                if len(shap_values_raw) == 1 and isinstance(shap_values_raw[0], np.ndarray):
                    shap_values_arr = shap_values_raw[0]
                    logger.debug(f"({self.model_name}) Извлечен один массив из списка SHAP values, shape: {shap_values_arr.shape}")
                else:
                    logger.warning(f"({self.model_name}) GradientExplainer вернул {len(shap_values_raw)} наборов SHAP-значений. Используем первый.")
                    if len(shap_values_raw) > 0 and isinstance(shap_values_raw[0], np.ndarray):
                        shap_values_arr = shap_values_raw[0]
                        logger.debug(f"({self.model_name}) Извлечен первый массив из списка SHAP values, shape: {shap_values_arr.shape}")
                    else:
                        logger.error(f"({self.model_name}) SHAP values_raw - это список, но его структура неожиданна.")
                        return None
            elif isinstance(shap_values_raw, np.ndarray):
                shap_values_arr = shap_values_raw
                logger.debug(f"({self.model_name}) SHAP values_raw уже является ndarray, shape: {shap_values_arr.shape}")
            else:
                logger.error(f"({self.model_name}) SHAP explainer вернул неожиданный тип: {type(shap_values_raw)}.")
                return None
            
            # Обработка случая с трехмерным массивом SHAP values
            shap_values_processed = None
            if shap_values_arr is not None:
                logger.debug(f"({self.model_name}) Начальная форма shap_values_arr: {shap_values_arr.shape}, ndim: {shap_values_arr.ndim}")
                
                if shap_values_arr.ndim == 3:
                    # Ожидаемая форма (num_samples, num_features, num_outputs_of_wrapper)
                    # Наша обертка имеет num_outputs_of_wrapper = 1
                    if shap_values_arr.shape[2] == 1:
                        shap_values_processed = shap_values_arr.squeeze(axis=2) # Удаляем последнее измерение, если оно = 1
                        logger.debug(f"({self.model_name}) SHAP values после squeeze(-1): {shap_values_processed.shape}")
                    else:
                        logger.error(f"({self.model_name}) SHAP values имеют 3 измерения, но последнее измерение ({shap_values_arr.shape[2]}) не равно 1.")
                        if original_device is not None: self.model.to(original_device)
                        return None
                elif shap_values_arr.ndim == 2:
                    # Предполагаем, что это уже (num_samples, num_features)
                    shap_values_processed = shap_values_arr
                    logger.debug(f"({self.model_name}) SHAP values уже 2D: {shap_values_processed.shape}")
                else:
                    logger.error(f"({self.model_name}) SHAP values имеют неожиданное количество измерений: {shap_values_arr.ndim}")
                    if original_device is not None: self.model.to(original_device)
                    return None

                # Проверка итоговой формы
                if shap_values_processed is None or \
                   shap_values_processed.ndim != 2 or \
                   shap_values_processed.shape[0] != processed_tensor_for_shap.shape[0] or \
                   shap_values_processed.shape[1] != len(self.features):
                    logger.error(f"({self.model_name}) Финальная форма SHAP values ({getattr(shap_values_processed, 'shape', 'N/A')}) некорректна. "
                                 f"Ожидалось: ({processed_tensor_for_shap.shape[0]}, {len(self.features)})")
                    if original_device is not None: self.model.to(original_device)
                    return None
                    
                explanations: List[Dict[str, float]] = []
                for i in range(shap_values_processed.shape[0]): # Используем num_samples из обработанного массива
                    instance_shap_values_flat = shap_values_processed[i] # Это уже (num_features,)
                    instance_explanation = {
                        feature_name: float(shap_val) 
                        for feature_name, shap_val in zip(self.features, instance_shap_values_flat)
                    }
                    explanations.append(instance_explanation)
                
                logger.info(f"({self.model_name}) SHAP values успешно вычислены для {len(explanations)} экземпляров.")
                if original_device is not None: self.model.to(original_device)
                return explanations
            else:
                logger.error(f"({self.model_name}) Не удалось извлечь корректный массив SHAP-значений (shap_values_arr is None) до обработки формы.")
                if original_device is not None: self.model.to(original_device)
                return None

        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при вычислении SHAP values: {e}", exc_info=True)
            # Убедимся, что модель вернулась на исходное устройство в случае исключения
            if 'original_device' in locals() and original_device is not None and hasattr(self.model, 'to'):
                self.model = self.model.to(original_device)
            return None