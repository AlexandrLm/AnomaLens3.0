import pandas as pd
import numpy as np
import networkx as nx
import os
import joblib # Добавляем импорт joblib
from typing import Dict, Any, List, Optional, Tuple
from .detector import AnomalyDetector # Импортируем базовый класс
import logging # Добавляем импорт logging

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

class GraphAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе графовых метрик.
    Анализирует связи между заказами, товарами, продавцами и клиентами.
    Метрики нормализуются индивидуально перед взвешенным суммированием.
    """
    def __init__(self, 
                 model_name: str = "graph_features", 
                 # Флаги включения метрик
                 use_seller_state_spread: bool = True,
                 use_category_diversity: bool = True,
                 use_seller_degree: bool = True,
                 # Веса для взвешенной суммы метрик
                 seller_state_weight: float = 1.0,
                 category_diversity_weight: float = 1.0,
                 seller_degree_weight: float = 1.0,
                 ): 
        super().__init__(model_name)
        # Сохраняем параметры конфигурации
        self.use_seller_state_spread = use_seller_state_spread
        self.use_category_diversity = use_category_diversity
        self.use_seller_degree = use_seller_degree
        # Сохраняем веса
        self.seller_state_weight = seller_state_weight
        self.category_diversity_weight = category_diversity_weight
        self.seller_degree_weight = seller_degree_weight
        
        # В self.model будут храниться статистики (min/max для каждой метрики)
        self.model = {
            'metrics_min_max': {} # Словарь для хранения (min, max) для каждой метрики
        }
        self.scaler = None # Не используется

    def _build_graph(self, data: pd.DataFrame) -> nx.Graph:
        """
        Строит граф NetworkX на основе входного DataFrame.

        Args:
            data (pd.DataFrame): DataFrame, содержащий как минимум колонки: 
                                 order_id, order_item_id, product_id, seller_id, 
                                 price, freight_value, customer_id, 
                                 product_category_name, seller_state, customer_state.

        Returns:
            nx.Graph: Построенный граф.
        """
        G = nx.Graph()
        logger.info("Построение графа...")
        start_time = pd.Timestamp.now()

        # Добавляем узлы с атрибутами
        # Используем set для быстрой проверки уникальности
        added_orders = set()
        added_products = set()
        added_sellers = set()
        added_customers = set()

        # Итерация по строкам DataFrame (order_items)
        for index, row in data.iterrows(): # index здесь - оригинальный индекс DF
            # Узел OrderItem (основной узел)
            # Индекс может быть не уникальным, если DF не индексирован по order_item_id
            # Используем комбинацию order_id и order_item_id для уникальности узла
            order_item_node_id = f"item_{row['order_id']}_{row['order_item_id']}"
            G.add_node(order_item_node_id, 
                       node_type='OrderItem', 
                       price=row['price'], 
                       freight_value=row['freight_value'],
                       original_index=index # Сохраняем исходный индекс DF
                       )

            # Узел Order
            order_node_id = str(row['order_id'])
            if order_node_id not in added_orders:
                G.add_node(order_node_id, node_type='Order', customer_id=row['customer_id'])
                added_orders.add(order_node_id)
            G.add_edge(order_item_node_id, order_node_id) # Связь OrderItem <-> Order

            # Узел Customer
            customer_node_id = str(row['customer_id'])
            if customer_node_id not in added_customers:
                G.add_node(customer_node_id, node_type='Customer', state=row.get('customer_state'))
                added_customers.add(customer_node_id)
            G.add_edge(order_node_id, customer_node_id) # Связь Order <-> Customer
            
            # Узел Product
            product_node_id = str(row['product_id'])
            if product_node_id not in added_products:
                G.add_node(product_node_id, node_type='Product', category=row.get('product_category_name'))
                added_products.add(product_node_id)
            G.add_edge(order_item_node_id, product_node_id) # Связь OrderItem <-> Product
            
            # Узел Seller
            seller_node_id = str(row['seller_id'])
            if seller_node_id not in added_sellers:
                G.add_node(seller_node_id, node_type='Seller', state=row.get('seller_state'))
                added_sellers.add(seller_node_id)
            G.add_edge(order_item_node_id, seller_node_id) # Связь OrderItem <-> Seller

        end_time = pd.Timestamp.now()
        logger.info(f"Граф построен за {(end_time - start_time).total_seconds():.2f} сек. " 
              f"Узлов: {G.number_of_nodes()}, Ребер: {G.number_of_edges()}")
        return G

    def preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Предобработка данных. Для графа, возможно, просто возвращает данные как есть,
        или выполняет минимальную очистку/проверку необходимых колонок.
        """
        # Проверяем наличие необходимых колонок
        required_cols = ['order_id', 'order_item_id', 'product_id', 'seller_id', 
                         'price', 'freight_value', 'customer_id',
                         'product_category_name', 'seller_state', 'customer_state'] # Добавили customer_state
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют необходимые колонки для построения графа: {missing_cols}")
        
        # Можно добавить обработку пропусков в ключевых колонках ID, если необходимо
        # data = data.dropna(subset=['order_id', 'order_item_id', 'product_id', 'seller_id', 'customer_id'])
        
        # scaler не используется, fit_scaler игнорируется
        return data

    def train(self, data: pd.DataFrame):
        """
        Метод обучения. Вычисляет min/max для каждой графовой метрики 
        на обучающих данных и обучает нормализатор для итогового взвешенного скора.
        """
        logger.info(f"Запуск 'обучения' для {self.model_name}...")
        processed_data = self.preprocess(data)
        if processed_data.empty:
            logger.warning(f"Обучение {self.model_name} невозможно: нет данных.")
            self.is_trained = False
            return
            
        # 1. Построить граф на обучающих данных
        graph = self._build_graph(processed_data)
        
        # 2. Рассчитать СЫРЫЕ значения графовых метрик для всех узлов OrderItem
        logger.info("Расчет сырых графовых метрик для обучения...")
        raw_metrics_data = {
            'seller_states': [],
            'categories': [],
            'seller_degrees': []
        }
        final_combined_scores = [] # Для обучения итогового нормализатора
        
        item_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'OrderItem']
        if not item_nodes:
            logger.warning(f"Обучение {self.model_name} невозможно: нет узлов OrderItem в графе.")
            self.is_trained = False
            return
            
        # Первый проход: собираем сырые значения
        for item_node in item_nodes:
            metrics = self._calculate_raw_metrics(graph, item_node) # Новая функция для сырых метрик
            if self.use_seller_state_spread: raw_metrics_data['seller_states'].append(metrics.get('num_states', np.nan))
            if self.use_category_diversity: raw_metrics_data['categories'].append(metrics.get('num_categories', np.nan))
            if self.use_seller_degree: raw_metrics_data['seller_degrees'].append(metrics.get('seller_degree', np.nan))

        # 3. Вычислить и сохранить min/max для каждой метрики
        metrics_min_max = {}
        for metric_name, values in raw_metrics_data.items():
            valid_values = np.array([v for v in values if pd.notna(v)])
            if len(valid_values) > 0:
                 min_val = float(np.min(valid_values))
                 max_val = float(np.max(valid_values))
                 # Обработка случая min == max
                 if max_val == min_val:
                     logger.warning(f"Предупреждение ({self.model_name}): Все значения для метрики '{metric_name}' одинаковы ({min_val}). Нормализация этой метрики вернет 0.5.")
                 metrics_min_max[metric_name] = (min_val, max_val)
            else:
                 logger.warning(f"Предупреждение ({self.model_name}): Нет валидных значений для метрики '{metric_name}'.")
                 metrics_min_max[metric_name] = (0.0, 1.0) # Дефолтные значения
        
        self.model['metrics_min_max'] = metrics_min_max
        logger.info(f"Min/Max для метрик вычислены: {self.model['metrics_min_max']}")

        # 4. Рассчитать итоговые взвешенные скоры для обучения нормализатора
        logger.info("Расчет итоговых взвешенных скоров для обучения нормализатора...")
        # Используем собранные сырые значения
        num_items = len(item_nodes)
        for i in range(num_items):
            # Нормализуем сырые значения, ТОЛЬКО если метрика используется
            norm_num_states = np.nan
            if self.use_seller_state_spread and i < len(raw_metrics_data['seller_states']):
                norm_num_states = self._normalize_metric(raw_metrics_data['seller_states'][i], 'seller_states')
            
            norm_num_categories = np.nan
            if self.use_category_diversity and i < len(raw_metrics_data['categories']):
                norm_num_categories = self._normalize_metric(raw_metrics_data['categories'][i], 'categories')
                
            norm_seller_degree = np.nan
            if self.use_seller_degree and i < len(raw_metrics_data['seller_degrees']):
                norm_seller_degree = self._normalize_metric(raw_metrics_data['seller_degrees'][i], 'seller_degrees')
            
            # Считаем взвешенную сумму
            combined_score = 0.0
            total_weight = 0.0
            if self.use_seller_state_spread and pd.notna(norm_num_states):
                combined_score += norm_num_states * self.seller_state_weight
                total_weight += self.seller_state_weight
            if self.use_category_diversity and pd.notna(norm_num_categories):
                combined_score += norm_num_categories * self.category_diversity_weight
                total_weight += self.category_diversity_weight
            if self.use_seller_degree and pd.notna(norm_seller_degree):
                combined_score += norm_seller_degree * self.seller_degree_weight
                total_weight += self.seller_degree_weight
                
            # Усредняем по весам (если есть хотя бы одна метрика)
            final_score = combined_score / total_weight if total_weight > 0 else 0.0
            final_combined_scores.append(final_score)
            
        # 5. Обучить нормализатор на итоговых взвешенных скорах
        if final_combined_scores:
             self.fit_normalizer(np.array(final_combined_scores))
        else:
             logger.warning("Не удалось рассчитать итоговые скоры для обучения нормализатора.")
             self.min_score_ = 0.0
             self.max_score_ = 1.0
             
        self.is_trained = True # Считаем обученным, если есть min/max метрик и нормализатор
        logger.info(f"'Обучение' {self.model_name} завершено (min/max метрик и нормализатор настроены).")

    def _normalize_metric(self, value: Optional[float], metric_name: str) -> Optional[float]:
        """Нормализует значение метрики к [0, 1] используя сохраненные min/max."""
        if pd.isna(value):
            return np.nan
            
        min_max_tuple = self.model.get('metrics_min_max', {}).get(metric_name)
        if not min_max_tuple:
            # Если min/max не найдены (не должно происходить после train)
            logger.warning(f"Предупреждение: Не найдены min/max для метрики '{metric_name}'. Возвращаем 0.5.")
            return 0.5
            
        min_val, max_val = min_max_tuple
        
        if max_val == min_val:
            return 0.5 # Или 0? Возвращаем 0.5 для консистентности
        else:
            normalized = (value - min_val) / (max_val - min_val)
            return float(np.clip(normalized, 0.0, 1.0))

    def _calculate_raw_metrics(self, graph: nx.Graph, item_node_id: str) -> Dict[str, float]:
        """Вычисляет СЫРЫЕ значения метрик для одного узла OrderItem."""
        raw_metrics = {}
        
        # --- Поиск связанных узлов (как было в _calculate_anomaly_score) ---
        item_data = graph.nodes[item_node_id]
        order_node_id = None
        seller_node_id = None
        
        for neighbor in graph.neighbors(item_node_id):
            node_type = graph.nodes[neighbor].get('node_type')
            if node_type == 'Order': order_node_id = neighbor
            elif node_type == 'Seller': seller_node_id = neighbor

        if not order_node_id:
            logger.warning(f"Предупреждение (_calculate_raw_metrics): Не найден Order для Item {item_node_id}")
            return raw_metrics 
            
        order_items_in_same_order = [n for n in graph.neighbors(order_node_id) 
                                     if graph.nodes[n].get('node_type') == 'OrderItem']
        # -----------------------------------------------------------------

        # --- Метрика 1: Географический разброс продавцов --- 
        if self.use_seller_state_spread:
            seller_states_in_order = set()
            for oi_node in order_items_in_same_order:
                current_seller_id = None
                for neighbor in graph.neighbors(oi_node):
                    if graph.nodes[neighbor].get('node_type') == 'Seller':
                        current_seller_id = neighbor
                        break
                if current_seller_id:
                    seller_state = graph.nodes[current_seller_id].get('state')
                    if seller_state:
                        seller_states_in_order.add(seller_state)
            raw_metrics['num_states'] = float(len(seller_states_in_order))
        
        # --- Метрика 2: Разнообразие категорий --- 
        if self.use_category_diversity:
            categories_in_order = set()
            for oi_node in order_items_in_same_order:
                current_product_id = None
                for neighbor in graph.neighbors(oi_node):
                     if graph.nodes[neighbor].get('node_type') == 'Product':
                         current_product_id = neighbor
                         break
                if current_product_id:
                    category = graph.nodes[current_product_id].get('category')
                    if category:
                         categories_in_order.add(category)
            raw_metrics['num_categories'] = float(len(categories_in_order))

        # --- Метрика 3: Степень узла Продавца --- 
        if self.use_seller_degree:
            seller_degree = 0.0
            if seller_node_id: 
                seller_degree = float(graph.degree(seller_node_id))
            else:
                logger.warning(f"Предупреждение (_calculate_raw_metrics): Не найден Seller для Item {item_node_id}")
            raw_metrics['seller_degree'] = seller_degree
            
        return raw_metrics

    def _calculate_anomaly_score(self, graph: nx.Graph, item_node_id: str) -> float:
        """
        Вычисляет итоговый взвешенный скор аномальности для одного узла OrderItem.
        Использует нормализованные метрики.
        """
        # 1. Получаем сырые метрики
        raw_metrics = self._calculate_raw_metrics(graph, item_node_id)
        
        # 2. Нормализуем каждую метрику
        norm_num_states = self._normalize_metric(raw_metrics.get('num_states'), 'seller_states')
        norm_num_categories = self._normalize_metric(raw_metrics.get('num_categories'), 'categories')
        norm_seller_degree = self._normalize_metric(raw_metrics.get('seller_degree'), 'seller_degrees') # Используем ключ 'seller_degrees' как в min_max
        
        # 3. Считаем взвешенную сумму
        combined_score = 0.0
        total_weight = 0.0
        if self.use_seller_state_spread and pd.notna(norm_num_states):
            combined_score += norm_num_states * self.seller_state_weight
            total_weight += self.seller_state_weight
        if self.use_category_diversity and pd.notna(norm_num_categories):
            combined_score += norm_num_categories * self.category_diversity_weight
            total_weight += self.category_diversity_weight
        if self.use_seller_degree and pd.notna(norm_seller_degree):
            combined_score += norm_seller_degree * self.seller_degree_weight
            total_weight += self.seller_degree_weight
            
        # Усредняем по весам (если есть хотя бы одна метрика)
        final_score = combined_score / total_weight if total_weight > 0 else 0.0
           
        # Этот метод должен возвращать СЫРОЙ (но уже взвешенный и основанный на норм. метриках) скор.
        # Окончательная нормализация к [0,1] делается через self.normalize_score(),
        # который был обучен в train() на этих final_score.
        return float(final_score)

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаруживает аномалии с использованием графовых метрик.
        """
        if not self.is_trained:
            logger.warning(f"Ошибка: Детектор {self.model_name} не обучен или не загружен корректно. Детекция невозможна.")
            result_df = data.copy()
            # Устанавливаем is_anomaly в False и anomaly_score в NaN или нейтральное значение (например, 0.0 или 0.5)
            # чтобы последующая обработка в ансамбле не вызывала ошибок и этот детектор не влиял на результат.
            result_df['is_anomaly'] = False 
            result_df['anomaly_score'] = 0.0 # Используем 0.0 как нейтральный скор, который не будет считаться аномалией
                                          # и не сломает normalize_score в ансамбле, если он все же будет вызван.
            return result_df

        processed_data = self.preprocess(data)
        if processed_data.empty:
            logger.warning(f"Детекция {self.model_name} невозможна: нет данных.")
            result_df = data.copy()
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = np.nan
            return result_df
            
        # 1. Строим граф
        graph = self._build_graph(processed_data)
        
        # 2. Считаем скоры для всех узлов OrderItem
        results = {}
        item_nodes = {n: d for n, d in graph.nodes(data=True) if d.get('node_type') == 'OrderItem'}
        
        logger.info(f"Расчет графовых аномальных скоров для {len(item_nodes)} узлов OrderItem...")
        for item_node_id, node_data in item_nodes.items():
            original_index = node_data['original_index']
            raw_score = self._calculate_anomaly_score(graph, item_node_id)
            results[original_index] = raw_score # Сохраняем сырой скор по исходному индексу
            
        # 3. Создаем DataFrame с результатами
        result_series = pd.Series(results, name='anomaly_score')
        result_df = data.join(result_series)
        
        # 4. Нормализуем скоры (если нормализатор обучен)
        # Метод detect должен возвращать СЫРЫЕ скоры.
        # Нормализация происходит позже в ensemble_service.
        # Здесь мы пока не определяем is_anomaly, т.к. нет порога для графовых метрик.
        result_df['is_anomaly'] = False # Заглушка
        result_df['anomaly_score'] = result_df['anomaly_score'].fillna(0.0) # Заполняем NaN нулями? Или оставлять?
        
        logger.info(f"Детектор {self.model_name}: обнаружено {result_df['is_anomaly'].sum()} аномалий из {len(result_df)} ({len(results)} обработано графом)")
        return result_df

    def _get_attributes_to_save(self) -> Dict[str, Any]:
        """Возвращает специфичные атрибуты GraphAnomalyDetector для сохранения."""
        # self.model (содержащий metrics_min_max), min_score_, max_score_ (для итогового нормализатора),
        # и параметры конфигурации.
        # is_trained не сохраняем, он определяется при загрузке.
        # scaler здесь None.
        return {
            'model_state': self.model,  # self.model здесь это {'metrics_min_max': {...}}
            # min_score_ и max_score_ для итогового нормализатора будут сохранены базовым классом.
            
            # Параметры конфигурации (из конструктора)
            'use_seller_state_spread': self.use_seller_state_spread,
            'use_category_diversity': self.use_category_diversity,
            'use_seller_degree': self.use_seller_degree,
            'seller_state_weight': self.seller_state_weight,
            'category_diversity_weight': self.category_diversity_weight,
            'seller_degree_weight': self.seller_degree_weight,
        }

    def _load_additional_attributes(self, loaded_data: Dict[str, Any]) -> None:
        """Загружает специфичные атрибуты GraphAnomalyDetector."""
        # Базовый load_model уже попытался загрузить self.model (из 'model_state'), 
        # self.scaler (будет None), self.min_score_, self.max_score_.

        # Если self.model (из model_state) не загрузился или некорректен, инициализируем его
        if self.model is None or not isinstance(self.model, dict) or 'metrics_min_max' not in self.model:
            # Попытка загрузить по старому ключу 'model' если 'model_state' не сработал
            old_model_format = loaded_data.get('model')
            if old_model_format and isinstance(old_model_format, dict) and 'metrics_min_max' in old_model_format:
                self.model = old_model_format
                logger.info(f"Загружен self.model для {self.model_name} из старого ключа 'model'.")
            else:
                self.model = {'metrics_min_max': {}} # Гарантируем структуру
                logger.warning(f"model (содержащий metrics_min_max) не найден или некорректен для {self.model_name}. Инициализирован по умолчанию.")
        
        # Загружаем параметры конфигурации. Используем текущие значения экземпляра как fallback.
        self.use_seller_state_spread = loaded_data.get('use_seller_state_spread', self.use_seller_state_spread)
        self.use_category_diversity = loaded_data.get('use_category_diversity', self.use_category_diversity)
        self.use_seller_degree = loaded_data.get('use_seller_degree', self.use_seller_degree)
        self.seller_state_weight = loaded_data.get('seller_state_weight', self.seller_state_weight)
        self.category_diversity_weight = loaded_data.get('category_diversity_weight', self.category_diversity_weight)
        self.seller_degree_weight = loaded_data.get('seller_degree_weight', self.seller_degree_weight)

        # Определяем is_trained
        valid_model_metrics = False
        if isinstance(self.model, dict) and 'metrics_min_max' in self.model and isinstance(self.model['metrics_min_max'], dict):
            # Считаем валидным, если словарь metrics_min_max существует, даже если он пуст
            # (потому что train может его оставить пустым, если ни одна метрика не используется)
            valid_model_metrics = True 
            # Дополнительная проверка: если метрики используются, то metrics_min_max не должен быть пустым
            any_metric_actually_used = self.use_seller_state_spread or self.use_category_diversity or self.use_seller_degree
            if any_metric_actually_used and not bool(self.model['metrics_min_max']):
                logger.warning(f"{self.model_name}: metrics_min_max пуст, хотя графовые метрики включены. Это может быть ошибкой обучения.")
                # Не ставим valid_model_metrics в False, так как формально структура есть, но логируем.

        normalizer_loaded = self.min_score_ is not None and self.max_score_ is not None

        if valid_model_metrics and normalizer_loaded:
            self.is_trained = True
        else:
            self.is_trained = False
            missing_parts_log = []
            if not valid_model_metrics:
                missing_parts_log.append("статистики метрик (model.metrics_min_max)")
            if not normalizer_loaded:
                missing_parts_log.append("параметры итогового нормализатора (min_score_, max_score_)")
            if missing_parts_log:
                logger.warning(f"{self.model_name} не считается обученным после загрузки. Отсутствуют: {', '.join(missing_parts_log)}.")

        self._custom_is_trained_logic_applied_in_load_additional = True

    def _reset_state(self):
        """Сбрасывает состояние детектора."""
        super()._reset_state()
        self.model = {'metrics_min_max': {}}
        logger.info(f"Сброс состояния GraphAnomalyDetector {self.model_name} выполнен.")
        
    def get_explanation_details(self, data_for_explanation_raw: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """
        Генерирует детали для объяснения аномалий на основе графовых метрик.
        
        Args:
            data_for_explanation_raw: DataFrame с "сырыми" данными для объяснения.
            
        Returns:
            List[Dict[str, Any]]: Список объяснений для каждой строки данных или None, если возникла ошибка.
        """
        if not self.is_trained or not self.model.get('metrics_min_max'):
            logger.warning(f"({self.model_name}) Детектор не обучен или отсутствуют статистики метрик. Невозможно предоставить объяснения.")
            return None
        
        try:
            # Предобработка данных и построение графа
            processed_df = self.preprocess(data_for_explanation_raw.copy())
            if processed_df.empty:
                logger.warning(f"({self.model_name}) После предобработки данные пусты.")
                return None
                
            graph = self._build_graph(processed_df)
            
            # Получаем словарь узлов OrderItem с их индексами в исходном DataFrame
            item_nodes_map = {
                node_data['original_index']: node_id
                for node_id, node_data in graph.nodes(data=True) 
                if node_data.get('node_type') == 'OrderItem' and 'original_index' in node_data
            }
            
            explanations = []
            for index, row in processed_df.iterrows():
                item_node_id = item_nodes_map.get(index)
                if item_node_id:
                    # Получаем сырые метрики для текущего узла OrderItem
                    raw_metrics = self._calculate_raw_metrics(graph, item_node_id)
                    
                    # Нормализуем метрики
                    normalized_metrics = {
                        'norm_num_states': self._normalize_metric(raw_metrics.get('num_states'), 'seller_states'),
                        'norm_num_categories': self._normalize_metric(raw_metrics.get('num_categories'), 'categories'),
                        'norm_seller_degree': self._normalize_metric(raw_metrics.get('seller_degree'), 'seller_degrees')
                    }
                    
                    # Вычисляем итоговый скор
                    combined_score = 0.0
                    total_weight = 0.0
                    
                    if self.use_seller_state_spread and pd.notna(normalized_metrics['norm_num_states']):
                        combined_score += normalized_metrics['norm_num_states'] * self.seller_state_weight
                        total_weight += self.seller_state_weight
                    
                    if self.use_category_diversity and pd.notna(normalized_metrics['norm_num_categories']):
                        combined_score += normalized_metrics['norm_num_categories'] * self.category_diversity_weight
                        total_weight += self.category_diversity_weight
                    
                    if self.use_seller_degree and pd.notna(normalized_metrics['norm_seller_degree']):
                        combined_score += normalized_metrics['norm_seller_degree'] * self.seller_degree_weight
                        total_weight += self.seller_degree_weight
                    
                    final_score = combined_score / total_weight if total_weight > 0 else 0.0
                    
                    # Подготовка словаря с объяснением
                    explanation_info = {
                        "raw_graph_metrics": {
                            "num_seller_states": float(raw_metrics.get('num_states')) if 'num_states' in raw_metrics and pd.notna(raw_metrics.get('num_states')) else None,
                            "num_product_categories": float(raw_metrics.get('num_categories')) if 'num_categories' in raw_metrics and pd.notna(raw_metrics.get('num_categories')) else None,
                            "seller_degree": float(raw_metrics.get('seller_degree')) if 'seller_degree' in raw_metrics and pd.notna(raw_metrics.get('seller_degree')) else None
                        },
                        "normalized_graph_metrics": {
                            "norm_num_seller_states": float(normalized_metrics['norm_num_states']) if pd.notna(normalized_metrics['norm_num_states']) else None,
                            "norm_num_product_categories": float(normalized_metrics['norm_num_categories']) if pd.notna(normalized_metrics['norm_num_categories']) else None,
                            "norm_seller_degree": float(normalized_metrics['norm_seller_degree']) if pd.notna(normalized_metrics['norm_seller_degree']) else None
                        },
                        "weights": {
                            "seller_state_weight": float(self.seller_state_weight) if self.use_seller_state_spread else 0.0,
                            "category_diversity_weight": float(self.category_diversity_weight) if self.use_category_diversity else 0.0,
                            "seller_degree_weight": float(self.seller_degree_weight) if self.use_seller_degree else 0.0
                        },
                        "final_score": float(final_score),
                        "explanation": self._generate_text_explanation(raw_metrics, normalized_metrics)
                    }
                    
                    explanations.append({"detector_specific_info": explanation_info})
                else:
                    explanations.append({"detector_specific_info": {"error": "OrderItem node not found in temp graph"}})
            
            return explanations
            
        except Exception as e:
            logger.error(f"({self.model_name}) Ошибка при генерации объяснений: {e}", exc_info=True)
            return None
    
    def _generate_text_explanation(self, raw_metrics: Dict[str, float], normalized_metrics: Dict[str, float]) -> str:
        """Генерирует текстовое объяснение на основе метрик."""
        explanation_parts = []
        
        # Объяснение для каждой метрики
        if self.use_seller_state_spread and 'num_states' in raw_metrics:
            num_states = raw_metrics.get('num_states')
            if pd.notna(num_states):
                if num_states > 1:
                    explanation_parts.append(f"Заказ содержит товары из {int(num_states)} разных штатов продавцов, что {self._get_text_anomaly_level(normalized_metrics['norm_num_states'])}.")
        
        if self.use_category_diversity and 'num_categories' in raw_metrics:
            num_categories = raw_metrics.get('num_categories')
            if pd.notna(num_categories):
                if num_categories > 1:
                    explanation_parts.append(f"Заказ содержит {int(num_categories)} разных категорий товаров, что {self._get_text_anomaly_level(normalized_metrics['norm_num_categories'])}.")
        
        if self.use_seller_degree and 'seller_degree' in raw_metrics:
            seller_degree = raw_metrics.get('seller_degree')
            if pd.notna(seller_degree):
                explanation_parts.append(f"Продавец связан с {int(seller_degree)} другими узлами в графе, что {self._get_text_anomaly_level(normalized_metrics['norm_seller_degree'])}.")
        
        if explanation_parts:
            return " ".join(explanation_parts)
        return "Недостаточно данных для формирования объяснения."
    
    def _get_text_anomaly_level(self, normalized_score: Optional[float]) -> str:
        """Возвращает текстовое описание уровня аномальности."""
        if normalized_score is None or pd.isna(normalized_score):
            return "неопределенно"
        if normalized_score >= 0.8:
            return "крайне необычно"
        elif normalized_score >= 0.6:
            return "весьма необычно"
        elif normalized_score >= 0.4:
            return "немного необычно"
        else:
            return "обычно"