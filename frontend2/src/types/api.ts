// Общие модели данных из документации
export interface PaginatedResponse<T> {
  total: number;
  items: T[];
}

export interface TaskCreationResponse {
  task_id: string;
  message: string;
  status_endpoint: string;
  initial_status: string;
}

export interface TaskStatusResult {
  task_id: string;
  status: string; // "pending", "processing", "completed", "failed", "completed_no_data", "completed_with_errors"
  start_time: string; // ISO 8601
  end_time?: string; // ISO 8601
  details: string;
  result?: Record<string, any>;
  error_type?: string;
}

// Модели для аномалий
export interface RootAnomalySchema {
  id: number;
  order_item_id?: number | null;
  order_id: string;
  detection_date: string; // ISO 8601
  anomaly_score?: number | null;
  detector_type: string;
  details?: Record<string, any> | null; // Уже был Record<string, any> | null, оставляем так
}

export interface LLMExplanationResponse {
  anomaly_id: number;
  original_details: RootAnomalySchema;
  llm_explanation: string;
}

// Параметры для запроса списка аномалий
export interface FetchAnomaliesParams {
  skip?: number;
  limit?: number;
  start_date?: string; // ISO 8601
  end_date?: string; // ISO 8601
  min_score?: number;
  max_score?: number;
  detector_type?: string;
}

export interface DeleteAllAnomaliesResponse {
  message: string;
}

// Модели для запросов обучения и детекции ОДИНОЧНЫХ моделей (из предыдущего шага, могут быть полезны)
export interface TrainModelRequest {
  start_date?: string; // ISO 8601, По умолчанию "2000-01-01T00:00:00"
  end_date?: string;   // ISO 8601
  detector_type: string; // По умолчанию "isolation_forest"
  detector_config_payload: Record<string, any>; // Конфигурация модели
}

export interface DetectAnomaliesRequest {
  start_date?: string; // ISO 8601, По умолчанию "2000-01-01T00:00:00"
  end_date?: string;   // ISO 8601
  detector_type: string; // По умолчанию "isolation_forest"
  detector_config_payload: Record<string, any>; // Конфигурация модели
}

// Для задач Celery (используется на страницах обучения/детекции)
export interface TaskStatusResponse {
  task_id: string;
  status: string; // e.g., PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
  result?: any; // Результат выполнения задачи, если она завершена успешно
  error?: string; // Сообщение об ошибке, если задача не удалась
  progress?: number; // От 0 до 100, если задача поддерживает прогресс
  details?: Record<string, any>; // Дополнительные детали о задаче
}

// >>> ДОБАВЛЕНО: Модели для Многоуровневой Детекции Аномалий

export interface DetectorStatus {
  is_trained: boolean;
  detector_type: string;
  model_filename?: string | null;
  expected_path?: string | null;
  exists?: boolean | null;
  can_load?: boolean | null;
  error_message?: string | null;
  params_from_config?: Record<string, any> | null;
  internal_params?: Record<string, any> | null;
}

export type DetectorLevelStatus = Record<string, DetectorStatus>; // "detector_name1": { ...DetectorStatus... }

export interface MultilevelStatus {
  transaction_level: DetectorLevelStatus;
  behavior_level: DetectorLevelStatus;
  time_series_level: DetectorLevelStatus;
  // Можно добавить и другие уровни, если API их вернет
  [key: string]: DetectorLevelStatus; // Для гибкости, если появятся новые уровни
}

export interface MultilevelDetectorConfigEntry {
  type: string;
  model_filename?: string | null;
  weight?: number | null;
  // ...другие параметры...
  [key: string]: any; // Для дополнительных параметров конфигурации детектора
}

export interface MultilevelConfig {
  transaction_level: MultilevelDetectorConfigEntry[];
  behavior_level: MultilevelDetectorConfigEntry[];
  time_series_level: MultilevelDetectorConfigEntry[];
  combination_weights: {
    transaction: number;
    behavior: number;
    time_series: number;
    [key: string]: number; // Для гибкости
  };
  // Можно добавить и другие уровни, если API их вернет/примет
  [key: string]: any; // Для гибкости, если появятся новые уровни конфигурации
}

export interface DetectionParams {
  transaction_threshold?: number; // (default: 0.6)
  behavior_threshold?: number;    // (default: 0.6)
  time_series_threshold?: number; // (default: 0.6)
  final_threshold?: number;       // (default: 0.5)
  filter_period_days?: number | null; // (default: 10000)
}

export interface AvailableDetectorsResponse {
  transaction_level: string[];
  behavior_level: string[];
  time_series_level: string[];
  [key: string]: string[]; // Для гибкости
}

// <<< КОНЕЦ ДОБАВЛЕННЫХ МОДЕЛЕЙ

// >>> ДОБАВЛЕНО: Модели для Продуктов

export interface ProductCategoryNameTranslationSchema {
  product_category_name: string; // Оригинальное название категории
  product_category_name_english: string; // Переведенное название
}

export interface ProductSchema {
  product_id: string;
  product_category_name?: string | null;
  product_name_lenght?: number | null; // Опечатка в документации (lenght), используем как есть или уточняем
  product_description_lenght?: number | null; // Опечатка в документации (lenght)
  product_photos_qty?: number | null;
  product_weight_g?: number | null;
  product_length_cm?: number | null;
  product_height_cm?: number | null;
  product_width_cm?: number | null;
  // Поле category_translation было в примере в документации, но не в описании схемы.
  // Если оно приходит от API, его нужно добавить.
  // Предположим, что оно приходит как часть ProductSchema по аналогии с другими вложенными объектами.
  category_translation?: ProductCategoryNameTranslationSchema | null; 
}

// <<< КОНЕЦ ДОБАВЛЕННЫХ МОДЕЛЕЙ для Продуктов

// >>> ДОБАВЛЕНО/ОБНОВЛЕНО: Модели для Заказов, Клиентов, Продавцов, Отзывов, Платежей, Элементов Заказа

// Если эти схемы еще не определены из других разделов, добавляем их:
export interface CustomerSchema { // Используется в OrderSchema
  customer_id: string;
  customer_unique_id: string;
  customer_zip_code_prefix: number;
  customer_city: string;
  customer_state: string;
}

export interface SellerSchema { // Используется в OrderItemSchema
  seller_id: string;
  seller_zip_code_prefix: number;
  seller_city: string;
  seller_state: string;
}

export interface OrderItemSchema { // Используется в OrderSchema
  order_item_id: number; // Это ID элемента в заказе, а не ID самого продукта
  product_id: string;
  seller_id: string;
  shipping_limit_date: string; // ISO 8601
  price: number;
  freight_value: number;
  // Добавим опционально ProductSchema и SellerSchema, если API их возвращает во вложенном виде
  product?: ProductSchema | null;
  seller?: SellerSchema | null; 
}

export interface OrderPaymentSchema { // Используется в OrderSchema
  order_id: string; // Повторяется, но может быть полезно
  payment_sequential: number;
  payment_type: string;
  payment_installments: number;
  payment_value: number;
}

export interface OrderReviewSchema { // Используется в OrderSchema
  review_id: string;
  order_id: string; // Повторяется
  review_score: number;
  review_comment_title?: string | null;
  review_comment_message?: string | null;
  review_creation_date: string; // ISO 8601
  review_answer_timestamp: string; // ISO 8601
}


export interface OrderSchema {
  order_id: string;
  customer_id: string;
  order_status: string;
  order_purchase_timestamp: string; // ISO 8601
  order_approved_at?: string | null; // ISO 8601
  order_delivered_carrier_date?: string | null; // ISO 8601
  order_delivered_customer_date?: string | null; // ISO 8601
  order_estimated_delivery_date?: string | null; // ISO 8601
  
  // Вложенные данные согласно документации
  customer?: CustomerSchema | null; // API может возвращать customer как объект или null
  items: OrderItemSchema[];
  payments: OrderPaymentSchema[];
  reviews: OrderReviewSchema[];
}

// Параметры для запроса списка заказов
export interface FetchOrdersParams {
  skip?: number;
  limit?: number;
  start_date?: string; // ISO 8601
  end_date?: string;   // ISO 8601
}

// Параметры для запроса списка переводов
export interface FetchTranslationsParams {
  skip?: number;
  limit?: number;
}

// <<< КОНЕЦ ДОБАВЛЕННЫХ/ОБНОВЛЕННЫХ МОДЕЛЕЙ

// >>> ДОБАВЛЕНО: Модели для Геолокации

export interface GeolocationSchema {
  id: number;
  geolocation_zip_code_prefix: number;
  geolocation_lat: number;
  geolocation_lng: number;
  geolocation_city: string;
  geolocation_state: string;
}

// Параметры для запроса списка геолокаций
export interface FetchGeolocationParams {
  skip?: number;
  limit?: number;
}

// <<< КОНЕЦ ДОБАВЛЕННЫХ МОДЕЛЕЙ для Геолокации

// Параметры для запроса списка отзывов
export interface FetchReviewsParams {
  skip?: number;
  limit?: number;
}

// Параметры для запроса списка продавцов (если еще не определен)
export interface FetchSellersParams {
  skip?: number;
  limit?: number;
}

// Параметры для запроса списка клиентов (если еще не определен)
export interface FetchCustomersParams {
  skip?: number;
  limit?: number;
}

export interface AnomalyLevelScores {
  transaction_score: number;
  behavior_score: number;
  time_series_score: number;
  [key: string]: any;
}

export interface AnomalyDetails {
  product_id?: string | null;
  seller_id?: string | null;
  final_threshold_used?: number | null;
  level_scores?: AnomalyLevelScores | null;
  level_thresholds?: Record<string, number> | null; // e.g. { transaction: 0.5, ... }
  contributing_detectors_explanations?: ContributingDetectorsExplanations | null;
  [key: string]: any;
}

export interface DetectorExplanationEntry {
  detector_name: string;
  score: number;
  explanation?: string | null; // Приоритет ниже, если есть explanation_text
  explanation_text?: string | null; // Приоритет выше
  [key: string]: any; // для других возможных полей
}

export interface ContributingDetectorsExplanations {
  transaction_level?: DetectorExplanationEntry[] | null;
  behavior_level?: DetectorExplanationEntry[] | null;
  time_series_level?: DetectorExplanationEntry[] | null;
  [key: string]: DetectorExplanationEntry[] | null | undefined; // Разрешаем undefined для индексной сигнатуры
}

export interface ApiStatusResponse {
  message: string;
  environment: string;
  // Можно добавить версию API, время работы и т.д., если API это возвращает
}
