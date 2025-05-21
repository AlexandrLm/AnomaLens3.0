// Типы для многоуровневой системы обнаружения аномалий

// Статус детектора многоуровневой системы
export interface MultilevelDetectorStatus {
  is_trained: boolean;
  type: string;
}

// Статус всей многоуровневой системы
export interface MultilevelStatus {
  transaction_level: Record<string, MultilevelDetectorStatus>;
  behavior_level: Record<string, MultilevelDetectorStatus>;
  time_series_level: Record<string, MultilevelDetectorStatus>;
}

// Конфигурация детектора в системе
export interface MultilevelDetectorConfig {
  type: string;
  model_filename: string;
  weight: number;
  threshold: number;
  [key: string]: any; // Дополнительные параметры, специфичные для каждого типа детектора
}

// Веса для комбинации результатов разных уровней
export interface CombinationWeights {
  transaction: number;
  behavior: number;
  time_series: number;
}

// Полная конфигурация многоуровневой системы
export interface MultilevelConfig {
  transaction_level: MultilevelDetectorConfig[];
  behavior_level: MultilevelDetectorConfig[];
  time_series_level: MultilevelDetectorConfig[];
  combination_weights: CombinationWeights;
}

// Параметры для запуска детекции
export interface DetectionParams {
  transaction_threshold: number;
  behavior_threshold: number;
  time_series_threshold: number;
  final_threshold: number;
  filter_period_days?: number | null;
}

// Статистика сохранения обнаруженных аномалий
export interface SaveStatistics {
  total_detected_anomalies_before_save: number;
  newly_saved_anomalies_count: number;
  newly_saved_anomaly_ids: number[];
  skipped_duplicates_count: number;
  errors_on_save: number;
}

// Ответ API на запрос детекции
export interface DetectResponse {
  status: 'success' | 'error';
  message: string;
  elapsed_time_seconds: number;
  save_statistics: SaveStatistics;
}

// Ответ API на запрос обучения
export interface TrainResponse {
  status: 'success' | 'error';
  message: string;
}

// Доступные типы детекторов
export interface AvailableDetectors {
  transaction_level: string[];
  behavior_level: string[];
  time_series_level: string[];
} 