// Определяет структуру статуса фоновой задачи

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'error' | 'completed_no_data'; // Added 'completed_no_data' from API doc
  start_time?: string; // ISO string (время постановки в очередь)
  start_processing_time?: string; // ISO string (время начала фактической обработки)
  end_time?: string; // ISO string
  details?: string;
  result?: {
    anomalies_count?: number; // Example for detect/ensemble
    anomalies?: Record<string, any>[]; // Example for detect/ensemble
    // Fields from API doc example for /detect/ensemble result
    total_detected_anomalies?: number;
    newly_saved_anomalies_count?: number;
    newly_saved_anomaly_ids?: number[];
    skipped_duplicates_count?: number;
    // Other fields depending on task type
    [key: string]: any; 
  } | null;
  error_type?: string; // Появляется, если status == "failed"
}

// Тип для ответа от эндпоинтов запуска задач, которые НЕ возвращают task_id сразу
// (например, /anomalies/train, /anomalies/detect для отдельных моделей)
export interface TaskMessageResponse {
  message: string;
}

// Тип для ответа от эндпоинтов запуска задач, которые ВОЗВРАЩАЮТ task_id
// (например, /anomalies/detect/ensemble)
export interface TaskLaunchResponse {
  task_id: string;
  status: string; // Обычно 'pending'
  message: string;
}