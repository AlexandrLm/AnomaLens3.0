// Определяет структуру конфигурации модели для API
export interface ModelConfig {
  type: 'statistical' | 'isolation_forest' | 'autoencoder' | 'graph';
  model_filename: string;
  feature?: string; // Для statistical
  threshold?: number; // Для statistical
  features?: string[]; // Для isolation_forest, autoencoder
  weight?: number; // Вес детектора в ансамбле
  // Параметры для graph
  use_seller_state_spread?: boolean;
  use_category_diversity?: boolean;
  use_seller_degree?: boolean;
  seller_state_weight?: number;
  category_diversity_weight?: number;
  seller_degree_weight?: number;
  // Другие возможные параметры инициализации...
  [key: string]: any; // Позволяет добавлять другие параметры
}

// Определяет структуру статуса модели из API (/models/status)
export interface ModelStatus {
  trained: boolean;
  path: string;
  message: string;
  features: string[] | null;
  config: ModelConfig;
} 