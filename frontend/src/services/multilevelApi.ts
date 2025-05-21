import axios from 'axios';
import { 
  MultilevelStatus, 
  MultilevelConfig, 
  DetectionParams, 
  DetectResponse, 
  TrainResponse,
  AvailableDetectors
} from '../types/multilevel';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';
const MULTILEVEL_BASE_URL = `${API_BASE_URL}/multilevel`;

const apiClient = axios.create({
  baseURL: MULTILEVEL_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Получает текущий статус обученности многоуровневой системы
 */
export const getMultilevelStatus = async (): Promise<MultilevelStatus> => {
  const response = await apiClient.get<MultilevelStatus>('/status');
  return response.data;
};

/**
 * Получает текущую конфигурацию многоуровневой системы
 */
export const getMultilevelConfig = async (): Promise<MultilevelConfig> => {
  const response = await apiClient.get<MultilevelConfig>('/config');
  return response.data;
};

/**
 * Обновляет конфигурацию многоуровневой системы
 */
export const updateMultilevelConfig = async (config: MultilevelConfig): Promise<boolean> => {
  const response = await apiClient.post<boolean>('/config', config);
  return response.data;
};

/**
 * Запускает обучение всех детекторов многоуровневой системы
 */
export const trainMultilevelSystem = async (): Promise<TrainResponse> => {
  const response = await apiClient.post<TrainResponse>('/train');
  return response.data;
};

/**
 * Запускает детекцию аномалий с помощью многоуровневой системы
 */
export const detectWithMultilevelSystem = async (params: DetectionParams): Promise<DetectResponse> => {
  const response = await apiClient.post<DetectResponse>('/detect', params);
  return response.data;
};

/**
 * Получает список доступных типов детекторов для каждого уровня
 */
export const getAvailableDetectors = async (): Promise<AvailableDetectors> => {
  const response = await apiClient.get<AvailableDetectors>('/available-detectors');
  return response.data;
}; 