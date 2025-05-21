import axios from 'axios';
import { 
    Anomaly, 
    AnomalyContextResponse, 
    AnomalyFilters,
    PaginatedResponse
} from './../types/anomaly';
import { ModelStatus, ModelConfig } from './../types/models';
import { 
    TaskStatus,
    TaskMessageResponse,
    TaskLaunchResponse
} from './../types/tasks';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

interface GetAnomaliesParams {
  skip?: number;
  limit?: number;
  start_date?: string;
  end_date?: string;
  min_score?: number;
  max_score?: number;
  detector_type?: string;
}

interface DetectEnsembleParams {
  start_date?: string;
  end_date?: string;
  ensemble_threshold?: number;
  combination_method?: 'average' | 'max' | 'weighted_average';
}

interface TrainDetectParams {
  start_date?: string;
  end_date?: string;
}

export const getModelsStatus = async (): Promise<Record<string, ModelStatus>> => {
  const response = await apiClient.get<{ [key: string]: ModelStatus }>('/anomalies/models/status');
  return response.data;
};

export const trainModel = async (config: ModelConfig, params?: TrainDetectParams): Promise<TaskMessageResponse> => {
  const response = await apiClient.post<TaskMessageResponse>('/anomalies/train', config, { params });
  return response.data;
};

export const detectAnomalies = async (config: ModelConfig, params?: TrainDetectParams): Promise<TaskMessageResponse> => {
  const response = await apiClient.post<TaskMessageResponse>('/anomalies/detect', config, { params });
  return response.data;
};

export const detectEnsemble = async (params?: DetectEnsembleParams): Promise<TaskLaunchResponse> => {
  const response = await apiClient.post<TaskLaunchResponse>('/anomalies/detect/ensemble', null, { params });
  return response.data;
};

/** Получить список сохраненных аномалий с пагинацией */
export const getAnomalies = async (params?: AnomalyFilters): Promise<PaginatedResponse<Anomaly>> => {
  const response = await apiClient.get<PaginatedResponse<Anomaly>>('/anomalies/', { params });
  return response.data;
};

export const getTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  try {
    const response = await apiClient.get<TaskStatus>(`/anomalies/task_status/${taskId}`); 
    return response.data;
  } catch (error) {
    console.error(`Error fetching status for task ${taskId}:`, error);
    const details = error instanceof Error ? error.message : 'Unknown error';
    return {
      task_id: taskId,
      status: 'error',
      details: `Failed to fetch status: ${details}`,
      start_time: new Date().toISOString(),
      end_time: new Date().toISOString(),
      result: null,
    };
  }
};

export const getAnomalyContext = async (anomalyId: number): Promise<AnomalyContextResponse> => {
  try {
    const response = await apiClient.get<AnomalyContextResponse>(`/anomalies/${anomalyId}/context`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching context for anomaly ${anomalyId}:`, error);
    if (axios.isAxiosError(error) && error.response) {
        const errorMsg = `Failed to fetch anomaly context: ${error.response.status} ${error.response.statusText}. ${error.response.data?.detail || ''}`.trim();
        throw new Error(errorMsg);
    } else if (error instanceof Error) {
        throw new Error(`Failed to fetch anomaly context: ${error.message}`);
    }
    throw new Error('Failed to fetch anomaly context due to an unexpected error.');
  }
};