import type {
  MultilevelStatus,
  MultilevelConfig,
  DetectionParams,
  AvailableDetectorsResponse,
  TaskCreationResponse, // Общий тип для задач
} from '../types/api';

const API_BASE_URL = '/api/multilevel';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch (e) {
      // Если тело ответа не JSON или пустое, используем response.statusText
      errorData = { message: response.statusText };
    }
    // FastAPI часто возвращает ошибки в поле "detail"
    throw new Error(errorData?.detail || errorData?.message || `HTTP error ${response.status}`);
  }
  // Для POST /config, который возвращает просто true (200 OK), но может не быть JSON
  if (response.headers.get("content-type")?.includes("application/json")) {
    return response.json() as Promise<T>;
  }
  // Если ответ не JSON, но успешный (например, true для POST /config)
  // В данном случае API возвращает `true` как JSON, так что этот кейс может не понадобиться,
  // но оставим для общей обработки.
  // Если бэкенд возвращает просто `true` не как JSON, а как текст:
  // const text = await response.text();
  // if (text === 'true') return true as unknown as T;
  // return text as unknown as T;
  // Так как API возвращает JSON { "success": true } или просто true как JSON, json() должно работать.
  // Если POST /config возвращает только статус 200 OK и тело "true" (не json):
  // const textData = await response.text();
  // return (textData === 'true') as any; // Предполагаем, что T будет boolean
  return response.json() as Promise<T>; // По документации /api/multilevel/config возвращает true (булево значение в JSON).
}


export const fetchMultilevelStatus = async (): Promise<MultilevelStatus> => {
  const response = await fetch(`${API_BASE_URL}/status`);
  return handleResponse<MultilevelStatus>(response);
};

export const fetchMultilevelConfig = async (): Promise<MultilevelConfig> => {
  const response = await fetch(`${API_BASE_URL}/config`);
  return handleResponse<MultilevelConfig>(response);
};

export const updateMultilevelConfig = async (config: MultilevelConfig): Promise<boolean> => { // API возвращает true
  const response = await fetch(`${API_BASE_URL}/config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  });
  // API возвращает true (булево значение)
  const result = await handleResponse<{ success: boolean } | boolean>(response);
  if (typeof result === 'boolean') {
    return result;
  }
  return result.success; // Если бэкенд возвращает { "success": true }
};

export const trainMultilevelSystem = async (): Promise<TaskCreationResponse> => {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
  });
  return handleResponse<TaskCreationResponse>(response);
};

export const detectMultilevelAnomalies = async (params: DetectionParams): Promise<TaskCreationResponse> => {
  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  return handleResponse<TaskCreationResponse>(response);
};

export const fetchAvailableDetectors = async (): Promise<AvailableDetectorsResponse> => {
  const response = await fetch(`${API_BASE_URL}/available-detectors`);
  return handleResponse<AvailableDetectorsResponse>(response);
}; 