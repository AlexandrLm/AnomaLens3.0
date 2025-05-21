import type {
  PaginatedResponse,
  RootAnomalySchema,
  LLMExplanationResponse,
  FetchAnomaliesParams,
  DeleteAllAnomaliesResponse,
} from '../types/api';

const API_BASE_URL = '/api'; // Как указано в документации

// // Имитация задержки сети
// const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// // Мок данные для аномалий
// let mockAnomalies: RootAnomalySchema[] = Array.from({ length: 25 }, (_, i) => ({
//   id: i + 1,
//   order_item_id: 1000 + i,
//   order_id: `ORDER_XYZ_${100 + i}`,
//   detection_date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
//   anomaly_score: Math.random() > 0.3 ? parseFloat(Math.random().toFixed(2)) : null,
//   detector_type: i % 3 === 0 ? 'isolation_forest' : (i % 3 === 1 ? 'statistical' : 'rules_based'),
//   details: {
//     reason: `Mock detail ${i + 1}`,
//     value: Math.random() * 100,
//     context: { some_field: `value_${i}` }
//   },
// }));

// export const fetchAnomalies = async (
//   params: FetchAnomaliesParams
// ): Promise<PaginatedResponse<RootAnomalySchema>> => {
//   console.log('Fetching anomalies with params:', params);
//   await sleep(500); // Имитация загрузки

//   // Простая имитация фильтрации и пагинации
//   let filteredItems = [...mockAnomalies];

//   if (params.start_date) {
//     filteredItems = filteredItems.filter(item => new Date(item.detection_date) >= new Date(params.start_date!));
//   }
//   if (params.end_date) {
//     filteredItems = filteredItems.filter(item => new Date(item.detection_date) <= new Date(params.end_date!));
//   }
//   if (params.detector_type) {
//     filteredItems = filteredItems.filter(item => item.detector_type.includes(params.detector_type!));
//   }
//   if (params.min_score !== undefined) {
//     filteredItems = filteredItems.filter(item => item.anomaly_score !== null && item.anomaly_score !== undefined && item.anomaly_score >= params.min_score!);
//   }
//   if (params.max_score !== undefined) {
//     filteredItems = filteredItems.filter(item => item.anomaly_score !== null && item.anomaly_score !== undefined && item.anomaly_score <= params.max_score!);
//   }

//   const total = filteredItems.length;
//   const skip = params.skip || 0;
//   const limit = params.limit || 10;
//   const items = filteredItems.slice(skip, skip + limit);

//   // Имитация ошибки (раскомментируйте для теста)
//   // if (Math.random() > 0.8) {
//   //   throw new Error("Failed to fetch anomalies (mock error)");
//   // }

//   return { total, items };
// };

// export const createAnomaly = async (
//   data: RootAnomalyCreateSchema
// ): Promise<RootAnomalySchema> => {
//   console.log('Creating anomaly:', data);
//   await sleep(500);
//   const newId = mockAnomalies.length > 0 ? Math.max(...mockAnomalies.map(a => a.id)) + 1 : 1;
//   const newAnomaly: RootAnomalySchema = {
//     ...data,
//     id: newId,
//     details: typeof data.details === 'string' ? JSON.parse(data.details) : data.details,
//     detection_date: data.detection_date || new Date().toISOString(), // Убедимся, что дата есть
//   };
//   mockAnomalies.unshift(newAnomaly); // Добавляем в начало
//   return newAnomaly;
// };

// export const deleteAnomaly = async (anomalyId: number): Promise<RootAnomalySchema> => {
//   console.log('Deleting anomaly:', anomalyId);
//   await sleep(300);
//   const index = mockAnomalies.findIndex(a => a.id === anomalyId);
//   if (index === -1) {
//     throw new Error(`Аномалия с ID ${anomalyId} не найдена (mock error)`);
//   }
//   const deleted = mockAnomalies.splice(index, 1)[0];
//   return deleted;
// };

// export const fetchLlmExplanation = async (
//   anomalyId: number
// ): Promise<LLMExplanationResponse> => {
//   console.log('Fetching LLM explanation for anomaly:', anomalyId);
//   await sleep(1000);
//   const anomaly = mockAnomalies.find(a => a.id === anomalyId);
//   if (!anomaly) {
//     throw new Error(`Аномалия с ID ${anomalyId} не найдена для LLM (mock error)`);
//   }
//   return {
//     anomaly_id: anomalyId,
//     original_details: anomaly,
//     llm_explanation: `Это объяснение LLM для аномалии ${anomalyId}. Тип детектора ${anomaly.detector_type} и оценка ${anomaly.anomaly_score} указывают на потенциальную проблему с ${JSON.stringify(anomaly.details)}. Возможно, требуется дальнейшее расследование.`,
//   };
// };


// Для реальных запросов (закомментировано, пока используем моки):

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch (e) {
      errorData = { message: response.statusText };
    }
    throw new Error(errorData?.detail || errorData?.message || `HTTP error ${response.status}`);
  }
  // Для DELETE запросов, которые могут возвращать 204 No Content или JSON
  if (response.status === 204) {
    return {} as T; // Возвращаем пустой объект, если нет контента
  }
  return response.json() as Promise<T>;
}

export const fetchAnomalies = async (
  params: FetchAnomaliesParams
): Promise<PaginatedResponse<RootAnomalySchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  if (params.start_date) queryParams.append('start_date', params.start_date);
  if (params.end_date) queryParams.append('end_date', params.end_date);
  if (params.min_score !== undefined) queryParams.append('min_score', String(params.min_score));
  if (params.max_score !== undefined) queryParams.append('max_score', String(params.max_score));
  if (params.detector_type) queryParams.append('detector_type', params.detector_type);

  const response = await fetch(`${API_BASE_URL}/anomalies/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<RootAnomalySchema>>(response);
};

export const deleteAnomaly = async (anomalyId: number): Promise<RootAnomalySchema> => {
  const response = await fetch(`${API_BASE_URL}/anomalies/${anomalyId}`, {
    method: 'DELETE',
  });
  return handleResponse<RootAnomalySchema>(response);
};

export const fetchLlmExplanation = async (
  anomalyId: number
): Promise<LLMExplanationResponse> => {
  const response = await fetch(`${API_BASE_URL}/anomalies/${anomalyId}/explain-llm`);
  return handleResponse<LLMExplanationResponse>(response);
};

export const deleteAllAnomalies = async (): Promise<DeleteAllAnomaliesResponse> => {
  const response = await fetch(`${API_BASE_URL}/anomalies/`, {
    method: 'DELETE',
  });
  return handleResponse<DeleteAllAnomaliesResponse>(response);
}; 