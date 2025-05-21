import type { TaskStatusResult } from '../types/api';

const API_BASE_URL = '/api'; // Базовый URL API

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
  // Проверяем, есть ли тело ответа перед парсингом JSON
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.indexOf("application/json") !== -1) {
    return response.json() as Promise<T>;
  }
  // Если нет JSON, но ответ успешный (например, 204 No Content), возвращаем null или что-то подходящее
  // В данном случае, TaskStatusResult всегда ожидается как JSON, так что это больше для общего handleResponse
  return null as T; // Или можно бросить ошибку, если JSON строго обязателен
}

/**
 * Получает статус фоновой задачи.
 * @param relativeTaskStatusEndpoint Относительный путь к эндпоинту статуса задачи, например, /tasks/task_status/{task_id}
 */
export const fetchTaskStatus = async (relativeTaskStatusEndpoint: string): Promise<TaskStatusResult> => {
  // Убедимся, что путь начинается с /
  const fullEndpoint = relativeTaskStatusEndpoint.startsWith('/') ? relativeTaskStatusEndpoint : `/${relativeTaskStatusEndpoint}`;
  const response = await fetch(`${API_BASE_URL}${fullEndpoint}`);
  return handleResponse<TaskStatusResult>(response);
}; 