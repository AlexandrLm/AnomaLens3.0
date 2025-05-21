import type { ApiStatusResponse } from '../types/api';

// Заметьте, базовый URL здесь пустой, так как эндпоинт /
const API_BASE_URL = ''; 

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
  return response.json() as Promise<T>;
}

export const fetchApiStatus = async (): Promise<ApiStatusResponse> => {
  const response = await fetch(`${API_BASE_URL}/`);
  return handleResponse<ApiStatusResponse>(response);
}; 