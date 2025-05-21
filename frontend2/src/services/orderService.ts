import type { PaginatedResponse, OrderSchema, FetchOrdersParams } from '../types/api';

const API_BASE_URL = '/api/orders';

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

export const fetchOrders = async (
  params: FetchOrdersParams
): Promise<PaginatedResponse<OrderSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  if (params.start_date) queryParams.append('start_date', params.start_date);
  if (params.end_date) queryParams.append('end_date', params.end_date);
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<OrderSchema>>(response);
};

export const fetchOrderById = async (orderId: string): Promise<OrderSchema> => {
  const response = await fetch(`${API_BASE_URL}/${orderId}`);
  return handleResponse<OrderSchema>(response);
}; 