import type { PaginatedResponse, SellerSchema, FetchSellersParams } from '../types/api';

const API_BASE_URL = '/api/sellers';

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

export const fetchSellers = async (
  params: FetchSellersParams
): Promise<PaginatedResponse<SellerSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<SellerSchema>>(response);
};

export const fetchSellerById = async (sellerId: string): Promise<SellerSchema> => {
  // Предполагаем, что API для получения продавца по ID использует тот же формат URL, что и для продуктов/клиентов
  // Если формат другой (например, /api/sellers/{seller_id} без query параметра), это нужно будет скорректировать.
  const literalPathSegment = "%7Bseller_id%7D"; // Предполагаемый сегмент, если API его требует
  const queryParams = new URLSearchParams({ item_id: sellerId }); // Предполагаемый query параметр

  // Итоговый URL: /api/sellers/%7Bseller_id%7D?item_id=<sellerId_value>
  // или /api/sellers/<sellerId_value> если API проще
  // Используем вариант с query параметром, аналогично customerService
  const response = await fetch(`${API_BASE_URL}/${literalPathSegment}?${queryParams.toString()}`);
  // Если API ожидает просто /api/sellers/{seller_id}:
  // const response = await fetch(`${API_BASE_URL}/${sellerId}`);
  return handleResponse<SellerSchema>(response);
}; 