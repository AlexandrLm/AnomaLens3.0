import type { PaginatedResponse, ProductCategoryNameTranslationSchema, FetchTranslationsParams } from '../types/api';

const API_BASE_URL = '/api/translations';

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

export const fetchTranslations = async (
  params: FetchTranslationsParams
): Promise<PaginatedResponse<ProductCategoryNameTranslationSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<ProductCategoryNameTranslationSchema>>(response);
};

// API для получения одного перевода по product_category_name
export const fetchTranslationByName = async (categoryName: string): Promise<ProductCategoryNameTranslationSchema> => {
  // API ожидает, что product_category_name будет частью URL path, поэтому его нужно правильно закодировать
  const encodedCategoryName = encodeURIComponent(categoryName);
  const response = await fetch(`${API_BASE_URL}/${encodedCategoryName}`);
  return handleResponse<ProductCategoryNameTranslationSchema>(response);
}; 