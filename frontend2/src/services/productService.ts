import type { PaginatedResponse, ProductSchema } from '../types/api';

const API_BASE_URL = '/api/products';

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

export interface FetchProductsParams {
  skip?: number;
  limit?: number;
}

export const fetchProducts = async (
  params: FetchProductsParams
): Promise<PaginatedResponse<ProductSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));

  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<ProductSchema>>(response);
};

export const fetchProductById = async (productId: string): Promise<ProductSchema> => {
  const literalPathSegment = "%7Bproduct_id%7D";
  const queryParams = new URLSearchParams({ item_id: productId });

  const response = await fetch(`${API_BASE_URL}/${literalPathSegment}?${queryParams.toString()}`);
  return handleResponse<ProductSchema>(response);
}; 