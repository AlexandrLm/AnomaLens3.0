import type { PaginatedResponse, OrderReviewSchema, FetchReviewsParams } from '../types/api';

const API_BASE_URL = '/api/reviews';

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

export const fetchReviews = async (
  params: FetchReviewsParams
): Promise<PaginatedResponse<OrderReviewSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<OrderReviewSchema>>(response);
};

export const fetchReviewById = async (reviewId: string): Promise<OrderReviewSchema> => {
  const literalPathSegment = "%7Breview_id%7D";
  const queryParams = new URLSearchParams({ item_id: reviewId });

  const response = await fetch(`${API_BASE_URL}/${literalPathSegment}?${queryParams.toString()}`);
  return handleResponse<OrderReviewSchema>(response);
}; 