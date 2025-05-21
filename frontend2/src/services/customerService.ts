import type { PaginatedResponse, CustomerSchema, FetchCustomersParams } from '../types/api';

const API_BASE_URL = '/api/customers';

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

export const fetchCustomers = async (
  params: FetchCustomersParams
): Promise<PaginatedResponse<CustomerSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<CustomerSchema>>(response);
};

export const fetchCustomerById = async (customerId: string): Promise<CustomerSchema> => {
  const literalPathSegment = "%7Bcustomer_id%7D";
  const queryParams = new URLSearchParams({ item_id: customerId });

  const response = await fetch(`${API_BASE_URL}/${literalPathSegment}?${queryParams.toString()}`);
  return handleResponse<CustomerSchema>(response);
}; 