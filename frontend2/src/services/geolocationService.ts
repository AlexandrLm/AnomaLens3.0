import type { PaginatedResponse, GeolocationSchema, FetchGeolocationParams } from '../types/api';

const API_BASE_URL = '/api/geolocation';

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

export const fetchGeolocations = async (
  params: FetchGeolocationParams
): Promise<PaginatedResponse<GeolocationSchema>> => {
  const queryParams = new URLSearchParams();
  if (params.skip !== undefined) queryParams.append('skip', String(params.skip));
  if (params.limit !== undefined) queryParams.append('limit', String(params.limit));
  
  const response = await fetch(`${API_BASE_URL}/?${queryParams.toString()}`);
  return handleResponse<PaginatedResponse<GeolocationSchema>>(response);
};

// Если понадобится получение по ID (хотя в API не указано)
// export const fetchGeolocationById = async (id: number): Promise<GeolocationSchema> => {
//   const response = await fetch(`${API_BASE_URL}/${id}`);
//   return handleResponse<GeolocationSchema>(response);
// }; 