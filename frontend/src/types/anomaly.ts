// Определяет структуру данных аномалии, возвращаемую API (/anomalies/)

export interface Anomaly {
  id: number;
  order_id: string;
  order_item_id?: number | null | undefined; 
  detector_type: string;
  anomaly_score?: number | null;
  details?: string | null; 
  details_dict?: Record<string, any> | null; 
  detection_date: string;
  parsed_details?: Record<string, any>; 
} 

// Добавляем универсальный интерфейс для пагинированных ответов API
export interface PaginatedResponse<T> {
  total: number;   // Общее количество записей
  items: T[];      // Массив элементов на текущей странице
}

// --- Types for Anomaly Context ---

export interface Customer {
  customer_id: string;
  customer_unique_id: string;
  customer_zip_code_prefix: number;
  customer_city: string;
  customer_state: string;
}

export interface Seller {
  seller_id: string;
  seller_zip_code_prefix: number;
  seller_city: string;
  seller_state: string;
}

export interface Product {
  product_id: string;
  product_category_name?: string | null;
  product_name_lenght?: number | null;
  product_description_lenght?: number | null;
  product_photos_qty?: number | null;
  product_weight_g?: number | null;
  product_length_cm?: number | null;
  product_height_cm?: number | null;
  product_width_cm?: number | null;
}

export interface OrderItem {
  order_item_id: number;
  price: number;
  freight_value: number;
  product: Product;
  seller: Seller;
  product_id?: string; 
  seller_id?: string; 
  shipping_limit_date?: string;
}

export interface OrderPayment {
  payment_sequential: number;
  payment_type: string;
  payment_installments: number;
  payment_value: number;
}

export interface OrderReview {
  review_id: string;
  review_score: number;
  review_comment_title?: string | null;
  review_comment_message?: string | null;
  review_creation_date: string; 
  review_answer_timestamp: string; 
}

export interface OrderDetails {
  order_id: string;
  order_status: string;
  order_purchase_timestamp: string; 
  order_approved_at?: string | null;
  order_delivered_carrier_date?: string | null;
  order_delivered_customer_date?: string | null;
  order_estimated_delivery_date: string; 
  customer: Customer;
  items: OrderItem[];
  payments: OrderPayment[];
  reviews: OrderReview[];
}

export interface AnomalyContextResponse {
  order_details: OrderDetails;
}

export interface AnomalyFilters {
  skip?: number;
  limit?: number;
  start_date?: string; 
  end_date?: string; 
  min_score?: number;
  max_score?: number;
  detector_type?: string;
  order_id?: string;
}