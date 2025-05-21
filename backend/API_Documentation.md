# Документация API для Фронтенд-Разработки

## Общая информация

*   **Базовый URL API**: `/api` (предполагается, что приложение развернуто так, что этот префикс доступен)
*   **Формат данных**: Все запросы и ответы используют JSON.
*   **Аутентификация**: В предоставленном коде не указана явная система аутентификации. Если она есть, её детали должны быть добавлены.
*   **Обработка ошибок**:
    *   `400 Bad Request`: Некорректный запрос (например, невалидные данные).
    *   `401 Unauthorized`: Требуется аутентификация (если применимо).
    *   `403 Forbidden`: Аутентификация пройдена, но нет прав на операцию.
    *   `404 Not Found`: Ресурс не найден.
    *   `422 Unprocessable Entity`: Ошибка валидации данных (FastAPI стандарт).
    *   `500 Internal Server Error`: Внутренняя ошибка сервера.
    *   `503 Service Unavailable`: Сервис временно недоступен (например, при инициализации MultilevelService).

## Общие Модели Данных (Pydantic Schemas)

Эти модели часто используются в ответах API.

### `PaginatedResponse[ItemSchema]`

Универсальная схема для ответов со списком элементов и пагинацией.

```json
{
  "total": 100, // Общее количество элементов
  "items": [
    // ... массив элементов типа ItemSchema ...
  ]
}
```

### `TaskCreationResponse`

Ответ при запуске фоновой задачи.

```json
{
  "task_id": "string", // ID созданной задачи
  "message": "string", // Сообщение о запуске
  "status_endpoint": "/api/tasks/task_status/{task_id}", // Эндпоинт для проверки статуса
  "initial_status": "string" // Начальный статус задачи (e.g., "pending", "processing")
}
```

### `TaskStatusResult` (Ответ от `/api/tasks/task_status/{task_id}`)

```json
{
  "status": "string", // (e.g., "pending", "processing", "completed", "failed", "completed_no_data", "completed_with_errors")
  "start_time": "datetime", // (ISO 8601)
  "end_time": "Optional[datetime]", // (ISO 8601)
  "details": "string", // Описание текущего состояния или результата
  "result": "Optional[Dict[str, Any]]", // Результат выполнения, если есть
  "error_type": "Optional[string]" // Тип ошибки, если status="failed"
}
```

---

## Эндпоинты

### 1. Корень (`/`)

*   **`GET /`**
    *   **Описание**: Проверка работоспособности API.
    *   **Ответ (`200 OK`)**:
        ```json
        {
          "message": "Welcome to the Anomaly Detection API",
          "environment": "development" // или другое значение из настроек
        }
        ```

### 2. Управление Аномалиями (`/api/anomalies`)

#### 2.1. Обучение моделей

*   **`POST /api/anomalies/train_model`**
    *   **Описание**: Запускает обучение модели обнаружения аномалий в фоновом режиме.
    *   **Тело Запроса (`TrainModelRequest`)**:
        ```json
        {
          "start_date": "Optional[datetime]", // (ISO 8601) По умолчанию "2000-01-01T00:00:00"
          "end_date": "Optional[datetime]",   // (ISO 8601)
          "detector_type": "string",          // По умолчанию "isolation_forest"
          "detector_config_payload": "Dict[str, Any]" // Конфигурация модели, например {"model_filename": "my_if_model.joblib", "n_estimators": 100}
        }
        ```
    *   **Ответ (`202 Accepted`)**: `TaskCreationResponse`
    *   **Примечания**:
        *   Задача выполняется в фоне.
        *   Для отслеживания статуса используйте эндпоинт `GET /api/tasks/task_status/{task_id}`.

#### 2.2. Детекция аномалий

*   **`POST /api/anomalies/detect_anomalies`**
    *   **Описание**: Запускает обнаружение аномалий с использованием указанной модели в фоновом режиме.
    *   **Тело Запроса (`DetectAnomaliesRequest`)**:
        ```json
        {
          "start_date": "Optional[datetime]", // (ISO 8601) По умолчанию "2000-01-01T00:00:00"
          "end_date": "Optional[datetime]",   // (ISO 8601)
          "detector_type": "string",          // По умолчанию "isolation_forest"
          "detector_config_payload": "Dict[str, Any]" // Конфигурация модели, например {"model_filename": "my_if_model.joblib"}
        }
        ```
    *   **Ответ (`202 Accepted`)**: `TaskCreationResponse`
    *   **Примечания**:
        *   Задача выполняется в фоне.
        *   Для отслеживания статуса используйте эндпоинт `GET /api/tasks/task_status/{task_id}`.

#### 2.3. CRUD операции с записями об аномалиях

*   **`GET /api/anomalies/`**
    *   **Описание**: Получает список записей об аномалиях с пагинацией и фильтрацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
        *   `start_date: Optional[datetime]` (ISO 8601)
        *   `end_date: Optional[datetime]` (ISO 8601)
        *   `min_score: Optional[float]`
        *   `max_score: Optional[float]`
        *   `detector_type: Optional[str]`
    *   **Ответ (`200 OK`)**: `PaginatedResponse[RootAnomalySchema]`
        *   `RootAnomalySchema`:
            ```json
            {
              "id": "int",
              "order_item_id": "int",
              "order_id": "string",
              "detection_date": "datetime", // (ISO 8601)
              "anomaly_score": "Optional[float]",
              "detector_type": "string",
              "details": "Optional[Dict[str, Any]]" // Распарсенный JSON из БД
            }
            ```

*   **`POST /api/anomalies/`**
    *   **Описание**: Создает новую запись об аномалии вручную.
    *   **Тело Запроса (`RootAnomalyCreateSchema`)**:
        ```json
        {
          "order_item_id": "int",
          "order_id": "string",
          "detection_date": "datetime", // (ISO 8601)
          "anomaly_score": "Optional[float]",
          "detector_type": "string",
          "details": "Optional[string]" // JSON-строка или Dict
        }
        ```
    *   **Ответ (`201 Created`)**: `RootAnomalySchema`

*   **`GET /api/anomalies/{anomaly_id}`**
    *   **Описание**: Получает одну аномалию по её ID.
    *   **Path Параметры**:
        *   `anomaly_id: int`
    *   **Ответ (`200 OK`)**: `RootAnomalySchema`
    *   **Ответ (`404 Not Found`)**: Если аномалия не найдена.

*   **`DELETE /api/anomalies/{anomaly_id}`**
    *   **Описание**: Удаляет аномалию по ID.
    *   **Path Параметры**:
        *   `anomaly_id: int`
    *   **Ответ (`200 OK`)**: `RootAnomalySchema` (удаленный объект)
    *   **Ответ (`404 Not Found`)**: Если аномалия не найдена.

*   **`DELETE /api/anomalies/`**
    *   **Описание**: Удаляет ВСЕ записи об аномалиях.
    *   **Ответ (`200 OK`)**:
        ```json
        {
          "message": "Все аномалии (N) успешно удалены."
        }
        ```

#### 2.4. Объяснение аномалий с помощью LLM

*   **`GET /api/anomalies/{anomaly_id}/explain-llm`**
    *   **Описание**: Получает объяснение для аномалии, сгенерированное LLM.
    *   **Path Параметры**:
        *   `anomaly_id: int`
    *   **Ответ (`200 OK`)**: `LLMExplanationResponse`
        ```json
        {
          "anomaly_id": "int",
          "original_details": "RootAnomalySchema", // Полные детали аномалии
          "llm_explanation": "string" // Текст объяснения от LLM
        }
        ```
    *   **Ответ (`404 Not Found`)**: Если аномалия не найдена.

### 3. Заказы (`/api/orders`)

*   **`GET /api/orders/`**
    *   **Описание**: Получает список заказов с детальной информацией, пагинацией и фильтрацией по дате.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
        *   `start_date: Optional[datetime]` (ISO 8601)
        *   `end_date: Optional[datetime]` (ISO 8601)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[OrderSchema]`
        *   `OrderSchema`: (определена в `backend/schemas.py`, включает `customer`, `items`, `payments`, `reviews`)
            ```json
            // Примерная структура OrderSchema
            {
              "order_id": "string",
              "customer_id": "string",
              "order_status": "string",
              "order_purchase_timestamp": "datetime",
              // ... другие поля заказа ...
              "customer": { /* CustomerSchema */ },
              "items": [ { /* OrderItemSchema */ } ],
              "payments": [ { /* OrderPaymentSchema */ } ],
              "reviews": [ { /* OrderReviewSchema */ } ]
            }
            ```

*   **`GET /api/orders/{order_id}`**
    *   **Описание**: Получает информацию о конкретном заказе по его ID.
    *   **Path Параметры**:
        *   `order_id: str`
    *   **Ответ (`200 OK`)**: `OrderSchema`
    *   **Ответ (`404 Not Found`)**: Если заказ не найден.

### 4. Настройки Приложения (`/api/settings`)

*   **`POST /api/settings/`**
    *   **Описание**: Создает новую настройку.
    *   **Тело Запроса (`SettingCreate`)**:
        ```json
        {
          "key": "string",
          "value": "string"
        }
        ```
    *   **Ответ (`201 Created`)**: `SettingSchema`
        ```json
        {
          "key": "string",
          "value": "string"
        }
        ```
    *   **Ответ (`400 Bad Request`)**: Если ключ уже существует.

*   **`GET /api/settings/`**
    *   **Описание**: Получает список всех настроек.
    *   **Query Параметры**:
        *   `skip: int` (default: 0)
        *   `limit: int` (default: 100)
    *   **Ответ (`200 OK`)**: `List[SettingSchema]`

*   **`GET /api/settings/{key}`**
    *   **Описание**: Получает настройку по ключу.
    *   **Path Параметры**:
        *   `key: str`
    *   **Ответ (`200 OK`)**: `SettingSchema`
    *   **Ответ (`404 Not Found`)**: Если настройка не найдена.

*   **`PUT /api/settings/{key}`**
    *   **Описание**: Обновляет значение настройки по ключу.
    *   **Path Параметры**:
        *   `key: str`
    *   **Тело Запроса (`SettingUpdate`)**:
        ```json
        {
          "value": "string"
        }
        ```
    *   **Ответ (`200 OK`)**: `SettingSchema`
    *   **Ответ (`404 Not Found`)**: Если настройка не найдена.

*   **`DELETE /api/settings/{key}`**
    *   **Описание**: Удаляет настройку по ключу.
    *   **Path Параметры**:
        *   `key: str`
    *   **Ответ (`200 OK`)**: `SettingSchema` (удаленный объект)
    *   **Ответ (`404 Not Found`)**: Если настройка не найдена.

### 5. Геолокация (`/api/geolocation`)

*   **`GET /api/geolocation/`**
    *   **Описание**: Получает список записей геолокации с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[GeolocationSchema]`
        *   `GeolocationSchema`: (определена в `backend/schemas.py`)
            ```json
            // Примерная структура GeolocationSchema
            {
              "id": "int", // Это поле добавлено в schemas.Geolocation, хотя в models.Geolocation его нет как отдельного id (там составной ключ) - уточнить! Вероятно, это автоинкрементный id, если таблица так создана. Если нет, то id не будет. В коде schemas.Geolocation есть "id: int", в models.py - нет.
              "geolocation_zip_code_prefix": "int",
              "geolocation_lat": "float",
              "geolocation_lng": "float",
              "geolocation_city": "string",
              "geolocation_state": "string"
            }
            ```
            **Примечание по GeolocationSchema**: В `models.py` у `Geolocation` составной первичный ключ, а не `id`. В `schemas.py` `Geolocation` имеет поле `id: int`. Это может быть несоответствием или `id` добавляется как автоинкрементное поле при создании таблицы, не являясь частью PK. Фронтенду следует ожидать `id` в ответе, если он есть в схеме.

### 6. Многоуровневая Детекция Аномалий (`/api/multilevel`)

*   **`GET /api/multilevel/status`**
    *   **Описание**: Получает статус всех детекторов в многоуровневой системе.
    *   **Ответ (`200 OK`)**: `MultilevelStatus`
        ```json
        // Примерная структура MultilevelStatus
        {
          "transaction_level": {
            "detector_name1": { /* DetectorStatus */ },
            "detector_name2": { /* DetectorStatus */ }
          },
          "behavior_level": { /* ... */ },
          "time_series_level": { /* ... */ }
        }
        // DetectorStatus:
        {
          "is_trained": "bool",
          "detector_type": "string",
          "model_filename": "Optional[str]",
          "expected_path": "Optional[str]",
          "exists": "Optional[bool]",
          "can_load": "Optional[bool]",
          "error_message": "Optional[str]",
          "params_from_config": "Optional[Dict[str, Any]]",
          "internal_params": "Optional[Dict[str, Any]]"
        }
        ```
    *   **Ответ (`503 Service Unavailable`)**: Если сервис не инициализирован.

*   **`GET /api/multilevel/config`**
    *   **Описание**: Получает текущую конфигурацию многоуровневой системы.
    *   **Ответ (`200 OK`)**: `MultilevelConfig`
        ```json
        // Примерная структура MultilevelConfig
        {
          "transaction_level": [ { "type": "string", "model_filename": "Optional[str]", "weight": "Optional[float]", "...другие параметры..." } ],
          "behavior_level": [ /* ... */ ],
          "time_series_level": [ /* ... */ ],
          "combination_weights": { "transaction": 0.4, "behavior": 0.4, "time_series": 0.2 }
        }
        ```
    *   **Ответ (`404 Not Found`)**: Если конфигурация не найдена.
    *   **Ответ (`503 Service Unavailable`)**: Если сервис не инициализирован.

*   **`POST /api/multilevel/config`**
    *   **Описание**: Обновляет конфигурацию многоуровневой системы.
    *   **Тело Запроса**: `MultilevelConfig`
    *   **Ответ (`200 OK`)**: `true`
    *   **Ответ (`500 Internal Server Error`)**: Если не удалось обновить.
    *   **Ответ (`503 Service Unavailable`)**: Если сервис не инициализирован.

*   **`POST /api/multilevel/train`**
    *   **Описание**: Запускает обучение всех моделей в многоуровневой системе.
    *   **Ответ (`202 Accepted`)**: `TaskCreationResponse`
    *   **Примечания**:
        *   Задача выполняется в фоне.
        *   Для отслеживания статуса используйте эндпоинт `GET /api/tasks/task_status/{task_id}`.
        *   Обучение происходит на всех данных из БД.
    *   **Ответ (`503 Service Unavailable`)**: Если сервис не инициализирован.

*   **`POST /api/multilevel/detect`**
    *   **Описание**: Запускает детекцию аномалий многоуровневой системой.
    *   **Тело Запроса (`DetectionParams`)**:
        ```json
        {
          "transaction_threshold": "float", // (default: 0.6)
          "behavior_threshold": "float",    // (default: 0.6)
          "time_series_threshold": "float", // (default: 0.6)
          "final_threshold": "float",       // (default: 0.5)
          "filter_period_days": "Optional[int]" // (default: 10000) Количество дней назад для фильтрации данных.
        }
        ```
    *   **Ответ (`202 Accepted`)**: `TaskCreationResponse`
    *   **Примечания**:
        *   Задача выполняется в фоне.
        *   Для отслеживания статуса используйте эндпоинт `GET /api/tasks/task_status/{task_id}`.
    *   **Ответ (`503 Service Unavailable`)**: Если сервис не инициализирован.

*   **`GET /api/multilevel/available-detectors`**
    *   **Описание**: Получает список доступных типов детекторов для каждого уровня.
    *   **Ответ (`200 OK`)**:
        ```json
        {
          "transaction_level": ["statistical", "isolation_forest", /* ... */],
          "behavior_level": ["seller_pricing_behavior", /* ... */],
          "time_series_level": ["seasonal_deviation", /* ... */]
        }
        ```

### 7. Управление Фоновыми Задачами (`/api/tasks`)

*   **`GET /api/tasks/task_status/{task_id}`**
    *   **Описание**: Получает статус фоновой задачи по её ID.
    *   **Path Параметры**:
        *   `task_id: str`
    *   **Ответ (`200 OK`)**: `TaskStatusResult` (см. "Общие Модели Данных")
    *   **Ответ (`404 Not Found`)**: Если задача с таким ID не найдена.

### 8. Продукты (`/api/products`)

*   **`GET /api/products/`**
    *   **Описание**: Получает список продуктов с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[ProductSchema]`
        *   `ProductSchema`: (определена в `backend/schemas.py`)
            ```json
            // Примерная структура ProductSchema
            {
              "product_id": "string",
              "product_category_name": "Optional[string]",
              // ... другие поля продукта ...
              "category_translation": { /* ProductCategoryNameTranslationSchema */ }
            }
            ```

*   **`GET /api/products/{product_id}`**
    *   **Описание**: Получает продукт по его ID.
    *   **Path Параметры**:
        *   `product_id: str` (в коде указано `item_id: str`, но для продукта это `product_id`)
    *   **Ответ (`200 OK`)**: `ProductSchema`
    *   **Ответ (`404 Not Found`)**: Если продукт не найден.

### 9. Покупатели (`/api/customers`)

*   **`GET /api/customers/`**
    *   **Описание**: Получает список покупателей с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[CustomerSchema]`
        *   `CustomerSchema`: (определена в `backend/schemas.py`)
            ```json
            {
              "customer_id": "string",
              "customer_unique_id": "string",
              "customer_zip_code_prefix": "int",
              "customer_city": "string",
              "customer_state": "string"
            }
            ```

*   **`GET /api/customers/{customer_id}`**
    *   **Описание**: Получает покупателя по его ID.
    *   **Path Параметры**:
        *   `customer_id: str` (в коде указано `item_id: str`, для покупателя это `customer_id`)
    *   **Ответ (`200 OK`)**: `CustomerSchema`
    *   **Ответ (`404 Not Found`)**: Если покупатель не найден.

### 10. Продавцы (`/api/sellers`)

*   **`GET /api/sellers/`**
    *   **Описание**: Получает список продавцов с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[SellerSchema]`
        *   `SellerSchema`: (определена в `backend/schemas.py`)
            ```json
            {
              "seller_id": "string",
              "seller_zip_code_prefix": "int",
              "seller_city": "string",
              "seller_state": "string"
            }
            ```

*   **`GET /api/sellers/{seller_id}`**
    *   **Описание**: Получает продавца по его ID.
    *   **Path Параметры**:
        *   `seller_id: str` (в коде указано `item_id: str`, для продавца это `seller_id`)
    *   **Ответ (`200 OK`)**: `SellerSchema`
    *   **Ответ (`404 Not Found`)**: Если продавец не найден.

### 11. Отзывы (`/api/reviews`)

*   **`GET /api/reviews/`**
    *   **Описание**: Получает список отзывов с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[OrderReviewSchema]`
        *   `OrderReviewSchema`: (определена в `backend/schemas.py`)
            ```json
            {
              "review_id": "string",
              "order_id": "string",
              "review_score": "int",
              // ... другие поля отзыва ...
            }
            ```

*   **`GET /api/reviews/{review_id}`**
    *   **Описание**: Получает отзыв по его ID.
    *   **Path Параметры**:
        *   `review_id: str` (в коде указано `item_id: str`, для отзыва это `review_id`)
    *   **Ответ (`200 OK`)**: `OrderReviewSchema`
    *   **Ответ (`404 Not Found`)**: Если отзыв не найден.

### 12. Переводы Категорий (`/api/translations`)

*   **`GET /api/translations/`**
    *   **Описание**: Получает список переводов названий категорий продуктов с пагинацией.
    *   **Query Параметры**:
        *   `skip: int` (default: 0, min: 0)
        *   `limit: int` (default: 100, min: 1, max: 1000)
    *   **Ответ (`200 OK`)**: `PaginatedResponse[ProductCategoryNameTranslationSchema]`
        *   `ProductCategoryNameTranslationSchema`: (определена в `backend/schemas.py`)
            ```json
            {
              "product_category_name": "string",
              "product_category_name_english": "string"
            }
            ```

*   **`GET /api/translations/{product_category_name}`**
    *   **Описание**: Получает перевод категории по её оригинальному названию.
    *   **Path Параметры**:
        *   `product_category_name: str` (в коде указано `item_id: str`, для перевода это `product_category_name`)
    *   **Ответ (`200 OK`)**: `ProductCategoryNameTranslationSchema`
    *   **Ответ (`404 Not Found`)**: Если перевод не найден.