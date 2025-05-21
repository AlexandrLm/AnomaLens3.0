# Разработка интеллектуальной системы обнаружения аномалий в данных интернет-магазинов

## Обзор проекта

Проект AnomaLens 3.0 представляет собой интеллектуальную систему для обнаружения аномалий в данных интернет-магазинов. Система использует различные методы машинного обучения и статистического анализа для выявления необычных паттернов в транзакциях, поведении пользователей и характеристиках товаров, которые могут указывать на мошенничество, технические проблемы или другие аномальные ситуации.

### Цель проекта

Основная цель системы – разработать и реализовать комплексный подход к обнаружению аномалий в данных электронной коммерции, включающий:
* Механизмы для обучения различных моделей обнаружения аномалий
* Механизмы для детекции аномалий с использованием обученных моделей
* Многоуровневую систему детекции, комбинирующую результаты различных детекторов
* Возможность сохранения и управления записями об обнаруженных аномалиях
* Генерацию человекочитаемых объяснений для аномалий с помощью LLM (Ollama)
* Управление фоновыми задачами для длительных операций (обучение, детекция)

### Используемые данные

Система разработана для работы с данными бразильского интернет-магазина Olist. Датасет Olist содержит информацию о:
* Заказах и товарных позициях
* Покупателях и продавцах
* Ценах и стоимости доставки
* Платежах и отзывах
* Географических данных
* Категориях товаров

Эти данные используются как для обучения моделей, так и для обнаружения аномалий в новых транзакциях.

## 1. Обзор проекта

**Цель:** Разработать и реализовать интеллектуальную систему для обнаружения различных типов аномалий в данных электронной коммерции на примере датасета Olist. Система должна включать:
*   Бэкенд API на FastAPI.
*   Механизмы для обучения моделей обнаружения аномалий.
*   Механизмы для детекции аномалий с использованием обученных моделей.
*   Многоуровневую систему детекции, комбинирующую результаты различных детекторов.
*   Возможность сохранения и управления записями об обнаруженных аномалиях.
*   Генерацию человекочитаемых объяснений для аномалий с помощью LLM (Ollama).
*   Управление фоновыми задачами для длительных операций (обучение, детекция).

**Ключевые технологии:**
*   **Бэкенд:** Python, FastAPI
*   **База данных:** SQLAlchemy (ORM), SQLite (для разработки/тестирования), PostgreSQL (потенциально для продакшена)
*   **Моделирование данных (API):** Pydantic
*   **Машинное обучение:**
    *   Scikit-learn (IsolationForest, StandardScaler)
    *   PyTorch (для Autoencoder, VAE)
    *   NetworkX (для графовых детекторов)
    *   SHAP (для объяснения моделей)
    *   Pandas, NumPy (для обработки данных)
*   **Сериализация моделей:** Joblib
*   **LLM для объяснений:** Ollama (через HTTP API)
*   **Логирование:** Стандартный модуль `logging` Python.

## 2. Архитектура системы

Система состоит из нескольких основных компонентов:

1.  **FastAPI Backend (`main.py`, `api/`):**
    *   Предоставляет REST API для взаимодействия с системой.
    *   Обрабатывает HTTP-запросы, валидирует данные.
    *   Использует маршрутизаторы (routers) для организации эндпоинтов.
    *   Инициализирует подключение к БД и создает таблицы при старте (`lifespan` функция).

2.  **Слой данных (`database.py`, `models.py`, `schemas.py`, `crud.py`):**
    *   **`database.py`**: Настройка SQLAlchemy engine, `SessionLocal` для сессий БД, функция `get_db` для dependency injection сессии.
    *   **`models.py`**: Определения SQLAlchemy моделей, соответствующих таблицам Olist (Customers, Orders, Products и т.д.) и кастомным таблицам (`Anomalies`, `Settings`).
    *   **`schemas.py`**: Pydantic схемы для валидации данных API-запросов и формирования ответов. Включает схемы для моделей БД, а также вспомогательные схемы (например, `PaginatedResponse`, `TaskCreationResponse`). Существует также `backend/ml_service/schemas.py` для специфичных ML-ответов.
    *   **`crud.py`**: Функции для выполнения операций CRUD (Create, Read, Update, Delete) с БД. Содержит как общие функции (`get_items`, `get_item_by_id`), так и специализированные (например, `get_orders_with_details` с предзагрузкой связанных данных, CRUD для аномалий и настроек).

3.  **Сервис Машинного Обучения (`backend/ml_service/`):**
    *   Содержит всю логику, связанную с обнаружением аномалий.
    *   **`common.py`**: Утилиты для загрузки данных из БД (`load_data_from_db`) и инженерии признаков (`engineer_features`).
    *   **`detector.py`**: Базовый абстрактный класс `AnomalyDetector` и реализации конкретных детекторов (`StatisticalDetector`, `IsolationForestDetector`, `AutoencoderDetector`).
    *   **`vae_detector.py`**: Реализация `VAEDetector`.
    *   **`graph_detector.py`**: Реализация `GraphAnomalyDetector`.
    *   **Специализированные детекторы (`transaction_detectors.py`, `behavior_detectors.py`, `time_series_detectors.py`):** Детекторы, ориентированные на конкретные типы аномалий или уровни анализа.
    *   **`detector_factory.py`**: Фабрика для создания экземпляров детекторов по их типу и параметрам.
    *   **`multilevel_detector.py`**: Реализация `MultilevelAnomalyDetector`, который оркеструет детекторы на транзакционном, поведенческом и временном уровнях.
    *   **`multilevel_service.py`**: Сервисный слой для `MultilevelAnomalyDetector`, управляющий его конфигурацией, обучением и детекцией (включая асинхронные задачи).
    *   **`llm_explainer.py`**: Класс `LLMExplainer` для взаимодействия с Ollama и генерации объяснений.
    *   **`protocols.py`**: Определения протоколов для типов.
    *   **`metrics.py`, `monitoring.py`**: Модули для оценки и мониторинга моделей (их использование в текущей API не очевидно, но они часть ML-сервиса).

4.  **Менеджер Задач (`backend/services/task_manager.py`):**
    *   Отвечает за управление фоновыми задачами (например, обучение моделей, детекция аномалий).
    *   Использует `Task` класс для отслеживания статуса задачи (`pending`, `running`, `completed`, `failed` и т.д.), времени выполнения, деталей и результата.
    *   `TaskManager` управляет созданием и обновлением статусов задач.
    *   Интегрируется с `BackgroundTasks` FastAPI для выполнения функций в фоне.

5.  **LLM Сервис (Внешний - Ollama):**
    *   Используется для генерации человекочитаемых объяснений аномалий.
    *   Взаимодействие происходит через HTTP API (`ml_service/llm_explainer.py`).

## 3. Модель Данных (Olist + Кастомные)

Используется датасет Olist, для которого определены следующие SQLAlchemy модели в `models.py`:

*   `Customer`: Покупатели.
*   `Geolocation`: Геолокационные данные.
*   `Order`: Заказы.
*   `OrderItem`: Товарные позиции в заказе (ключевая таблица для многих анализов).
*   `OrderPayment`: Платежи по заказам.
*   `OrderReview`: Отзывы о заказах.
*   `Product`: Товары.
*   `Seller`: Продавцы.
*   `ProductCategoryNameTranslation`: Переводы названий категорий товаров.

**Кастомные модели:**

*   **`Anomaly` (`models.py`):**
    *   `id`: PK, Integer.
    *   `order_id`: String, ID заказа, где обнаружена аномалия (или связана с ней).
    *   `order_item_id`: Integer, ID товарной позиции (если аномалия связана с конкретным товаром в заказе).
    *   `detection_date`: DateTime, дата обнаружения.
    *   `anomaly_score`: Float, оценка аномальности.
    *   `detector_type`: String, тип детектора, обнаружившего аномалию (например, `isolation_forest`, `statistical`, `multilevel`).
    *   `details`: String, JSON-строка с дополнительной информацией об аномалии (например, значения признаков, SHAP values, объяснения от компонентных детекторов для `multilevel`).
*   **`Setting` (`models.py`):**
    *   `key`: String, PK, ключ настройки.
    *   `value`: String, значение настройки.

**Схемы Pydantic (`schemas.py`):**
Для каждой модели SQLAlchemy определены соответствующие Pydantic схемы (Base, Create, Read) для валидации и сериализации данных в API.
*   `RootAnomalySchema` и `RootAnomalyCreateSchema` в `api/anomalies.py` используются для CRUD операций с аномалиями. `RootAnomalySchema` при чтении автоматически парсит поле `details` (JSON-строку) в словарь.
*   `OrderSchema` включает вложенные схемы для `Customer`, `OrderItem`, `OrderPayment`, `OrderReview` для предоставления полной информации о заказе.
*   Общие схемы: `PaginatedResponse` для пагинированных ответов, `TaskCreationResponse` и `TaskStatusResult` для управления задачами.

## 4. Ключевая Логика и Функциональность

### 4.1. Взаимодействие с Базой Данных (`crud.py`)

*   **Получение Заказов:**
    *   `get_orders_with_details`: Получает список заказов с предзагрузкой (`joinedload`) всех связанных данных (товары, платежи, отзывы, покупатель, категория товара, продавец). Поддерживает фильтрацию по дате и пагинацию.
    *   `get_order_by_id_with_details`: Получает один заказ по ID со всеми связанными данными.
    *   `get_order_count`: Получает общее количество заказов с фильтрацией по дате.
*   **Общие CRUD-операции:**
    *   `get_items`: Получение списка записей из любой таблицы с пагинацией.
    *   `get_item_count`: Получение количества записей в таблице.
    *   `get_item_by_id`: Получение одной записи по ID (работает для моделей с одним первичным ключом).
*   **Операции с Аномалиями:**
    *   `create_anomaly`: Создает новую запись об аномалии. Поле `details` (словарь/список) конвертируется в JSON-строку.
    *   `get_anomalies`: Получает список аномалий с фильтрацией по дате, скору, типу детектора и пагинацией.
    *   `get_anomaly`: Получает одну аномалию по ID.
    *   `delete_anomaly`: Удаляет одну аномалию по ID.
    *   `delete_all_anomalies`: Удаляет ВСЕ аномалии.
*   **Операции с Настройками (`Setting`):**
    *   Стандартные CRUD-операции: `create_setting`, `get_settings`, `get_setting_by_key`, `update_setting`, `delete_setting`.

### 4.2. Ядро Обнаружения Аномалий (`ml_service/`)

#### 4.2.1. Загрузка и Подготовка Данных (`ml_service/common.py`)

*   `load_data_from_db`: Загружает данные из БД. Ключевой является таблица `OrderItem`.
    *   Принимает `start_date`, `end_date`.
    *   Флаг `load_associations`: если `True`, подгружает данные из связанных таблиц (`Order`, `Product`, `Customer`, `ProductCategoryNameTranslation`, `Seller`, `OrderPayment`, `OrderReview`) через JOIN.
    *   Возвращает Pandas DataFrame.
*   `engineer_features`: Добавляет производные признаки в DataFrame.
    *   `price_deviation_from_category_mean`: Отклонение цены товара от средней по его категории.
    *   `freight_to_price_ratio`: Отношение стоимости доставки к цене товара.

#### 4.2.2. Детекторы Аномалий (`ml_service/detector.py`, и др.)

*   **Базовый Класс `AnomalyDetector` (`ml_service/detector.py`):**
    *   Абстрактный класс, определяющий интерфейс для всех детекторов.
    *   Атрибуты: `model_name`, `model` (обученная модель/статистики), `scaler` (для стандартизации), `is_trained` (флаг).
    *   Параметры для нормализации скоров: `min_score_`, `max_score_`.
    *   Методы:
        *   `preprocess(data, fit_scaler)`: Абстрактный, для подготовки данных. `fit_scaler=True` означает обучение скейлера.
        *   `train(data)`: Абстрактный, для обучения модели/статистик. Включает обучение `scaler` и `normalizer`.
        *   `detect(data)`: Абстрактный, для обнаружения аномалий. Возвращает DataFrame с `anomaly_score` (сырой) и `is_anomaly`. Может добавлять `anomaly_score_normalized`.
        *   `save_model(path)`, `load_model(path)`: Сохранение/загрузка состояния детектора (модель, скейлер, параметры нормализации) с использованием `joblib`. Используют вспомогательные `_get_attributes_to_save()` и `_load_additional_attributes()` для специфичных для детектора данных.
        *   `fit_normalizer(scores)`, `normalize_score(scores)`: Обучение и применение Min-Max нормализации для скоров аномальности (чтобы привести их к диапазону [0, 1]).
        *   `get_shap_explainer()`, `get_explanation_details()`: Для интеграции с SHAP и предоставления деталей для LLM.
        *   `_reset_state()`: Сброс состояния детектора.

*   **Конкретные Реализации Детекторов:**
    *   **`StatisticalDetector` (`ml_service/detector.py`):**
        *   Анализирует один признак (`feature`) на основе Z-оценки.
        *   `threshold`: порог Z-оценки.
        *   `model` хранит `mean` и `std_dev` признака, вычисленные на этапе `train`.
    *   **`IsolationForestDetector` (`ml_service/detector.py`):**
        *   Использует `sklearn.ensemble.IsolationForest`.
        *   `features`: список признаков для модели.
        *   `preprocess` стандартизирует данные с помощью `StandardScaler`.
        *   `train` обучает `IsolationForest` и `StandardScaler`. Инициализирует SHAP `TreeExplainer`.
        *   `detect` использует `model.score_samples()` (инвертированные, т.к. чем выше, тем нормальнее) для получения сырых скоров.
        *   `get_shap_explanations` использует `self.shap_explainer`.
    *   **`AutoencoderDetector` (`ml_service/detector.py`):**
        *   Использует нейросетевой автоэнкодер на PyTorch (`Autoencoder` класс).
        *   Ошибка реконструкции (MSE) используется как скор аномальности.
        *   `train` обучает автоэнкодер, `StandardScaler` и определяет `threshold_` (порог ошибки реконструкции, например, 99-й перцентиль на обучающей выборке). Инициализирует SHAP `GradientExplainer` с использованием `_AEModelWrapperForSHAP`.
        *   `preprocess` также использует `StandardScaler`.
    *   **`VAEDetector` (`ml_service/vae_detector.py`):**
        *   Аналогичен `AutoencoderDetector`, но использует вариационный автоэнкодер (`VAE` класс).
        *   Функция потерь включает ошибку реконструкции и KL-дивергенцию (`kld_weight`).
        *   Ошибка реконструкции используется как скор.
        *   Также интегрирован с SHAP через `_VAEModelWrapperForSHAP` и `GradientExplainer`.
    *   **`GraphAnomalyDetector` (`ml_service/graph_detector.py`):**
        *   Строит граф связей (Order, OrderItem, Product, Seller, Customer) с помощью `networkx`.
        *   Вычисляет графовые метрики для узлов `OrderItem` (или связанных с ними сущностей):
            *   `num_states`: Географический разброс продавцов в заказе.
            *   `num_categories`: Разнообразие категорий товаров в заказе.
            *   `seller_degree`: Степень узла продавца.
        *   `train` вычисляет min/max для каждой сырой метрики на обучающих данных для последующей нормализации этих метрик. Затем обучает общий нормализатор на взвешенной сумме нормализованных метрик.
        *   `detect` вычисляет сырые метрики, нормализует их индивидуально, затем считает взвешенную сумму этих нормализованных метрик как `anomaly_score`.
    *   **Специализированные детекторы:**
        *   **Транзакционные (`ml_service/transaction_detectors.py`):**
            *   `PriceFreightRatioDetector`: Наследуется от `StatisticalDetector`, анализирует признак `freight_to_price_ratio`.
            *   `CategoryPriceOutlierDetector`: Самостоятельный, анализирует цены относительно средних в категории. Хранит статистики по категориям.
            *   `MultiFeatureIsolationForestDetector`: Наследуется от `IsolationForestDetector`, использует набор транзакционных признаков и генерирует производные (`freight_to_price_ratio`, `price_per_gram`).
            *   `TransactionVAEDetector`: Наследуется от `VAEDetector`, использует транзакционные признаки и генерирует производные.
        *   **Поведенческие (`ml_service/behavior_detectors.py`):**
            *   `SellerPricingBehaviorDetector`: Анализирует стабильность цен продавца (волатильность, диапазон, соотношение доставки к цене). Хранит глобальные статистики.
            *   `SellerCategoryMixDetector`: Анализирует разнообразие категорий товаров у продавца.
            *   `BehaviorIsolationForestDetector`: Наследуется от `IsolationForestDetector`, работает на агрегированных по продавцам поведенческих признаках.
        *   **Временных рядов (`ml_service/time_series_detectors.py`):**
            *   `SeasonalDeviationDetector`: Анализирует отклонения от сезонных паттернов (дневных, недельных, месячных).
            *   `MovingAverageVolatilityDetector`: Анализирует волатильность вокруг скользящей средней.
            *   `CumulativeSumDetector (CUSUM)`: Обнаруживает устойчивые изменения (дрифт) во временном ряду.

#### 4.2.3. Фабрика Детекторов (`ml_service/detector_factory.py`)

*   Класс `DetectorFactory` используется для создания экземпляров детекторов.
*   `_detector_classes`: Словарь, отображающий строковый тип детектора на его класс.
*   `create_detector(detector_type, **params)`: Создает экземпляр детектора.
*   `create_and_load_detector(detector_type, model_path, **params)`: Создает детектор и загружает его состояние из файла.
*   `create_detectors_from_config(configs, model_base_path)`: Создает словарь детекторов на основе списка конфигураций.
*   `_generate_detector_name(detector_type, config)`: Генерирует уникальное имя для экземпляра детектора.

#### 4.2.4. Процесс Обучения Модели (Фоновая Задача) (`api/anomalies.py::train_model_task`)

1.  **Инициализация:**
    *   Получение параметров: `start_date`, `end_date`, `detector_type`, `detector_config_payload`.
    *   Функция `_initialize_detector` (в `api/anomalies.py`):
        *   Определяет путь к файлу модели (`model_filename` из `detector_config_payload`).
        *   Использует `DetectorFactory.get_detector_class` для получения класса детектора.
        *   Создает экземпляр детектора с параметрами из `detector_config_payload`.
        *   Возвращает экземпляр, путь к модели и список необходимых базовых признаков.
2.  **Загрузка данных:**
    *   Вызов `common.load_data_from_db` с `start_date`, `end_date` и `load_associations=True`.
3.  **Инженерия признаков:**
    *   Вызов `common.engineer_features` для добавления производных признаков. Пропускается, если детектор использует только базовые признаки, которые уже есть.
4.  **Обучение детектора:**
    *   Вызов метода `detector_instance.train(items_df)`.
    *   Внутри `train` происходит обучение модели/статистик, обучение `StandardScaler` (если используется) и обучение `normalizer` для скоров (`fit_normalizer`).
    *   Для детекторов, поддерживающих SHAP (например, `IsolationForestDetector`, `AutoencoderDetector`, `VAEDetector`), инициализируется SHAP explainer.
5.  **Сохранение модели:**
    *   Если `detector_instance.is_trained` == `True`, вызывается `detector_instance.save_model(model_path)`.
6.  **Обновление статуса задачи:**
    *   Используется `task_manager.update_task_status` для информирования о прогрессе и результате.

#### 4.2.5. Процесс Детекции Аномалий (Фоновая Задача) (`api/anomalies.py::detect_anomalies_task`)

1.  **Инициализация и загрузка модели:**
    *   Аналогично `train_model_task`, используется `_initialize_detector`, но с `load_trained_model_if_path_exists=True`. Это вызовет `detector_instance.load_model()`.
    *   Если модель не обучена (`is_trained` == `False`), задача завершается с ошибкой.
2.  **Загрузка данных:**
    *   `common.load_data_from_db` для периода детекции.
3.  **Инженерия признаков:**
    *   `common.engineer_features`.
4.  **Детекция:**
    *   Вызов `detector_instance.detect(items_df)`. Этот метод возвращает DataFrame с колонками `anomaly_score` (сырой скор) и `is_anomaly` (булево значение). Некоторые детекторы могут также добавлять `anomaly_score_normalized`.
5.  **Обработка результатов:**
    *   Фильтрация аномалий (`results_df['is_anomaly'] == True`).
    *   Для каждой аномалии:
        *   **Генерация SHAP объяснений (если применимо):** Если детектор имеет метод `get_shap_explanations` (или более общий `get_explanation_details`), он вызывается для получения SHAP values.
        *   Формирование поля `details` для записи в БД:
            *   Включает значения признаков аномальной записи.
            *   Включает SHAP values (если есть) в `details['shap_values']`.
            *   Конвертируется в JSON-строку.
        *   Используется `anomaly_score_normalized` если есть, иначе `anomaly_score`.
        *   Создание объекта `RootAnomalyCreateSchema`.
        *   Сохранение в БД через `crud.create_anomaly`.
6.  **Обновление статуса задачи:**
    *   Информирование о количестве обнаруженных и сохраненных аномалий.

### 4.3. Многоуровневая Система Детекции (`ml_service/multilevel_detector.py`, `ml_service/multilevel_service.py`)

*   **`MultilevelAnomalyDetector` (`multilevel_detector.py`):**
    *   Инициализируется конфигурацией, которая определяет детекторы для каждого уровня (транзакционный, поведенческий, временной), их веса и методы комбинации скоров на уровне.
    *   `_initialize_detectors`: Создает и загружает все сконфигурированные детекторы с помощью `DetectorFactory`.
    *   **Детекция по уровням:**
        *   `_detect_transaction_level(data)`:
            *   Запускает все транзакционные детекторы на входных данных (`data`).
            *   Использует `_get_normalized_scores_from_detectors` для получения нормализованных скоров от каждого детектора.
            *   Комбинирует скоры с помощью `_combine_scores_for_single_level` (например, `weighted_average`, `max`).
            *   Возвращает DataFrame с `transaction_score` и индивидуальными скорами детекторов.
        *   `_detect_behavior_level(data)`:
            *   Вызывает `_prepare_behavior_data` для агрегации данных по продавцам/покупателям (вычисляет статистики, разнообразие категорий и т.д.).
            *   `GraphAnomalyDetector` получает исходные данные, остальные поведенческие детекторы - агрегированные.
            *   Аналогично транзакционному уровню, получает и комбинирует скоры.
            *   Возвращает DataFrame (агрегированный по ключу поведения, например `seller_id`) с `behavior_score`.
        *   `_detect_time_series_level(data)`:
            *   Вызывает `_prepare_time_series_data` для агрегации данных по временным точкам (например, по дням).
            *   Аналогично, получает и комбинирует скоры.
            *   Возвращает DataFrame (агрегированный по времени) с `time_series_score`.
    *   `detect(...)`:
        1.  Вызывает методы детекции для каждого уровня.
        2.  Мержит результаты уровней к исходному DataFrame (например, `behavior_score` по `seller_id`, `time_series_score` по дате).
        3.  Вычисляет итоговый `multilevel_score` путем взвешенного суммирования скоров уровней (`level_weights`).
        4.  Определяет итоговый флаг `is_anomaly` на основе `final_threshold`.
        5.  Для аномалий генерирует `detailed_explanations_json`, собирая объяснения от значимых детекторов каждого уровня с помощью их метода `get_explanation_details`.
        6.  Переименовывает колонки со скорами для совместимости при сохранении.
*   **`MultilevelDetectorService` (`multilevel_service.py`):**
    *   Сервисный слой для `MultilevelAnomalyDetector`.
    *   При инициализации загружает конфигурацию многоуровневого детектора из `settings.common.multilevel_detector_default_config` (которая, в свою очередь, читается из `config.yaml`).
    *   `train_task_wrapper`, `detect_async_task_wrapper`: Обертки для запуска обучения и детекции многоуровневой системы как фоновых задач через `task_manager`.
    *   `train()`: Метод для обучения всех детекторов во всех уровнях. Загружает данные, проводит инженерию признаков, затем итерируется по детекторам каждого уровня, вызывает их `train()` и `save_model()`.
    *   `detect()`: Основной метод детекции. Загружает данные, передает их в `self.multilevel_detector.detect()`. Затем обрабатывает результат (`anomalies_df`) и сохраняет обнаруженные "multilevel" аномалии в БД через `crud.create_anomaly`. Поле `details` для этих аномалий содержит скоры уровней, пороги и `detailed_explanations_json` от `MultilevelAnomalyDetector`.
    *   `update_config()`: Позволяет обновить конфигурацию в памяти (не сохраняет в YAML).
    *   `get_config()`: Возвращает текущую конфигурацию.
    *   `get_detector_status()`: Собирает детальный статус по каждому детектору на каждом уровне (обучен ли, путь к модели, существует ли файл и т.д.).

### 4.4. LLM Объяснения (`ml_service/llm_explainer.py`, `api/anomalies.py`)

*   **`LLMExplainer` (`ml_service/llm_explainer.py`):**
    *   Инициализируется URL Ollama и именем модели.
    *   `generate_explanation(anomaly_data, shap_values, detector_specific_info, anomaly_score)`:
        *   Формирует промпт для LLM на основе шаблона. Промпт включает:
            *   Детали самой аномалии (из `anomaly_data`, обычно это `details` из БД).
            *   Имя детектора, тип, скор.
            *   Техническое объяснение, сформированное из SHAP values и `detector_specific_info` (если есть).
        *   Отправляет запрос к Ollama API (`/api/generate`).
        *   Возвращает текстовое объяснение от LLM.
        *   Обрабатывает специфичный формат ответа от некоторых LLM (удаляет блок `<think>...</think>`).
*   **Эндпоинт `GET /api/anomalies/{anomaly_id}/explain-llm` (`api/anomalies.py`):**
    *   Получает аномалию из БД по `anomaly_id` (`crud.get_anomaly`).
    *   Извлекает из поля `db_anomaly.details` (распарсенного JSON) `shap_values` и `detector_specific_info`.
    *   Вызывает `llm_explainer.generate_explanation()`.
    *   Возвращает `LLMExplanationResponse` (ID аномалии, оригинальные детали аномалии, текст объяснения).

### 4.5. Управление Фоновыми Задачами (`services/task_manager.py`)

*   **`Task` класс:**
    *   Атрибуты: `task_id`, `description`, `status` (enum `TaskStatus`), `created_at`, `updated_at`, `details`, `result`, `error_type`.
    *   Метод `update_status` для изменения состояния задачи.
*   **`TaskManager` класс:**
    *   `tasks`: Словарь для хранения экземпляров `Task`.
    *   `_lock`: `threading.Lock` для потокобезопасного доступа к `tasks`.
    *   `create_task(description)`: Генерирует UUID, создает `Task`, добавляет в `tasks`.
    *   `update_task_status(task_id, status, details, result, error_type)`: Обновляет статус существующей задачи.
    *   `get_task_status(task_id)`: Возвращает объект `Task`.
    *   `run_task_in_background(background_tasks, task_id, task_func, *args, **kwargs)`:
        *   Обновляет статус задачи на `TaskStatus.RUNNING`.
        *   Использует `background_tasks.add_task` (от FastAPI) для добавления `task_func` в очередь выполнения.
*   **Роутер (`task_manager.router`):**
    *   Эндпоинт `GET /api/tasks/task_status/{task_id}`: Вызывает `task_manager.get_task_status()` и возвращает Pydantic модель `TaskStatus` (которая является представлением `Task` класса).

### 4.6. Конфигурация (`config/config.py`, `config.yaml`)

*   Предполагается, что `config.py` загружает настройки из `config.yaml` (или переменных окружения) в Pydantic модели.
*   `AppSettings` - корневая модель настроек.
*   Настройки разделены на секции:
    *   `common`: CORS origins, директория логов, URL Ollama, имя LLM модели, базовый путь к ML моделям.
    *   `api`: Название, описание, версия API.
    *   `database`: `DATABASE_URL`.
    *   `ml_service`:
        *   `multilevel_detector_default_config`: Детальная конфигурация для `MultilevelAnomalyDetector`, включая списки детекторов для каждого уровня с их параметрами (тип, имя файла модели, вес, специфичные параметры детектора) и методы комбинации скоров на уровне.

### 4.7. Общие CRUD Эндпоинты (`api/utils.py`)

*   `create_generic_router(model, response_schema, prefix, tags)`:
    *   Фабричная функция, создающая APIRouter для указанной SQLAlchemy `model` и Pydantic `response_schema`.
    *   Автоматически создает эндпоинты:
        *   `GET /` (список с пагинацией, использует `crud.get_items`, `crud.get_item_count`).
        *   `GET /{primary_key}` (одна запись по ID, использует `crud.get_item_by_id`).
    *   Используется в `main.py` для создания роутеров для `Product`, `Customer`, `Seller`, `OrderReview`, `ProductCategoryNameTranslation`.

## 5. Примеры Рабочих Процессов

1.  **Обучение одного детектора (например, Isolation Forest):**
    1.  Пользователь отправляет POST запрос на `/api/anomalies/train_model` с `detector_type="isolation_forest"` и `detector_config_payload={"model_filename": "if_model.joblib", "n_estimators": 150}`.
    2.  `api.anomalies.train_model_endpoint` получает запрос.
    3.  `task_manager.create_task` создает новую задачу.
    4.  `task_manager.run_task_in_background` добавляет `api.anomalies.train_model_task` в фоновые задачи FastAPI.
    5.  `train_model_task` выполняется:
        *   `_initialize_detector` создает экземпляр `IsolationForestDetector`.
        *   `common.load_data_from_db` загружает данные.
        *   `common.engineer_features` добавляет признаки.
        *   `detector.train()` обучает модель IF, StandardScaler, Normalizer и SHAP explainer.
        *   `detector.save_model()` сохраняет состояние в `models/if_model.joblib`.
        *   `task_manager.update_task_status` обновляет статус задачи на "completed".
    6.  Пользователь может проверять статус по `GET /api/tasks/task_status/{task_id}`.

2.  **Детекция аномалий многоуровневой системой:**
    1.  Пользователь отправляет POST запрос на `/api/multilevel/detect` с параметрами порогов.
    2.  `api.multilevel.detect_anomalies_multilevel_endpoint` получает запрос.
    3.  `task_manager.create_task` создает задачу.
    4.  `task_manager.run_task_in_background` запускает `multilevel_service.detect_async_task_wrapper`.
    5.  `detect_async_task_wrapper` вызывает `multilevel_service.detect_async`, который в свою очередь выполняет `multilevel_service.detect` в отдельном потоке.
    6.  `multilevel_service.detect`:
        *   Загружает данные (`load_data_from_db`).
        *   Инженерия признаков (`engineer_features`).
        *   Вызывает `self.multilevel_detector.detect()`.
            *   `MultilevelAnomalyDetector` последовательно вызывает `_detect_transaction_level`, `_detect_behavior_level`, `_detect_time_series_level`.
            *   Каждый из этих методов запускает соответствующие детекторы, получает их нормализованные скоры и комбинирует их.
            *   Итоговые скоры уровней комбинируются в `multilevel_score`.
            *   Генерируются `detailed_explanations_json`.
        *   Сохраняет обнаруженные аномалии "multilevel" в БД через `crud.create_anomaly`.
    7.  `task_manager.update_task_status` обновляет статус.

3.  **Получение LLM объяснения для аномалии:**
    1.  Пользователь отправляет GET запрос на `/api/anomalies/{anomaly_id}/explain-llm`.
    2.  `api.anomalies.get_anomaly_llm_explanation` получает запрос.
    3.  `crud.get_anomaly` извлекает аномалию из БД.
    4.  Из поля `details` аномалии извлекаются `shap_values` и `detector_specific_info`.
    5.  Создается экземпляр `LLMExplainer`.
    6.  `llm_explainer.generate_explanation()` формирует промпт и отправляет запрос к Ollama.
    7.  Ответ от Ollama возвращается пользователю.

## 6. Важные моменты и детали реализации

*   **Предзагрузка в SQLAlchemy (`joinedload`):** Активно используется в `crud.py` для `get_orders_with_details` и `get_order_by_id_with_details` для эффективной загрузки связанных объектов и избежания проблемы N+1 запросов.
*   **Обработка JSON в `Anomaly.details`:** При создании аномалии (`crud.create_anomaly`) словарь `details` сериализуется в JSON-строку. При чтении через `RootAnomalySchema` (в `api/anomalies.py`) JSON-строка автоматически десериализуется обратно в словарь с помощью `@field_validator`.
*   **Сохранение и Загрузка Детекторов:** Механизм `save_model`/`load_model` в `AnomalyDetector` и его переопределениях (`_get_attributes_to_save`, `_load_additional_attributes`) позволяет сохранять не только саму модель (например, sklearn или PyTorch state_dict), но и скейлер, параметры нормализации скоров и другие специфичные атрибуты. SHAP эксплейнеры обычно не сериализуются, а пересоздаются при загрузке или во время обучения.
*   **Нормализация Скоров:** Все детекторы обучают `normalizer` (`fit_normalizer`) на своих сырых скорах на обучающей выборке, чтобы привести их к диапазону [0,1] (`normalize_score`). Это важно для консистентного комбинирования скоров в многоуровневой системе.
*   **SHAP Интеграция:**
    *   `IsolationForestDetector` использует `shap.TreeExplainer`.
    *   `AutoencoderDetector` и `VAEDetector` используют обертки (`_AEModelWrapperForSHAP`, `_VAEModelWrapperForSHAP`) для вычисления ошибки реконструкции и `shap.GradientExplainer` с фоновыми данными. Фоновые данные (`background_data_for_shap`) сохраняются вместе с моделью.
*   **Конфигурация `MultilevelDetectorService`:** Загружается из `settings.ml_service.multilevel_detector_default_config`, что позволяет гибко настраивать состав и параметры детекторов на каждом уровне через `config.yaml`.
*   **Именование детекторов в `MultilevelDetectorService`:** Имена генерируются `DetectorFactory._generate_detector_name`, что обеспечивает уникальность и позволяет сопоставлять экземпляры детекторов с их конфигурациями.

