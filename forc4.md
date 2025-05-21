C4 Model (Context, Containers, Components, Code) — это "увеличительное стекло" для архитектуры программного обеспечения. Она позволяет описывать систему на разных уровнях детализации:

1.  **Уровень 1: Контекст (System Context)** - Общий взгляд, показывающий, как ваша система вписывается в существующий ИТ-ландшафт.
2.  **Уровень 2: Контейнеры (Containers)** - Показывает высокоуровневые технологические "строительные блоки" (приложения, базы данных, файловые системы), из которых состоит ваша система, и как они взаимодействуют.
3.  **Уровень 3: Компоненты (Components)** - Детализирует каждый контейнер, показывая его основные модули/компоненты и их взаимосвязи.
4.  **Уровень 4: Код (Code)** - (Опционально) Детализирует отдельные компоненты до уровня классов, интерфейсов и их взаимодействий (например, UML-диаграммы).

---

### Уровень 1: Контекст Системы (System Context Diagram)

**Описание:** Этот уровень показывает вашу систему как "черный ящик" и её взаимодействие с внешними пользователями и другими системами.

**Система в центре:**

*   **Название:** `Интеллектуальная Система Обнаружения Аномалий Olist` (или короче `Система Обнаружения Аномалий`)
*   **Описание:** Веб-приложение и API для обучения моделей, обнаружения аномалий в данных Olist, сохранения результатов и предоставления объяснений.
*   **Технологии (ключевые):** Python, FastAPI, SQLAlchemy, ML-библиотеки.

**Внешние Пользователи (Люди):**

1.  **`Аналитик Данных / ML Инженер`**
    *   **Описание:** Пользователь, отвечающий за обучение моделей, запуск процессов детекции, анализ обнаруженных аномалий и настройку системы.
    *   **Взаимодействия с Системой:**
        *   Отправляет запросы на обучение моделей (через API).
        *   Отправляет запросы на детекцию аномалий (через API).
        *   Получает статус фоновых задач (через API).
        *   Запрашивает, создает, обновляет, удаляет записи об аномалиях (через API).
        *   Запрашивает LLM-объяснения для аномалий (через API).
        *   Управляет настройками приложения (через API).
        *   Просматривает данные Olist (заказы, продукты и т.д.) (через API).
        *   Управляет конфигурацией многоуровневой системы (через API).
        *   *(Потенциально)* Взаимодействует с веб-интерфейсом (если он будет реализован поверх API, но в текущем скоупе файлов это не очевидно, поэтому фокусируемся на API).

2.  **`Менеджер / Бизнес-Пользователь`**
    *   **Описание:** Пользователь, заинтересованный в результатах обнаружения аномалий и их объяснениях для принятия бизнес-решений.
    *   **Взаимодействия с Системой (через API, возможно, через фронтенд-приложение, использующее это API):**
        *   Запрашивает список обнаруженных аномалий с фильтрами.
        *   Запрашивает детали конкретной аномалии.
        *   Запрашивает LLM-объяснения для аномалий.
        *   Просматривает данные заказов для контекста аномалий.

**Внешние Системы:**

1.  **`База Данных Olist`** (здесь это часть внутреннего устройства, но с точки зрения "черного ящика" на самом высоком уровне, система работает *с данными* Olist. Если БД Olist была бы полностью внешней и не управляемой приложением, она была бы здесь. В нашем случае она скорее контейнер). Для чистоты C4, на уровне контекста, можно её не показывать, а показать на уровне контейнеров. Однако, если подчеркнуть, что это *внешний датасет*, то можно. Будем считать её частью системы.
2.  **`Ollama LLM Сервис`**
    *   **Описание:** Внешний сервис, предоставляющий LLM для генерации текстовых объяснений.
    *   **Технологии:** HTTP API.
    *   **Взаимодействия с Системой:**
        *   `Система Обнаружения Аномалий` отправляет запрос на генерацию объяснения (HTTP POST JSON) в `Ollama LLM Сервис`.
        *   `Ollama LLM Сервис` возвращает текстовое объяснение (HTTP JSON ответ) в `Систему Обнаружения Аномалий`.

---

### Уровень 2: Контейнеры (Container Diagram)

**Описание:** "Увеличиваем" Систему Обнаружения Аномалий и показываем её основные технологические блоки (приложения, хранилища данных) и их взаимодействия.

**Контейнеры внутри `Системы Обнаружения Аномалий`:**

1.  **`FastAPI Веб-Приложение`** (Основной API бэкенд)
    *   **Описание:** Реализует API для всех взаимодействий с системой. Обрабатывает HTTP запросы, управляет бизнес-логикой, координирует фоновые задачи.
    *   **Технологии:** Python, FastAPI, Uvicorn (как ASGI сервер).
    *   **Размещение:** Например, Docker-контейнер.

2.  **`Реляционная База Данных`** (Хранилище данных Olist и результатов работы системы)
    *   **Описание:** Хранит данные датасета Olist (заказы, товары и т.д.), а также таблицы для записей об аномалиях (`anomalies`) и настроек приложения (`settings`).
    *   **Технологии:** SQLAlchemy (ORM), СУБД (например, SQLite для разработки, PostgreSQL для продакшена).
    *   **Размещение:** Например, отдельный Docker-контейнер или управляемый сервис БД.

3.  **`Файловое Хранилище Моделей`** (Хранилище обученных ML моделей)
    *   **Описание:** Директория на сервере (или облачное хранилище), где сохраняются сериализованные файлы обученных ML моделей (например, `.joblib` файлы).
    *   **Технологии:** Файловая система сервера, Joblib (для сериализации/десериализации).
    *   **Размещение:** Локальная файловая система сервера, где запущен FastAPI, или облачное хранилище (S3, Google Cloud Storage).

**Взаимодействия между Контейнерами и с Внешними Сущностями:**

*   **`Аналитик Данных / ML Инженер` -> `FastAPI Веб-Приложение`:**
    *   Отправляет HTTP API запросы (JSON) для:
        *   Управления моделями (обучение, детекция)
        *   CRUD операций с аномалиями и настройками
        *   Запроса данных Olist
        *   Управления многоуровневой системой
        *   Запроса LLM-объяснений
    *   Получает HTTP API ответы (JSON) с результатами, статусами задач.

*   **`Менеджер / Бизнес-Пользователь` -> `FastAPI Веб-Приложение`:**
    *   Отправляет HTTP API запросы (JSON) для:
        *   Получения списка аномалий и их деталей
        *   Запроса LLM-объяснений
        *   Просмотра данных заказов
    *   Получает HTTP API ответы (JSON).

*   **`FastAPI Веб-Приложение` -> `Реляционная База Данных`:**
    *   **Чтение данных Olist:** SQL-запросы (через SQLAlchemy) для получения данных заказов, товаров, клиентов и т.д.
    *   **CRUD Аномалий:** SQL-запросы (через SQLAlchemy) для создания, чтения, обновления, удаления записей в таблице `anomalies`.
    *   **CRUD Настроек:** SQL-запросы (через SQLAlchemy) для таблицы `settings`.
    *   **Получение данных для ML:** SQL-запросы (через SQLAlchemy в `ml_service/common.py::load_data_from_db`) для загрузки данных, необходимых для обучения и детекции.

*   **`FastAPI Веб-Приложение` -> `Файловое Хранилище Моделей`:**
    *   **Сохранение моделей:** Запись `.joblib` файлов (во время обучения моделей, например, в `ml_service/detector.py::save_model`).
    *   **Загрузка моделей:** Чтение `.joblib` файлов (во время инициализации детекторов или перед детекцией, например, в `ml_service/detector.py::load_model` или `api/anomalies.py::_initialize_detector`).

*   **`FastAPI Веб-Приложение` -> `Ollama LLM Сервис`:**
    *   Отправляет HTTP POST запрос (JSON) с промптом для генерации объяснения.
    *   Получает HTTP JSON ответ с текстом объяснения.

---

### Уровень 3: Компоненты (Component Diagram)

**Описание:** Детализируем ключевые контейнеры, показывая их основные компоненты (модули) и их взаимодействия.
Рассмотрим контейнер **`FastAPI Веб-Приложение`**.

**Компоненты внутри `FastAPI Веб-Приложение`:**

1.  **`API Маршрутизаторы (Routers)`** (`main.py`, `api/*.py`)
    *   **Описание:** Набор модулей FastAPI APIRouter, отвечающих за определение эндпоинтов, обработку HTTP запросов, валидацию входных данных (с помощью Pydantic схем) и вызов соответствующей бизнес-логики.
    *   **Ключевые модули/файлы:**
        *   `main.py`: Основное приложение, подключение всех роутеров.
        *   `api/anomalies.py`: Роутер для управления аномалиями, обучения и детекции моделей.
        *   `api/orders.py`: Роутер для получения информации о заказах.
        *   `api/settings.py`: Роутер для управления настройками.
        *   `api/multilevel.py`: Роутер для управления многоуровневой системой детекции.
        *   `api/geolocation.py`, `api/utils.py` (генерируемые роутеры для Products, Customers и т.д.).
    *   **Технологии:** FastAPI, Pydantic.

2.  **`Слой CRUD Операций (Database Access Layer)`** (`crud.py`)
    *   **Описание:** Модуль, инкапсулирующий всю логику взаимодействия с базой данных. Предоставляет функции для создания, чтения, обновления и удаления данных.
    *   **Технологии:** SQLAlchemy ORM.

3.  **`Сервис Машинного Обучения (ML Service Core Logic)`** (`ml_service/` все модули)
    *   **Описание:** Основной компонент, содержащий всю логику, связанную с моделями обнаружения аномалий. Включает:
        *   **`Detector Factory`** (`ml_service/detector_factory.py`): Создание экземпляров детекторов.
        *   **`Anomaly Detectors`** (`ml_service/detector.py`, `vae_detector.py`, etc.): Реализации различных алгоритмов обнаружения аномалий.
        *   **`Multilevel Anomaly Detector`** (`ml_service/multilevel_detector.py`): Оркестратор для многоуровневой детекции.
        *   **`Multilevel Detector Service`** (`ml_service/multilevel_service.py`): Сервис для управления многоуровневым детектором (конфигурация, обучение, детекция).
        *   **`Data Processing Utilities`** (`ml_service/common.py`): Функции для загрузки данных из БД и инженерии признаков.
        *   **`LLM Explainer Integration`** (`ml_service/llm_explainer.py`): Компонент для взаимодействия с Ollama.
    *   **Технологии:** Pandas, NumPy, Scikit-learn, PyTorch, NetworkX, Joblib, SHAP.

4.  **`Менеджер Фоновых Задач (Task Manager)`** (`services/task_manager.py`)
    *   **Описание:** Компонент для управления длительными операциями (обучение, детекция) в фоновом режиме. Предоставляет API для создания задач и получения их статуса.
    *   **Технологии:** FastAPI `BackgroundTasks`, `uuid`.

5.  **`Модели Данных и Схемы Валидации (Data Models & Schemas)`** (`models.py`, `schemas.py`)
    *   **Описание:** Определения структур данных: SQLAlchemy модели для БД и Pydantic схемы для API. Используются всеми компонентами для работы с данными.
    *   **Технологии:** SQLAlchemy, Pydantic.

**Взаимодействия Компонентов внутри `FastAPI Веб-Приложение`:**

*   **`API Маршрутизаторы` -> `Слой CRUD Операций`:**
    *   Вызовы функций из `crud.py` для получения, сохранения, обновления, удаления данных в/из `Реляционной Базы Данных`. (Например, `api/orders.py` вызывает `crud.get_orders_with_details`).

*   **`API Маршрутизаторы` -> `Сервис Машинного Обучения`:**
    *   **`api/anomalies.py`:**
        *   Для обучения/детекции одной модели: вызывает функции `train_model_task` / `detect_anomalies_task`, которые используют `DetectorFactory` для создания детектора и его методы `train`/`detect`.
        *   Для LLM-объяснений: вызывает `LLMExplainer.generate_explanation()`.
    *   **`api/multilevel.py`:**
        *   Вызывает методы `MultilevelDetectorService` (например, `train_task_wrapper`, `detect_async_task_wrapper`, `get_detector_status`, `update_config`).

*   **`API Маршрутизаторы` -> `Менеджер Фоновых Задач`:**
    *   Все эндпоинты, запускающие длительные операции (обучение, детекция), используют `TaskManager.create_task()` и `TaskManager.run_task_in_background()`.
    *   Эндпоинт `/api/tasks/task_status/{task_id}` вызывает `TaskManager.get_task_status()`.

*   **`Сервис Машинного Обучения` -> `Слой CRUD Операций`:**
    *   **`ml_service/common.py::load_data_from_db`:** Напрямую использует сессию SQLAlchemy для выполнения SQL-запросов и загрузки данных из `Реляционной Базы Данных`.
    *   **`ml_service/multilevel_service.py::detect`:** Вызывает `crud.create_anomaly` для сохранения обнаруженных многоуровневых аномалий в `Реляционную Базу Данных`.
    *   **`api/anomalies.py::detect_anomalies_task`:** Вызывает `crud.create_anomaly` для сохранения аномалий от отдельных детекторов.

*   **`Сервис Машинного Обучения` -> `Файловое Хранилище Моделей`:**
    *   Все `AnomalyDetector`ы используют `save_model` и `load_model` для взаимодействия с файловым хранилищем.

*   **`Сервис Машинного Обучения (LLMExplainer)` -> `Ollama LLM Сервис` (Внешняя система):**
    *   HTTP POST запросы для генерации объяснений.

*   **Все Компоненты -> `Модели Данных и Схемы Валидации`:**
    *   Используют SQLAlchemy модели для работы с БД и Pydantic схемы для данных API.

---

### Уровень 4: Код (Code Diagram - Примеры)

**Описание:** Детализация выбранных компонентов до уровня классов и их основных методов/атрибутов. Вместо полных UML диаграмм, опишем ключевые классы и их связи для понимания LLM.

**Пример 1: Компонент `AnomalyDetector` и его наследники (из `ml_service/detector.py`)**

*   **`AnomalyDetector(ABC)` (Абстрактный Базовый Класс):**
    *   **Атрибуты:** `model_name: str`, `model: Optional[Any]`, `scaler: Optional[StandardScaler]`, `is_trained: bool`, `min_score_: Optional[float]`, `max_score_: Optional[float]`.
    *   **Ключевые методы:**
        *   `preprocess(data, fit_scaler)` (abstract)
        *   `train(data)` (abstract)
        *   `detect(data)` (abstract)
        *   `save_model(path)`
        *   `load_model(filepath)`
        *   `_get_attributes_to_save()` (для переопределения)
        *   `_load_additional_attributes(loaded_data)` (для переопределения)
        *   `fit_normalizer(scores)`
        *   `normalize_score(scores)`
        *   `get_shap_explainer()`
        *   `get_explanation_details(data_for_explanation_raw)`
        *   `_reset_state()`

*   **`IsolationForestDetector(AnomalyDetector)`:**
    *   **Атрибуты (дополнительно):** `features: List[str]`, `n_estimators: int`, `contamination: Union[str, float]`, `random_state: Optional[int]`, `shap_explainer: Optional[shap.TreeExplainer]`.
    *   **`model`** здесь типа `sklearn.ensemble.IsolationForest`.
    *   **Переопределяет:** `preprocess`, `train`, `detect`, `_get_attributes_to_save`, `_load_additional_attributes`, `get_shap_explanations`.
    *   **Взаимодействия:**
        *   `train` использует `StandardScaler.fit_transform` и `IsolationForest.fit`.
        *   `detect` использует `StandardScaler.transform` и `IsolationForest.score_samples`.
        *   `get_shap_explanations` использует `shap.TreeExplainer`.

*   **`VAEDetector(AnomalyDetector)`:**
    *   **Атрибуты (дополнительно):** `features: List[str]`, `input_dim: int`, `latent_dim: int`, `hidden_dim: int`, `epochs: int`, `kld_weight: float`, `dropout_rate: float`, `device: str`, `threshold_: Optional[float]`, `explainer: Optional[shap.GradientExplainer]`, `background_data_for_shap: Optional[torch.Tensor]`.
    *   **`model`** здесь типа `VAE` (кастомный `nn.Module`).
    *   **Вложенный класс:** `_VAEModelWrapperForSHAP(nn.Module)`.
    *   **Переопределяет:** `preprocess`, `train`, `detect`, `_get_attributes_to_save`, `_load_additional_attributes`, `get_shap_explanations`, `_reset_state`.
    *   **Взаимодействия:**
        *   `train` использует `StandardScaler`, создает и обучает `VAE` модель, вычисляет `threshold_`, инициализирует `shap.GradientExplainer`.
        *   `detect` использует `StandardScaler`, `VAE` для реконструкции, вычисляет ошибку.
        *   `get_shap_explanations` использует `_VAEModelWrapperForSHAP` и `shap.GradientExplainer`.

**Пример 2: Компонент `MultilevelDetectorService` (из `ml_service/multilevel_service.py`)**

*   **`MultilevelDetectorService`:**
    *   **Атрибуты:** `model_base_path: str`, `config_filename: str`, `multilevel_detector: Optional[MultilevelAnomalyDetector]`.
    *   **Ключевые методы:**
        *   `_load_or_create_detector()` -> `_create_default_detector_from_settings()`: Инициализирует `self.multilevel_detector` на основе `settings.ml_service.multilevel_detector_default_config`.
        *   `train_task_wrapper(task_id, ...)`: Обертка для `TaskManager`.
        *   `train(data, load_data_params)`:
            *   Вызывает `load_data_from_db` и `engineer_features`.
            *   Итерируется по детекторам в `self.multilevel_detector.transaction_detectors`, `behavior_detectors`, `time_series_detectors`.
            *   Для каждого детектора вызывает `detector.train()` и `detector.save_model()`.
            *   Использует `self.multilevel_detector._prepare_behavior_data` и `_prepare_time_series_data` для подготовки специфичных данных для этих уровней.
        *   `detect_async_task_wrapper(task_id, ...)`: Обертка для `TaskManager`.
        *   `detect_async(...)`: Асинхронная обертка для `detect`.
        *   `detect(data, load_data_params, ...)`:
            *   Вызывает `load_data_from_db` и `engineer_features`.
            *   Вызывает `self.multilevel_detector.detect(...)`.
            *   Обрабатывает результат `result_df` (фильтрует аномалии).
            *   Для каждой аномалии формирует `AnomalyCreate` схему и вызывает `crud.create_anomaly` для сохранения в БД.
        *   `update_config(new_config_dict)`: Пересоздает `self.multilevel_detector` с новой конфигурацией.
        *   `get_config()`: Возвращает конфигурацию.
        *   `get_detector_status()`: Собирает статус каждого детектора, используя `DetectorFactory._generate_detector_name` для сопоставления с конфигурацией.

*   **Зависимость от `MultilevelAnomalyDetector`:** `MultilevelDetectorService` является фасадом и оркестратором для `MultilevelAnomalyDetector`.
*   **Зависимость от `DetectorFactory`:** Используется для генерации имен и потенциально для динамического создания детекторов (хотя в текущей реализации `_initialize_detectors` больше полагается на конфигурацию).
*   **Зависимость от `crud`:** Для сохранения аномалий.
*   **Зависимость от `common`:** Для загрузки данных и инженерии признаков.

**Пример 3: Взаимодействие в `api/anomalies.py`**

*   **Роутер `anomalies.router`:**
    *   **Эндпоинт `POST /train_model` -> `train_model_endpoint`:**
        *   Вызывает `task_manager.create_task()`.
        *   Вызывает `task_manager.run_task_in_background(..., train_model_task, ...)`.
    *   **Функция `train_model_task(task_id, ...)`:**
        *   Вызывает `_initialize_detector(detector_type, model_config, ...)`.
            *   `_initialize_detector` использует `DetectorFactory.get_detector_class()`.
        *   Вызывает `common.load_data_from_db()`.
        *   Вызывает `common.engineer_features()`.
        *   Вызывает `detector_instance.train()`.
        *   Вызывает `detector_instance.save_model()`.
        *   Вызывает `task_manager.update_task_status()`.
    *   **Эндпоинт `GET /api/anomalies/{anomaly_id}/explain-llm` -> `get_anomaly_llm_explanation`:**
        *   Вызывает `crud.get_anomaly(anomaly_id)`.
        *   Создает `LLMExplainer(ollama_base_url, model_name)`.
        *   Вызывает `llm_explainer.generate_explanation(...)`.