import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
import logging
from enum import Enum
import threading # Потокобезопасность для self.tasks

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" # Используется для задач, запущенных через run_task_in_background
    PROCESSING = "processing" # Используется внутри самой задачи для указания прогресса
    COMPLETED = "completed"
    COMPLETED_NO_DATA = "completed_no_data" # Для случаев, когда детекция прошла, но нечего было обрабатывать
    COMPLETED_NO_ANOMALIES_FOUND = "completed_no_anomalies_found" # Для случаев, когда детекция прошла, но аномалий не найдено
    FAILED = "failed"

class Task:
    def __init__(self, task_id: str, description: str, status: TaskStatus = TaskStatus.PENDING):
        self.task_id = task_id
        self.description = description
        self.status = status
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.details: Any = None # Может быть строкой или словарем
        self.result: Any = None
        self.error_type: Optional[str] = None # Тип ошибки, если была (например, 'ValueError')

    def update_status(self, status: TaskStatus, details: Any = None, result: Any = None, error_type: Optional[str] = None):
        self.status = status
        self.updated_at = datetime.utcnow()
        if details is not None:
            self.details = details
        if result is not None:
            self.result = result
        if error_type is not None:
            self.error_type = error_type
        logger.info(f"Статус задачи {self.task_id} обновлен: {status}, Детали: {details}")

# Словарь для хранения статусов задач
# Ключ - task_id (str), значение - TaskStatus
tasks_db: Dict[str, TaskStatus] = {}

router = APIRouter()

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = threading.Lock() # Для потокобезопасного доступа к self.tasks

    def create_task(self, description: str) -> str:
        task_id = str(uuid.uuid4())
        task = Task(task_id, description)
        with self._lock:
            self.tasks[task_id] = task
        logger.info(f"Задача {task_id} создана: {description}")
        return task_id

    def update_task_status(self, task_id: str, status: TaskStatus, details: Any = None, result: Any = None, error_type: Optional[str] = None):
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update_status(status, details, result, error_type)
            else:
                logger.warning(f"Попытка обновить статус несуществующей задачи: {task_id}")

    def get_task_status(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.tasks.get(task_id)

    async def run_task_in_background(
        self,
        background_tasks: BackgroundTasks, # Этот параметр теперь будет использоваться
        task_id: str,
        task_func: Callable, # Это будет ваша async def функция
        *args: Any,          # Аргументы для task_func
        **kwargs: Any        # Именованные аргументы для task_func
    ):
        """
        Планирует выполнение функции task_func в фоновом режиме с помощью FastAPI BackgroundTasks.
        """
        logger.info(f"Планирование фоновой задачи {task_id} для функции {task_func.__name__}...")
        try:
            # Обновляем статус задачи на "running" ПЕРЕД добавлением в background_tasks,
            # чтобы клиент сразу видел, что задача была принята в обработку.
            self.update_task_status(task_id, status=TaskStatus.RUNNING, details="Задача запущена в фоновом режиме.")
            
            # Используем background_tasks.add_task для корректного запуска.
            # FastAPI позаботится о правильном выполнении task_func, даже если она async.
            # Важно: НЕ используйте await здесь, add_task сама добавляет задачу в очередь.
            background_tasks.add_task(task_func, task_id=task_id, *args, **kwargs)
            # Сообщение о том, что задача "запланирована", теперь менее релевантно,
            # так как мы уже обновили статус на "running".
            # logger.info(f"Фоновая задача {task_id} успешно запланирована.")
        except Exception as e:
            logger.error(f"Ошибка при планировании фоновой задачи {task_id}: {e}", exc_info=True)
            # Обновляем статус задачи, если планирование не удалось
            self.update_task_status(
                task_id,
                status=TaskStatus.FAILED,
                details=f"Ошибка планирования задачи: {str(e)}",
                error_type="SchedulingError"
            )
            # Можно также перевыбросить исключение, если это критично для вызывающего кода,
            # но обычно для фоновых задач это не делается здесь.

# Глобальный экземпляр менеджера задач
task_manager = TaskManager()

@router.get("/task_status/{task_id}", response_model=TaskStatus, tags=["Tasks Management"])
async def get_task_status_api(task_id: str):
    """
    Получает статус фоновой задачи по ее ID.
    """
    status_info = task_manager.get_task_status(task_id)
    if status_info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")
    return status_info

# --- Схемы для ответов API, которые используют TaskID ---
class TaskCreationResponse(BaseModel):
    task_id: str
    message: str
    status_endpoint: str = Field(description="Эндпоинт для проверки статуса задачи")
    initial_status: TaskStatus = Field(description="Начальный статус задачи") 