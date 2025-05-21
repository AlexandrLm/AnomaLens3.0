import React, { useState, useEffect, useRef, useCallback } from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import TaskMonitor from '../components/TaskMonitor';
import { getTaskStatus } from '../services/api';
import { TaskStatus } from '../types/tasks';
import { TASK_STORAGE_KEY, getTrackedTaskIds } from '../utils/taskUtils';

const POLLING_INTERVAL = 5000;

const TaskStatusPage: React.FC = () => {
  const [trackedTaskIds, setTrackedTaskIds] = useState<string[]>(getTrackedTaskIds());
  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchStatus = async (taskId: string): Promise<TaskStatus | null> => {
    try {
      const status = await getTaskStatus(taskId);
      setTaskStatuses(prev => ({ ...prev, [taskId]: status }));
      return status;
    } catch (err) {
      console.error(`Ошибка загрузки статуса для задачи ${taskId}:`, err);
      setTaskStatuses(prev => ({
        ...prev,
        [taskId]: prev[taskId] || { 
          task_id: taskId, 
          status: 'error', 
          details: `Failed to fetch status: ${err instanceof Error ? err.message : 'Unknown error'}`,
          start_time: new Date().toISOString(),
          end_time: new Date().toISOString(),
          result: null 
        }
      }));
      return null;
    }
  };

  const pollActiveTasks = useCallback(async (isInitialLoad = false) => {
    if (!isInitialLoad && !isLoading) setIsLoading(true); 
    
    const currentTrackedIds = getTrackedTaskIds(); // Получаем актуальные ID на момент вызова
    
    // Обновляем состояние trackedTaskIds только если оно действительно изменилось,
    // чтобы избежать лишних срабатываний useEffect, который зависит от trackedTaskIds.
    if (JSON.stringify(currentTrackedIds) !== JSON.stringify(trackedTaskIds)) {
        setTrackedTaskIds(currentTrackedIds);
        // Если ID изменились, дальнейший опрос для старых ID может быть не нужен сразу,
        // так как новый pollActiveTasks будет вызван из-за изменения trackedTaskIds.
        // Однако, мы все равно продолжим этот цикл, чтобы опросить currentTrackedIds.
    }
    
    let activeTasksFound = false;
    if (currentTrackedIds.length > 0) {
        const promises = currentTrackedIds.map(async (taskId) => {
        const currentStatusInState = taskStatuses[taskId]; // Берем статус из состояния
        // Опрашиваем, если задачи нет в состоянии, или она 'pending'/'running'
        if (!currentStatusInState || currentStatusInState.status === 'pending' || currentStatusInState.status === 'running') {
            const newStatus = await fetchStatus(taskId);
            if (newStatus && (newStatus.status === 'pending' || newStatus.status === 'running')) {
            activeTasksFound = true;
            }
        } else {
            // Если задача уже имеет финальный статус в состоянии, не считаем ее активной для _нового_ поллинга
        }
        });
        await Promise.all(promises);
    }
    
    if (!isInitialLoad) setIsLoading(false);
    
    if (activeTasksFound) {
        if (!intervalRef.current) {
            console.log("Активные задачи обнаружены. Запуск/возобновление поллинга.");
            intervalRef.current = setInterval(() => pollActiveTasks(false), POLLING_INTERVAL);
        }
    } else {
        if (intervalRef.current) {
            console.log("Все отслеживаемые задачи завершены или неактивны. Остановка поллинга.");
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }
  }, [taskStatuses, trackedTaskIds, isLoading]); // Зависимости pollActiveTasks

  // Основной useEffect для инициализации и управления интервалом поллинга
  useEffect(() => {
    setIsLoading(true);
    pollActiveTasks(true).finally(() => setIsLoading(false)); // Первичный опрос

    // Этот return будет вызван при размонтировании компонента ИЛИ перед следующим запуском этого эффекта
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null; // Явно обнуляем ref
        console.log("Интервал поллинга очищен при размонтировании или перезапуске эффекта.");
      }
    };
  }, [pollActiveTasks]); // Перезапускаем этот эффект, если сама функция pollActiveTasks изменилась

  // useEffect для отслеживания изменений в localStorage
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === TASK_STORAGE_KEY) {
        console.log("Обнаружено изменение в localStorage для TASK_STORAGE_KEY (TaskStatusPage)");
        const newIds = getTrackedTaskIds();
        // Обновляем состояние trackedTaskIds, это вызовет срабатывание pollActiveTasks
        // через основной useEffect, так как pollActiveTasks зависит от trackedTaskIds
        setTrackedTaskIds(newIds); 
      }
    };

    window.addEventListener('storage', handleStorageChange);
    console.log("Listener для storage добавлен");
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      console.log("Listener для storage удален");
    };
  }, []); // Пустой массив зависимостей, чтобы listener добавился/удалился один раз


  const taskListToDisplay = Object.values(taskStatuses)
    .filter(task => trackedTaskIds.includes(task.task_id))
    .sort((a, b) => 
      (b.start_time && a.start_time) ? (new Date(b.start_time).getTime() - new Date(a.start_time).getTime()) : 0
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Статус фоновых задач
      </Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      {isLoading && taskListToDisplay.length === 0 && <Typography sx={{my: 2}}>Загрузка статусов...</Typography>}
      
      <TaskMonitor tasks={taskListToDisplay} />

      {!isLoading && taskListToDisplay.length === 0 && trackedTaskIds.length > 0 && (
         <Typography sx={{my: 2, color: 'text.secondary'}}>Нет данных о статусах для отслеживаемых задач. Задачи могли еще не обработаться или уже удалены из отслеживания.</Typography>
      )}
      {!isLoading && trackedTaskIds.length === 0 && (
         <Typography sx={{my: 2, color: 'text.secondary'}}>Нет задач для отслеживания. Запустите задачу (например, детекцию ансамблем), чтобы она появилась здесь.</Typography>
      )}
    </Box>
  );
};

export default TaskStatusPage;