export const TASK_STORAGE_KEY = 'anomalyLensTrackedTasks';

export const getTrackedTaskIds = (): string[] => {
  try {
    const stored = localStorage.getItem(TASK_STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (e) {
    console.error("Failed to parse tracked tasks from localStorage", e);
    return [];
  }
};

export const addTrackedTaskId = (taskId: string): void => {
  const currentIds = getTrackedTaskIds();
  if (!currentIds.includes(taskId)) {
    localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify([...currentIds, taskId]));
    console.log(`Task ID ${taskId} added to tracking list.`);
    // Диспетчеризация кастомного события, если нужно немедленно оповестить другие компоненты
    // window.dispatchEvent(new CustomEvent('trackedTasksUpdated'));
  } else {
    console.log(`Task ID ${taskId} is already in tracking list.`);
  }
};

// Можно добавить и другие утилиты для работы с задачами, если понадобятся
// export const removeTrackedTaskId = (taskId: string): void => {
//   let currentIds = getTrackedTaskIds();
//   currentIds = currentIds.filter(id => id !== taskId);
//   localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(currentIds));
//   console.log(`Task ID ${taskId} removed from tracking list.`);
// };

// export const clearAllTrackedTasks = (): void => {
//   localStorage.removeItem(TASK_STORAGE_KEY);
//   console.log('All tracked tasks cleared from localStorage.');
// };