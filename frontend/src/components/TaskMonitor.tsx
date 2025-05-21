import React from 'react';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PendingIcon from '@mui/icons-material/Pending';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import { TaskStatus } from '../types/tasks'; // Импортируем тип
import { formatDistanceToNow } from 'date-fns'; // Для отображения времени
import { ru } from 'date-fns/locale'; // Для русского языка

interface TaskMonitorProps {
  tasks: TaskStatus[];
}

const TaskStatusIcon: React.FC<{ status: TaskStatus['status'] }> = ({ status }) => {
  switch (status) {
    case 'pending':
      return <PendingIcon color="action" />;
    case 'running':
      return <CircularProgress size={20} />; // Используем Progress для running
    case 'completed':
      return <CheckCircleOutlineIcon color="success" />;
    case 'failed':
    case 'error':
      return <ErrorOutlineIcon color="error" />;
    default:
      return null;
  }
};

const TaskMonitor: React.FC<TaskMonitorProps> = ({ tasks }) => {
  if (!tasks || tasks.length === 0) {
    return <Typography>Нет активных или недавних задач для отображения.</Typography>;
  }

  return (
    <List dense>
      {tasks.map((task) => (
        <ListItem key={task.task_id} divider>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <TaskStatusIcon status={task.status} />
          </ListItemIcon>
          <ListItemText 
            primary={`Задача: ${task.task_id} (${task.status})`}
            secondary={
              <Box component="span">
                <Typography variant="body2" component="span" sx={{ display: 'block' }}>
                  {task.details}
                </Typography>
                {task.start_time && (
                  <Typography variant="caption" component="span" color="text.secondary">
                    {`Запущена: ${formatDistanceToNow(new Date(task.start_time), { addSuffix: true, locale: ru })}`}
                  </Typography>
                )}
                {task.end_time && task.status !== 'running' && task.status !== 'pending' && (
                  <Typography variant="caption" component="span" color="text.secondary" sx={{ ml: 1 }}>
                     {`| Завершена: ${formatDistanceToNow(new Date(task.end_time), { addSuffix: true, locale: ru })}`}
                  </Typography>
                )}
                 {/* TODO: Отображение результатов task.result? */}
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
};

export default TaskMonitor; 