import React, { useState, useEffect, useCallback } from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import { useNavigate } from 'react-router-dom';

import ModelCard from '../components/ModelCard';
import { ModelStatus, ModelConfig } from '../types/models';
import { 
  getModelsStatus, 
  trainModel, 
  detectAnomalies, 
  detectEnsemble 
} from '../services/api';
import { TaskLaunchResponse } from '../types/tasks'; 
import { addTrackedTaskId } from '../utils/taskUtils'; // Импорт из утилит

interface RunningTasksState {
  [modelFilename: string]: 'training' | 'detecting';
}

const ModelOverviewPage: React.FC = () => {
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus> | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [runningTasks, setRunningTasks] = useState<RunningTasksState>({});
  const [ensembleDetecting, setEnsembleDetecting] = useState<boolean>(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({ open: false, message: '', severity: 'success' });
  const navigate = useNavigate();

  const fetchStatuses = useCallback(async () => {
    try {
      const statuses = await getModelsStatus();
      setModelStatuses(statuses);
    } catch (err) {
      console.error("Ошибка загрузки статусов моделей:", err);
      setError(err instanceof Error ? err.message : 'Не удалось загрузить статусы моделей');
      setModelStatuses({});
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
    fetchStatuses().finally(() => setIsLoading(false));
  }, [fetchStatuses]);

  const handleTrain = async (config: ModelConfig) => {
    if (runningTasks[config.model_filename]) return;
    
    setRunningTasks(prev => ({ ...prev, [config.model_filename]: 'training' }));
    setSnackbar({ open: true, message: `Запуск обучения для ${config.model_filename}...`, severity: 'success' });
    
    try {
      const response = await trainModel(config);
      setSnackbar({ open: true, message: response.message || `Обучение ${config.model_filename} запущено.`, severity: 'success' });
      setTimeout(fetchStatuses, 5000);
    } catch (err) {
      console.error(`Ошибка запуска обучения для ${config.model_filename}:`, err);
      const errorMessage = err instanceof Error ? err.message : 'Неизвестная ошибка обучения';
      setSnackbar({ open: true, message: `Ошибка обучения ${config.model_filename}: ${errorMessage}`, severity: 'error' });
    } finally {
      setRunningTasks(prev => {
        const newState = { ...prev };
        delete newState[config.model_filename];
        return newState;
      });
    }
  };

  const handleDetect = async (config: ModelConfig) => {
    if (runningTasks[config.model_filename]) return;

    setRunningTasks(prev => ({ ...prev, [config.model_filename]: 'detecting' }));
    setSnackbar({ open: true, message: `Запуск детекции для ${config.model_filename}...`, severity: 'success' });

    try {
      const response = await detectAnomalies(config);
      setSnackbar({ open: true, message: response.message || `Детекция ${config.model_filename} запущена.`, severity: 'success' });
       setTimeout(fetchStatuses, 5000);
    } catch (err) {
      console.error(`Ошибка запуска детекции для ${config.model_filename}:`, err);
      const errorMessage = err instanceof Error ? err.message : 'Неизвестная ошибка детекции';
      setSnackbar({ open: true, message: `Ошибка детекции ${config.model_filename}: ${errorMessage}`, severity: 'error' });
    } finally {
       setRunningTasks(prev => {
        const newState = { ...prev };
        delete newState[config.model_filename];
        return newState;
      });
    }
  };

  const handleDetectEnsemble = async () => {
    if (ensembleDetecting) return;
    
    setEnsembleDetecting(true);
    setSnackbar({ open: true, message: 'Запуск детекции ансамблем...', severity: 'success' });
    
    try {
      const params: { combination_method?: 'average' | 'max' | 'weighted_average', ensemble_threshold?: number } = { 
        combination_method: 'weighted_average', 
        ensemble_threshold: 0.5 
      };
      const response: TaskLaunchResponse = await detectEnsemble(params);
      console.log('Ensemble detect task started:', response);
      setSnackbar({ open: true, message: `${response.message} (ID: ${response.task_id})`, severity: 'success' });
      
      if (response.task_id) {
        addTrackedTaskId(response.task_id);
        // Опционально: переходим на страницу задач
        // navigate('/tasks'); 
      }

    } catch (err) {
       console.error('Ошибка запуска детекции ансамблем:', err);
       const errorMessage = err instanceof Error ? err.message : 'Неизвестная ошибка детекции ансамблем';
      setSnackbar({ open: true, message: `Ошибка детекции ансамблем: ${errorMessage}`, severity: 'error' });
    } finally {
      setEnsembleDetecting(false);
    }
  };

  const handleCloseSnackbar = (event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Обзор моделей и Запуск
      </Typography>

      {isLoading && <CircularProgress sx={{ display: 'block', margin: 'auto', my: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {!isLoading && !error && modelStatuses && (
        <Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={ensembleDetecting ? <CircularProgress size={20} color="inherit" /> : <PlayCircleOutlineIcon />}
            onClick={handleDetectEnsemble}
            disabled={ensembleDetecting || Object.keys(runningTasks).length > 0}
            sx={{ mb: 3 }}
          >
            {ensembleDetecting ? 'Запуск Ансамбля...' : 'Запустить Детекцию Ансамблем'}
          </Button>

          <Grid container spacing={3}>
            {Object.entries(modelStatuses).map(([modelName, status]) => (
              status?.config?.model_filename ? (
                 <Grid key={modelName} size={{ xs: 12, md: 6, lg: 4 }}> 
                   <ModelCard 
                     modelName={modelName}
                     status={status}
                     onTrain={handleTrain}
                     onDetect={handleDetect}
                     isTraining={runningTasks[status.config.model_filename] === 'training'}
                     isDetecting={runningTasks[status.config.model_filename] === 'detecting'}
                   />
                 </Grid>
              ) : (
                <Grid key={modelName} size={{ xs: 12, md: 6, lg: 4 }}>
                   <Alert severity="warning">Конфигурация для модели {modelName} не полная или отсутствует.</Alert>
                </Grid>
              )
            ))}
          </Grid>
        </Box>
      )}
       <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }} variant="filled">
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ModelOverviewPage;