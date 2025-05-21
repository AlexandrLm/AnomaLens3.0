import React, { useState, useEffect, useCallback } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Grid, 
  Paper, 
  Alert, 
  Tabs,
  Tab,
  Stepper,
  Step,
  StepLabel,
  Chip,
  Card,
  CardHeader,
  CardContent,
  LinearProgress,
  useTheme
} from '@mui/material';
import LayersIcon from '@mui/icons-material/Layers';
import TuneIcon from '@mui/icons-material/Tune';
import AssessmentIcon from '@mui/icons-material/Assessment';
import MultilevelStatusCard from '../components/MultilevelStatusCard';
import MultilevelDetectionForm from '../components/MultilevelDetectionForm';
import MultilevelTrainButton from '../components/MultilevelTrainButton';
import { getMultilevelStatus } from '../services/multilevelApi';
import { MultilevelStatus, DetectResponse, TrainResponse } from '../types/multilevel';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`system-tabpanel-${index}`}
      aria-labelledby={`system-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const MultilevelSystemPage: React.FC = () => {
  const [status, setStatus] = useState<MultilevelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastDetection, setLastDetection] = useState<DetectResponse | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const theme = useTheme();

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const statusData = await getMultilevelStatus();
      setStatus(statusData);
    } catch (err) {
      console.error('Error fetching multilevel status:', err);
      setError(err instanceof Error ? err.message : 'Ошибка при получении статуса системы');
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleDetectionComplete = (response: DetectResponse) => {
    setLastDetection(response);
    if (response.status === 'success') {
      fetchStatus();
    }
  };

  const handleTrainingComplete = (response: TrainResponse) => {
    if (response.status === 'success') {
      fetchStatus();
    }
  };

  const getSystemReadiness = () => {
    if (!status) return 0;
    let trained = 0;
    let total = 0;
    const countDetectors = (level: Record<string, { is_trained: boolean }> | undefined) => {
      if (!level) return;
      Object.values(level).forEach(detector => {
        total++;
        if (detector.is_trained) trained++;
      });
    };
    if (status.transaction_level) countDetectors(status.transaction_level);
    if (status.behavior_level) countDetectors(status.behavior_level);
    if (status.time_series_level) countDetectors(status.time_series_level);
    return total > 0 ? (trained / total) * 100 : 0;
  };

  const systemReadiness = getSystemReadiness();
  const isSystemReady = systemReadiness === 100;

  const getActiveStep = () => {
    if (!status) return 0;
    const allTransactionTrained = status.transaction_level && 
      Object.values(status.transaction_level).every(d => d.is_trained);
    const allBehaviorTrained = status.behavior_level && 
      Object.values(status.behavior_level).every(d => d.is_trained);
    const allTimeSeriesTrained = status.time_series_level && 
      Object.values(status.time_series_level).every(d => d.is_trained);
    if (!allTransactionTrained) return 0;
    if (!allBehaviorTrained) return 1;
    if (!allTimeSeriesTrained) return 2;
    return 3;
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: theme.palette.primary.main, color: 'white', borderRadius: 2 }}>
        <Typography variant="h4" gutterBottom>
          Многоуровневая система обнаружения аномалий
        </Typography>
        <Typography variant="body1">
          Анализ данных на трех разных уровнях: транзакционном, поведенческом и временном, 
          для выявления сложных аномалий с минимумом ложных срабатываний.
        </Typography>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      <Stepper activeStep={getActiveStep()} alternativeLabel sx={{ mb: 4 }}>
        <Step>
          <StepLabel>Транзакционный уровень</StepLabel>
        </Step>
        <Step>
          <StepLabel>Поведенческий уровень</StepLabel>
        </Step>
        <Step>
          <StepLabel>Уровень временных рядов</StepLabel>
        </Step>
        <Step>
          <StepLabel>Система готова</StepLabel>
        </Step>
      </Stepper>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          variant="fullWidth"
          textColor="primary"
          indicatorColor="primary"
        >
          <Tab icon={<LayersIcon />} label="Статус системы" id="system-tab-0" />
          <Tab icon={<TuneIcon />} label="Обнаружение аномалий" id="system-tab-1" />
          <Tab icon={<AssessmentIcon />} label="Результаты" id="system-tab-2" disabled={!lastDetection} />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={4}>
          <Grid xs={12} md={6}>
            <MultilevelStatusCard status={status} loading={loading} />
          </Grid>
          <Grid xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Управление моделями
              </Typography>
              <Typography variant="body2" paragraph color="text.secondary">
                Обучение моделей необходимо для полноценной работы системы обнаружения аномалий.
                Процесс может занять продолжительное время в зависимости от объема данных.
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Общая готовность системы
                </Typography>
                <Box display="flex" alignItems="center">
                  <Box width="100%" mr={1}>
                    <LinearProgress 
                      variant="determinate" 
                      value={systemReadiness} 
                      color={isSystemReady ? "success" : "primary"}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Box>
                  <Box minWidth={35}>
                    <Typography variant="body2" color="text.secondary">
                      {`${Math.round(systemReadiness)}%`}
                    </Typography>
                  </Box>
                </Box>
              </Box>
              <Box mt={2}>
                <MultilevelTrainButton onTrainingComplete={handleTrainingComplete} />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        {!isSystemReady && status && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Не все детекторы обучены. Для полноценной работы необходимо обучить все компоненты системы.
          </Alert>
        )}
        <MultilevelDetectionForm onDetectionComplete={handleDetectionComplete} />
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        {lastDetection && (
          <Card>
            <CardHeader 
              title="Результаты обнаружения аномалий" 
              subheader={`Выполнено за ${lastDetection.elapsed_time_seconds.toFixed(2)} сек.`}
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                      {lastDetection.save_statistics.total_detected_anomalies_before_save}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Обнаружено аномалий
                    </Typography>
                  </Paper>
                </Grid>
                <Grid xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                      {lastDetection.save_statistics.newly_saved_anomalies_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Сохранено новых аномалий
                    </Typography>
                  </Paper>
                </Grid>
                <Grid xs={12} md={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                      {lastDetection.save_statistics.skipped_duplicates_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Пропущено дубликатов
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
              
              {lastDetection.save_statistics.newly_saved_anomaly_ids.length > 0 && (
                <Box mt={3}>
                  <Typography variant="subtitle1" gutterBottom>
                    Новые аномалии:
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1}>
                    {lastDetection.save_statistics.newly_saved_anomaly_ids.map(id => (
                      <Chip key={id} label={`ID: ${id}`} color="primary" size="small" />
                    ))}
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        )}
      </TabPanel>
    </Container>
  );
};

export default MultilevelSystemPage; 