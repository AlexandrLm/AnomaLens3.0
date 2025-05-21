import React, { useState } from 'react';
import { 
  Card, 
  CardContent,
  CardHeader,
  Typography, 
  Button, 
  Box, 
  Slider, 
  Stack,
  TextField,
  Alert,
  AlertTitle,
  CircularProgress,
  FormHelperText,
  Divider,
  Paper,
  InputAdornment,
  Tooltip,
  IconButton,
  Collapse
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import TuneIcon from '@mui/icons-material/Tune';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { DetectionParams, DetectResponse } from '../types/multilevel';
import { detectWithMultilevelSystem } from '../services/multilevelApi';

interface MultilevelDetectionFormProps {
  onDetectionComplete?: (response: DetectResponse) => void;
}

const MultilevelDetectionForm: React.FC<MultilevelDetectionFormProps> = ({ 
  onDetectionComplete 
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<DetectResponse | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  
  // Параметры детекции
  const [params, setParams] = useState<DetectionParams>({
    transaction_threshold: 0.6,
    behavior_threshold: 0.6,
    time_series_threshold: 0.6,
    final_threshold: 0.5,
    filter_period_days: null
  });

  const handleSliderChange = (name: keyof DetectionParams) => 
    (_event: Event, newValue: number | number[]) => {
      setParams({ ...params, [name]: newValue as number });
    };

  const handleInputChange = (name: keyof DetectionParams) => 
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value === '' ? null : Number(event.target.value);
      if (value !== null && isNaN(value)) return;
      
      // Ограничения для пороговых значений (0-1)
      if (name !== 'filter_period_days' && value !== null && (value < 0 || value > 1)) return;
      
      setParams({ ...params, [name]: value });
    };

  const handleDetection = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await detectWithMultilevelSystem(params);
      setSuccess(response);
      if (onDetectionComplete) {
        onDetectionComplete(response);
      }
    } catch (err) {
      console.error('Error during multilevel detection:', err);
      setError(err instanceof Error ? err.message : 'Ошибка при запуске детекции аномалий');
    } finally {
      setLoading(false);
    }
  };

  const toggleAdvanced = () => {
    setAdvancedOpen(!advancedOpen);
  };

  return (
    <Card variant="outlined">
      <CardHeader 
        title="Запуск детекции аномалий"
        avatar={<TuneIcon color="primary" />}
        action={
          <Tooltip title={advancedOpen ? "Скрыть расширенные настройки" : "Показать расширенные настройки"}>
            <IconButton onClick={toggleAdvanced}>
              {advancedOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Tooltip>
        }
      />
      <Divider />
      <CardContent>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} variant="filled">
            <AlertTitle>Ошибка</AlertTitle>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 3 }} variant="filled">
            <AlertTitle>Успешно!</AlertTitle>
            <Typography variant="body2">
              Обнаружено аномалий: {success.save_statistics.total_detected_anomalies_before_save}
            </Typography>
            <Typography variant="body2">
              Сохранено новых: {success.save_statistics.newly_saved_anomalies_count}
            </Typography>
            <Typography variant="body2">
              Время: {success.elapsed_time_seconds.toFixed(2)} сек.
            </Typography>
          </Alert>
        )}

        <Paper variant="outlined" sx={{ p: 3, mb: 3 }}>
          <Stack spacing={3}>
            <Box>
              <Typography gutterBottom variant="subtitle1" fontWeight={500}>
                Итоговый порог обнаружения
                <Tooltip title="Определяет чувствительность системы к обнаружению аномалий. Более низкое значение увеличивает количество обнаруженных аномалий, но может привести к ложным срабатываниям.">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" color="action" />
                  </IconButton>
                </Tooltip>
              </Typography>
              <Slider
                value={params.final_threshold}
                onChange={handleSliderChange('final_threshold')}
                aria-labelledby="final-threshold-slider"
                step={0.05}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
                min={0}
                max={1}
                sx={{ mt: 1 }}
                valueLabelDisplay="auto"
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Больше аномалий
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Меньше ложных срабатываний
                </Typography>
              </Box>
            </Box>

            <TextField
              label="Период анализа (дней)"
              value={params.filter_period_days === null ? '' : params.filter_period_days}
              onChange={handleInputChange('filter_period_days')}
              type="number"
              InputLabelProps={{
                shrink: true,
              }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <Tooltip title="Оставьте поле пустым для анализа всех доступных данных">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" color="action" />
                      </IconButton>
                    </Tooltip>
                  </InputAdornment>
                ),
              }}
              inputProps={{
                min: 1,
                max: 365,
                step: 1,
              }}
              fullWidth
              variant="outlined"
              sx={{ mt: 2 }}
              helperText="Оставьте поле пустым для анализа всех данных без ограничения по времени"
            />
          </Stack>
        </Paper>

        <Collapse in={advancedOpen}>
          <Paper variant="outlined" sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 2 }} fontWeight={500}>
              Пороги для отдельных уровней
            </Typography>
            
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" gutterBottom display="flex" alignItems="center">
                  Транзакционный уровень:
                  <Box component="span" fontWeight="bold" ml={1}>
                    {params.transaction_threshold.toFixed(2)}
                  </Box>
                </Typography>
                <Slider
                  value={params.transaction_threshold}
                  onChange={handleSliderChange('transaction_threshold')}
                  step={0.05}
                  min={0}
                  max={1}
                  size="small"
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom display="flex" alignItems="center">
                  Поведенческий уровень:
                  <Box component="span" fontWeight="bold" ml={1}>
                    {params.behavior_threshold.toFixed(2)}
                  </Box>
                </Typography>
                <Slider
                  value={params.behavior_threshold}
                  onChange={handleSliderChange('behavior_threshold')}
                  step={0.05}
                  min={0}
                  max={1}
                  size="small"
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom display="flex" alignItems="center">
                  Уровень временных рядов:
                  <Box component="span" fontWeight="bold" ml={1}>
                    {params.time_series_threshold.toFixed(2)}
                  </Box>
                </Typography>
                <Slider
                  value={params.time_series_threshold}
                  onChange={handleSliderChange('time_series_threshold')}
                  step={0.05}
                  min={0}
                  max={1}
                  size="small"
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </Paper>
        </Collapse>

        <Button
          variant="contained"
          color="primary"
          onClick={handleDetection}
          disabled={loading}
          fullWidth
          size="large"
          sx={{ py: 1.5 }}
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {loading ? 'Выполняется анализ...' : 'Запустить детекцию аномалий'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default MultilevelDetectionForm; 