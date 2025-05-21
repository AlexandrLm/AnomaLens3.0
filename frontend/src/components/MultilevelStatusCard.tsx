import React from 'react';
import { 
  Card, 
  CardContent,
  CardHeader, 
  Typography, 
  Box, 
  Chip, 
  Grid, 
  Divider,
  CircularProgress,
  LinearProgress,
  Tooltip,
  Avatar,
  Stack,
  Badge
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import TroubleshootIcon from '@mui/icons-material/Troubleshoot';
import MemoryIcon from '@mui/icons-material/Memory';
import TimelineIcon from '@mui/icons-material/Timeline';
import LayersIcon from '@mui/icons-material/Layers';
import { MultilevelStatus, MultilevelDetectorStatus } from '../types/multilevel';

interface MultilevelStatusCardProps {
  status: MultilevelStatus | null;
  loading: boolean;
}

// Получаем иконку для типа детектора
const getDetectorIcon = (type: string) => {
  switch (type.toLowerCase()) {
    case 'isolation_forest':
    case 'one_class_svm':
      return <TroubleshootIcon fontSize="small" />;
    case 'autoencoder':
    case 'lstm':
      return <MemoryIcon fontSize="small" />;
    default:
      return null;
  }
};

// Компонент для отображения статуса одного уровня
const LevelStatusSection: React.FC<{
  title: string;
  detectors: Record<string, MultilevelDetectorStatus> | undefined;
  icon: React.ReactNode;
}> = ({ title, detectors, icon }) => {
  if (!detectors) return null;
  
  const totalDetectors = Object.keys(detectors).length;
  const trainedDetectors = Object.values(detectors).filter(d => d.is_trained).length;
  const progress = totalDetectors > 0 ? (trainedDetectors / totalDetectors) * 100 : 0;

  return (
    <Box mb={3}>
      <Stack direction="row" alignItems="center" spacing={1} mb={1}>
        <Avatar 
          sx={{ 
            width: 32, 
            height: 32, 
            bgcolor: trainedDetectors === totalDetectors ? 'success.main' : 'primary.main' 
          }}
        >
          {icon}
        </Avatar>
        <Typography variant="subtitle1" fontWeight={500}>
          {title}
          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
            ({trainedDetectors}/{totalDetectors})
          </Typography>
        </Typography>
      </Stack>
      
      <Box mb={2}>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          color={progress === 100 ? "success" : "primary"}
          sx={{ height: 8, borderRadius: 4, mb: 1 }}
        />
      </Box>
      
      <Grid container spacing={1}>
        {Object.entries(detectors).map(([name, detector]) => (
          <Box key={name} sx={{ m: 0.5 }}>
            <Tooltip title={`Тип: ${detector.type}`}>
              <Chip
                label={name}
                icon={getDetectorIcon(detector.type) || undefined}
                color={detector.is_trained ? 'success' : 'default'}
                variant={detector.is_trained ? "filled" : "outlined"}
                size="small"
              />
            </Tooltip>
          </Box>
        ))}
      </Grid>
    </Box>
  );
};

const MultilevelStatusCard: React.FC<MultilevelStatusCardProps> = ({ status, loading }) => {
  // Подсчитываем общую статистику обученности
  const getTrainingStats = () => {
    if (!status) return { trained: 0, total: 0 };
    
    let trained = 0;
    let total = 0;
    
    const countDetectors = (level: Record<string, MultilevelDetectorStatus>) => {
      Object.values(level).forEach(detector => {
        total++;
        if (detector.is_trained) trained++;
      });
    };
    
    if (status.transaction_level) countDetectors(status.transaction_level);
    if (status.behavior_level) countDetectors(status.behavior_level);
    if (status.time_series_level) countDetectors(status.time_series_level);
    
    return { trained, total };
  };
  
  const { trained, total } = getTrainingStats();
  const readyPercentage = total > 0 ? Math.round((trained / total) * 100) : 0;
  
  return (
    <Card variant="outlined">
      <CardHeader
        title="Статус многоуровневой системы"
        avatar={
          <Badge badgeContent={`${readyPercentage}%`} color={readyPercentage === 100 ? "success" : "primary"}>
            <LayersIcon color="primary" />
          </Badge>
        }
      />
      <Divider />
      <CardContent>
        {loading ? (
          <Box display="flex" justifyContent="center" mt={3} mb={3}>
            <CircularProgress />
          </Box>
        ) : status ? (
          <>
            <Box display="flex" alignItems="center" mb={4} sx={{ position: 'relative' }}>
              <Box position="relative" display="inline-flex" mr={3}>
                <CircularProgress
                  variant="determinate"
                  value={readyPercentage}
                  size={80}
                  thickness={4}
                  color={readyPercentage === 100 ? "success" : "primary"}
                />
                <Box
                  sx={{
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    position: 'absolute',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography
                    variant="caption"
                    component="div"
                    fontSize="1rem"
                    fontWeight="bold"
                  >
                    {`${readyPercentage}%`}
                  </Typography>
                </Box>
              </Box>
              <Box>
                <Typography variant="subtitle1" fontWeight={500}>
                  Готовность системы
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {trained} из {total} детекторов обучены
                </Typography>
                {readyPercentage === 100 && (
                  <Chip 
                    icon={<CheckCircleIcon />} 
                    label="Система готова к работе" 
                    size="small" 
                    color="success" 
                    sx={{ mt: 1 }}
                  />
                )}
              </Box>
            </Box>
            
            <Divider sx={{ mb: 3 }} />
            
            {status.transaction_level && (
              <LevelStatusSection 
                title="Транзакционный уровень" 
                detectors={status.transaction_level}
                icon={<TroubleshootIcon fontSize="small" />}
              />
            )}
            
            {status.behavior_level && (
              <LevelStatusSection 
                title="Поведенческий уровень" 
                detectors={status.behavior_level}
                icon={<MemoryIcon fontSize="small" />}
              />
            )}
            
            {status.time_series_level && (
              <LevelStatusSection 
                title="Уровень временных рядов" 
                detectors={status.time_series_level}
                icon={<TimelineIcon fontSize="small" />}
              />
            )}
          </>
        ) : (
          <Typography color="textSecondary" align="center" mt={2}>
            Нет данных о статусе многоуровневой системы
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default MultilevelStatusCard; 