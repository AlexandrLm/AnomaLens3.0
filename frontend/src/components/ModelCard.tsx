import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import CircularProgress from '@mui/material/CircularProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HubIcon from '@mui/icons-material/Hub';
import InsightsIcon from '@mui/icons-material/Insights';
import ForestIcon from '@mui/icons-material/Forest';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import MemoryIcon from '@mui/icons-material/Memory';
import { ModelStatus, ModelConfig } from '../types/models';

interface ModelCardProps {
  modelName: string;
  status: ModelStatus;
  onTrain: (config: ModelConfig) => void;
  onDetect: (config: ModelConfig) => void;
  onClick?: (modelName: string) => void;
  isTraining?: boolean;
  isDetecting?: boolean;
}

const modelTypeIcons: { [key: string]: React.ElementType } = {
  statistical: InsightsIcon,
  isolation_forest: ForestIcon,
  autoencoder: HubIcon,
  default: HelpOutlineIcon,
};

const getStatusProps = (status: ModelStatus): { color: 'success' | 'warning' | 'error' | 'default', icon: React.ReactElement, label: string } => {
  if (status.trained && !status.message) {
    return { color: 'success', icon: <CheckCircleIcon />, label: 'Обучена' };
  } else if (status.trained && status.message) {
    return { color: 'warning', icon: <ErrorIcon />, label: 'Обучена (с ошибкой)' };
  } else if (!status.trained && status.message) {
    return { color: 'error', icon: <ErrorIcon />, label: 'Ошибка обучения' };
  } else {
    return { color: 'default', icon: <HelpOutlineIcon />, label: 'Не обучена' };
  }
};

const ModelCard: React.FC<ModelCardProps> = ({
  modelName,
  status,
  onTrain,
  onDetect,
  onClick,
  isTraining = false,
  isDetecting = false
}) => {
  const { features, config } = status;
  const statusProps = getStatusProps(status);
  const ModelIcon = modelTypeIcons[config.type] || modelTypeIcons.default;

  const handleTrainClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onTrain(config);
  };

  const handleDetectClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDetect(config);
  };

  const handleCardClick = () => {
    if (onClick) {
      onClick(modelName);
    }
  };

  return (
    <Card sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
          <Tooltip title={modelName} placement="top-start">
            <Typography gutterBottom variant="h6" component="div" sx={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', pr: 1 }}>
              {modelName}
            </Typography>
          </Tooltip>
          <Chip 
            icon={statusProps.icon} 
            label={statusProps.label} 
            color={statusProps.color}
            size="small" 
          />
        </Box>

        <Stack spacing={0.8}>
          <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
            <ModelIcon fontSize="inherit" sx={{ mr: 0.5, verticalAlign: 'bottom', opacity: 0.8 }} />
            <Tooltip title={config.type} placement="top-start">
              <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Тип: {config.type}
              </Box>
            </Tooltip>
          </Typography>

          {status.message && (
            <Tooltip title={status.message} placement="top-start">
              <Typography variant="caption" color={statusProps.color === 'success' ? 'text.secondary' : statusProps.color + '.main'} sx={{ display: 'flex', alignItems: 'center' }}>
                <MemoryIcon fontSize="inherit" sx={{ mr: 0.5, verticalAlign: 'bottom', opacity: 0.8 }} />
                <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  Статус: {status.message.substring(0, 50)}{status.message.length > 50 ? '...' : ''}
                </Box>
              </Typography>
            </Tooltip>
          )}

          {features && features.length > 0 && (
            <Tooltip title={features.join(', ')} placement="top-start">
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  Признаки: {features.join(', ')}
                </Box>
              </Typography>
            </Tooltip>
          )}
          <Tooltip title={config.model_filename} placement="top-start">
            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
              <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Файл: {config.model_filename}
              </Box>
            </Typography>
          </Tooltip>
        </Stack>
      </CardContent>

      <CardActions sx={{ pt: 0 }}>
        <Button 
          size="small" 
          onClick={handleTrainClick} 
          disabled={isTraining || isDetecting}
          startIcon={isTraining ? <CircularProgress size={16} /> : null}
        >
          {isTraining ? 'Обучение...' : 'Обучить'}
        </Button>
        <Button 
          size="small" 
          onClick={handleDetectClick} 
          disabled={isDetecting || isTraining || !status.trained}
          startIcon={isDetecting ? <CircularProgress size={16} /> : null}
        >
          {isDetecting ? 'Детекция...' : 'Запустить'}
        </Button>
      </CardActions>
    </Card>
  );
};

export default ModelCard; 