import React, { useState } from 'react';
import { 
  Button, 
  CircularProgress, 
  Alert, 
  AlertTitle, 
  Box,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Typography,
  Chip,
  Paper
} from '@mui/material';
import BuildIcon from '@mui/icons-material/Build';
import SchoolIcon from '@mui/icons-material/School';
import { trainMultilevelSystem } from '../services/multilevelApi';
import { TrainResponse } from '../types/multilevel';

interface MultilevelTrainButtonProps {
  onTrainingComplete?: (response: TrainResponse) => void;
}

const MultilevelTrainButton: React.FC<MultilevelTrainButtonProps> = ({ 
  onTrainingComplete 
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<TrainResponse | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const handleTraining = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    setDialogOpen(false);
    
    try {
      const response = await trainMultilevelSystem();
      setSuccess(response);
      if (onTrainingComplete) {
        onTrainingComplete(response);
      }
    } catch (err) {
      console.error('Error during multilevel training:', err);
      setError(err instanceof Error ? err.message : 'Ошибка при обучении моделей');
    } finally {
      setLoading(false);
    }
  };

  const openConfirmDialog = () => {
    setDialogOpen(true);
  };

  const closeConfirmDialog = () => {
    setDialogOpen(false);
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} variant="filled">
          <AlertTitle>Ошибка обучения</AlertTitle>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} variant="filled">
          <AlertTitle>Успешно!</AlertTitle>
          <Typography variant="body2">
            {success.message}
          </Typography>
        </Alert>
      )}

      <Paper variant="outlined" sx={{ p: 2, mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Обучение многоуровневой системы
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Процесс может занять продолжительное время
          </Typography>
        </Box>
        <Chip 
          label="ML модели" 
          color="primary" 
          size="small" 
          icon={<SchoolIcon />} 
        />
      </Paper>

      <Button
        variant="contained"
        color="primary"
        onClick={openConfirmDialog}
        disabled={loading}
        fullWidth
        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <BuildIcon />}
        size="large"
      >
        {loading ? 'Выполняется обучение...' : 'Обучить модели'}
      </Button>

      <Dialog
        open={dialogOpen}
        onClose={closeConfirmDialog}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          Подтверждение обучения моделей
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Процесс обучения может занять значительное время в зависимости от объема данных. 
            Во время обучения все модели будут переобучены с использованием текущих данных.
            Вы уверены, что хотите начать обучение?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={closeConfirmDialog} color="inherit">
            Отмена
          </Button>
          <Button onClick={handleTraining} color="primary" autoFocus variant="contained">
            Начать обучение
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MultilevelTrainButton; 