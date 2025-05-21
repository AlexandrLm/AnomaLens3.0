import React, { useState, useEffect, useCallback, useMemo } from 'react'; // Добавлен useMemo
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import Modal from '@mui/material/Modal';

import AnomalyCard from '../components/AnomalyCard';
import AnomalyDetailView from '../components/AnomalyDetailView';
import AnomalyCharts, { DetectorTypeDistributionItem } from '../components/AnomalyCharts'; // Импортируем компонент и типы
import { getAnomalies } from '../services/api';
import { Anomaly } from '../types/anomaly';

const INITIAL_LOAD_LIMIT = 50;

const modalStyle = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '80%',
  maxWidth: '1000px',
  maxHeight: '90vh',
  bgcolor: 'background.paper',
  border: '1px solid #ccc',
  borderRadius: '8px',
  boxShadow: 24,
  p: 4,
  overflowY: 'auto',
};

const AnomalyHistoryPage: React.FC = () => {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);
  const [totalCount, setTotalCount] = useState<number>(0);

  const fetchAnomalies = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const params = {
        limit: INITIAL_LOAD_LIMIT,
        skip: 0,
      };
      const response = await getAnomalies(params); 
      
      // Обрабатываем пагинированный ответ
      setTotalCount(response.total);
      const processedAnomalies = response.items.map(anomaly => {
          let parsedDetailsContent: Record<string, any> | undefined;
          if (anomaly.details_dict) {
              parsedDetailsContent = anomaly.details_dict;
          } else if (typeof anomaly.details === 'string' && anomaly.details.trim() !== '') {
              try {
                  const correctedJsonString = anomaly.details.replace(/\'/g, '"').replace(/None/g, 'null').replace(/True/g, 'true').replace(/False/g, 'false');
                  parsedDetailsContent = JSON.parse(correctedJsonString);
              } catch (e) {
                  console.error("Failed to parse details on load:", e, "Original details:", anomaly.details);
                  parsedDetailsContent = { parse_error: true, raw_details: anomaly.details };
              }
          }
          return { ...anomaly, parsed_details: parsedDetailsContent };
      });
      setAnomalies(processedAnomalies);
    } catch (err) {
      console.error("Error loading anomaly history:", err);
      setError(err instanceof Error ? err.message : 'Failed to load anomaly history');
      setAnomalies([]);
      setTotalCount(0);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnomalies();
  }, [fetchAnomalies]);

  const handleAnomalySelect = (anomaly: Anomaly) => {
      setSelectedAnomaly(anomaly);
  };

  const handleCloseModal = () => {
    setSelectedAnomaly(null);
  };

  // Подготовка данных для диаграмм с использованием useMemo для оптимизации
  const detectorTypeDistributionData = useMemo((): DetectorTypeDistributionItem[] => {
    if (!anomalies.length) return [];
    const counts: Record<string, number> = {};
    anomalies.forEach(anomaly => {
      const type = anomaly.detector_type || 'Unknown';
      counts[type] = (counts[type] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [anomalies]);

  // TODO: Подготовить данные для anomaliesOverTimeData и scoreDistributionData
  // const anomaliesOverTimeData = useMemo(() => { ... }, [anomalies]);
  // const scoreDistributionData = useMemo(() => { ... }, [anomalies]);


  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Anomaly History ({totalCount > 0 ? `${totalCount} total, ${anomalies.length} loaded` : '0'}) 
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', pt: 5 }}>
             <CircularProgress />
          </Box>
      )}
      
      {!isLoading && !error && anomalies.length > 0 && (
        // Отображаем диаграммы, если есть данные
        <AnomalyCharts
          detectorTypeDistribution={detectorTypeDistributionData}
          anomaliesOverTime={[]} // Заглушка
          scoreDistribution={[]} // Заглушка
        />
      )}

      {!isLoading && !error && (
          <Grid container spacing={2} sx={{mt: anomalies.length > 0 ? 2 : 0}}> 
             {anomalies.map((anomaly) => (
                  <Grid key={anomaly.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}> 
                    <AnomalyCard 
                      anomaly={anomaly} 
                      onClick={handleAnomalySelect}
                    />
                  </Grid>
              ))}
          </Grid>
      )}

      {!isLoading && !error && anomalies.length === 0 && (
          <Typography sx={{ textAlign: 'center', mt: 5, color: 'text.secondary' }}>
              No anomalies found.
          </Typography>
      )}
      
      <Modal
          open={selectedAnomaly !== null}
          onClose={handleCloseModal}
          aria-labelledby="anomaly-detail-modal-title"
          aria-describedby="anomaly-detail-modal-description"
      >
          <Box sx={modalStyle}>
              {selectedAnomaly && (
                  <AnomalyDetailView 
                      anomaly={selectedAnomaly}
                      onClose={handleCloseModal}
                  />
              )}
          </Box>
      </Modal>
    </Box>
  );
};

export default AnomalyHistoryPage;