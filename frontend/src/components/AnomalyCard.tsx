import React, { useMemo, useCallback } from 'react';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import InsightsIcon from '@mui/icons-material/Insights';
import HubIcon from '@mui/icons-material/Hub';
import ForestIcon from '@mui/icons-material/Forest';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import { Anomaly } from '../types/anomaly';
import ScoreChip from './ScoreChip';

interface AnomalyCardProps {
  anomaly: Anomaly;
  onClick: (anomaly: Anomaly) => void;
}

const formatTimestamp = (detection_date: string | null | undefined): string => {
  if (!detection_date) return 'N/A';
  try {
    const timestamp = Date.parse(detection_date);
    if (isNaN(timestamp)) {
      console.warn('Invalid date format received:', detection_date);
      return 'Invalid Date';
    }
    const date = new Date(timestamp);
    return date.toLocaleString(); 
  } catch (e) {
    console.error('Error parsing date:', detection_date, e);
    return 'Date Error';
  }
};

const getScoreColor = (score: number | null | undefined): string => {
  if (score === null || score === undefined || isNaN(score)) return '#cccccc';
  const normalizedScore = Math.max(0, Math.min(1, score));
  const hue = (1 - normalizedScore) * 120;
  return `hsl(${hue}, 80%, 70%)`;
};

const detectorIcons: { [key: string]: React.ElementType } = {
  statistical: InsightsIcon,
  isolation_forest: ForestIcon,
  autoencoder: HubIcon,
  graph: InsightsIcon,
  default: HelpOutlineIcon,
};

const AnomalyCard: React.FC<AnomalyCardProps> = ({ anomaly, onClick }) => {
  const { id, order_id, detector_type, anomaly_score, detection_date } = anomaly;

  const formattedTimestamp = useMemo(
    () => formatTimestamp(detection_date),
    [detection_date]
  );

  const handleCardClick = useCallback(() => {
    onClick(anomaly);
  }, [anomaly, onClick]);

  const displayDetector = detector_type || 'N/A';
  const displayOrderId = order_id || 'N/A';
  const cardTitle = `Аномалия #${id}`;

  const scoreColor = useMemo(() => getScoreColor(anomaly_score), [anomaly_score]);
  const DetectorIcon = detectorIcons[detector_type || 'default'] || detectorIcons.default;

  return (
    <Card 
      sx={{
        maxWidth: 400,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative',
        '&:hover .score-gradient-indicator': { 
           height: '6px',
           opacity: 1,
        }
      }}
    >
      <Box 
        className="score-gradient-indicator"
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '5px',
          opacity: 0.85,
          background: scoreColor,
          transition: 'height 0.3s ease-in-out, opacity 0.3s ease-in-out', 
          zIndex: 3,
          pointerEvents: 'none',
        }}
      />

      <CardActionArea onClick={handleCardClick} sx={{ flexGrow: 1 }}>
        <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', pt: 2.5 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
            <Tooltip title={cardTitle} placement="top-start">
              <Typography gutterBottom variant="h6" component="div" sx={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', pr: 1 }}>
                {cardTitle}
              </Typography>
            </Tooltip>
            <ScoreChip score={anomaly_score} size="small" />
          </Box>

          <Stack spacing={0.8}>
            <Tooltip title={displayOrderId} placement="top-start">
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                <Box component="strong" sx={{ minWidth: '75px', mr: 0.5 }}>Order ID:</Box>
                <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {displayOrderId}
                </Box>
              </Typography>
            </Tooltip>

            <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
              <Box component="strong" sx={{ minWidth: '75px', mr: 0.5 }}>Detector:</Box>
              <DetectorIcon fontSize="inherit" sx={{ mr: 0.5, verticalAlign: 'bottom', opacity: 0.8 }} />
              <Tooltip title={displayDetector} placement="top-start">
                  <Box component="span" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {displayDetector}
                  </Box>
               </Tooltip>
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
              <Box component="strong" sx={{ minWidth: '75px', mr: 0.5 }}>Detected:</Box>
              <span>{formattedTimestamp}</span>
            </Typography>
          </Stack>

        </CardContent>
      </CardActionArea>
    </Card>
  );
};

export default AnomalyCard;
