import React from 'react';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';

interface ScoreChipProps {
    score: number | null | undefined;
    size?: 'small' | 'medium';
}

// Function to determine chip color based on score
const getScoreChipColor = (score: number | null | undefined): 'default' | 'warning' | 'error' => {
  if (score === null || score === undefined || score < 0.5) return 'default';
  if (score < 0.8) return 'warning';
  return 'error';
};

const ScoreChip: React.FC<ScoreChipProps> = ({ score, size = 'medium' }) => {
    if (score === null || score === undefined) {
        return <Typography variant="caption" color="textSecondary">N/A</Typography>;
    }

    return (
        <Chip
            label={score.toFixed(4)}
            color={getScoreChipColor(score)}
            size={size}
        />
    );
};

export default ScoreChip; 