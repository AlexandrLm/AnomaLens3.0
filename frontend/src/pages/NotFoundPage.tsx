import React from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

const NotFoundPage: React.FC = () => {
  return (
    <Box sx={{ textAlign: 'center', mt: 8 }}>
      <Typography variant="h1" component="h1" gutterBottom>
        404
      </Typography>
      <Typography variant="h5" component="p">
        Страница не найдена
      </Typography>
    </Box>
  );
};

export default NotFoundPage; 