import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';

// Типы для данных диаграмм
export interface DetectorTypeDistributionItem {
  name: string; // detector_type
  value: number; // count
}

export interface AnomaliesOverTimeItem {
  date: string; // отформатированная дата (день, неделя, месяц)
  count: number; // количество аномалий
}

export interface ScoreDistributionItem {
  range: string; // диапазон скора, например "0.5-0.6"
  count: number; // количество аномалий в этом диапазоне
}

interface AnomalyChartsProps {
  detectorTypeDistribution: DetectorTypeDistributionItem[];
  anomaliesOverTime: AnomaliesOverTimeItem[]; // Пока не используем, добавим позже
  scoreDistribution: ScoreDistributionItem[]; // Пока не используем, добавим позже
}

// Цвета для PieChart (можно расширить)
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AA00FF', '#FF00AA'];

const AnomalyCharts: React.FC<AnomalyChartsProps> = ({
  detectorTypeDistribution,
  // anomaliesOverTime,
  // scoreDistribution,
}) => {
  return (
    <Box sx={{ mt: 4, mb: 3 }}>
      <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
        Аналитика по аномалиям
      </Typography>
      <Grid container spacing={3}>
        {/* Диаграмма распределения по типам детекторов */}
        {detectorTypeDistribution && detectorTypeDistribution.length > 0 && (
          <Grid size={{ xs: 12, md: 6, lg: 4}}>
            <Paper sx={{ p: 2, height: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Typography variant="h6" gutterBottom component="div">
                По типам детекторов
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={detectorTypeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    // label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`} // Можно кастомизировать label
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    nameKey="name"
                  >
                    {detectorTypeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number, name: string) => [`${value} (${((value / detectorTypeDistribution.reduce((sum, item) => sum + item.value, 0)) * 100).toFixed(1)}%)`, name]}/>
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}

        {/* Заглушка для других диаграмм */}
        {/* 
        <Grid size={{ xs: 12, md: 6, lg: 8}}>
            <Paper sx={{ p: 2, height: '400px' }}>
              <Typography variant="h6" gutterBottom>Аномалии по времени</Typography>
              <ResponsiveContainer width="100%" height="90%">
                  <BarChart data={anomaliesOverTime}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#82ca9d" name="Кол-во аномалий" />
                  </BarChart>
              </ResponsiveContainer>
            </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 6, lg: 4}}>
             <Paper sx={{ p: 2, height: '400px' }}>
              <Typography variant="h6" gutterBottom>Распределение скоров</Typography>
                 // ... диаграмма для scoreDistribution ...
             </Paper>
        </Grid>
        */}

      </Grid>
    </Box>
  );
};

export default AnomalyCharts;