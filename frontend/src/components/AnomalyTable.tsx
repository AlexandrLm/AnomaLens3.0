import React from 'react';
import { 
  DataGrid, 
  GridColDef, 
  GridPaginationModel, 
  GridFilterModel, 
  GridRowParams,
  GridValueGetter,
  GridRenderCellParams,
  GridToolbar,
} from '@mui/x-data-grid';
import Box from '@mui/material/Box';
import { Anomaly } from '../types/anomaly';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'; // Icon for details

interface AnomalyTableProps {
  anomalies: Anomaly[];
  loading: boolean;
  rowCount: number; // Общее количество для пагинации на стороне сервера
  paginationModel: GridPaginationModel;
  onPaginationModelChange: (model: GridPaginationModel) => void;
  onAnomalySelect: (anomaly: Anomaly) => void;
  // TODO: Добавить обработку фильтрации и сортировки, если нужна серверная
  // filterModel?: GridFilterModel;
  // onFilterModelChange?: (model: GridFilterModel) => void;
}

// Function to determine chip color based on score (example)
const getScoreChipColor = (score: number | null): 'default' | 'warning' | 'error' => {
  if (score === null || score < 0.5) return 'default';
  if (score < 0.8) return 'warning';
  return 'error';
};

// Определяем колонки для DataGrid
const columns: GridColDef<Anomaly>[] = [
  { field: 'id', headerName: 'ID', width: 90, type: 'number' },
  {
    field: 'order_id',
    headerName: 'Order ID',
    width: 250,
  },
  {
    field: 'order_item_id',
    headerName: 'Item ID',
    type: 'number',
    width: 100,
    align: 'right',
    headerAlign: 'right',
  },
  {
    field: 'detector_type',
    headerName: 'Detector',
    width: 150,
  },
  {
    field: 'anomaly_score',
    headerName: 'Score',
    type: 'number',
    width: 110,
    renderCell: (params: GridRenderCellParams<Anomaly, number | null>) => (
      params.value !== null && params.value !== undefined ? (
        <Chip 
          label={params.value.toFixed(4)}
          color={getScoreChipColor(params.value)}
          size="small"
          variant="outlined"
        />
      ) : (
        <Typography variant="caption" color="textSecondary">N/A</Typography>
      )
    ),
  },
  {
    field: 'detection_date',
    headerName: 'Detection Date',
    type: 'dateTime',
    width: 180,
    valueGetter: (value: string | null | undefined) => 
      value ? new Date(value) : null,
  },
  {
    field: 'actions',
    headerName: 'Details',
    width: 80,
    sortable: false,
    filterable: false,
    renderCell: (params: GridRenderCellParams<Anomaly>) => (
      <IconButton 
        size="small" 
        onClick={(e) => {
            e.stopPropagation(); // Prevent row click if clicking the button
            // No separate action needed here, row click handles selection
            // onAnomalySelect(params.row as Anomaly); // REMOVE THIS LINE
        }}
        aria-label={`View details for anomaly ${params.row.id}`}
      >
         <InfoOutlinedIcon fontSize="small" />
      </IconButton>
    ),
  },
];

const AnomalyTable: React.FC<AnomalyTableProps> = ({
  anomalies,
  loading,
  rowCount,
  paginationModel,
  onPaginationModelChange,
  onAnomalySelect,
}) => {
  return (
    <Box sx={{ height: 600, width: '100%' }}>
      <DataGrid
        rows={anomalies}
        columns={columns}
        loading={loading}
        // Пагинация на стороне сервера
        rowCount={rowCount} // TODO: Нужно получать общее количество с бэкенда
        pageSizeOptions={[10, 25, 50, 100]}
        paginationModel={paginationModel}
        paginationMode="server" // Включаем серверную пагинацию
        onPaginationModelChange={onPaginationModelChange}
        getRowId={(row) => row.id}
        autoHeight={false} 
        checkboxSelection={false} 
        disableRowSelectionOnClick
        slots={{ toolbar: GridToolbar }}
        slotProps={{
          toolbar: {
            showQuickFilter: true,
            printOptions: { disableToolbarButton: true },
            csvOptions: { disableToolbarButton: true },
          },
        }}
        sx={{ 
          '& .MuiDataGrid-cell:focus-within': {
            outline: 'none !important',
          },
          '& .MuiDataGrid-columnHeader:focus-within': {
            outline: 'none !important',
          },
          '& .MuiDataGrid-row': { cursor: 'pointer' },
        }}
      />
    </Box>
  );
};

export default AnomalyTable; 