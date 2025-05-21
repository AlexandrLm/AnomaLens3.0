import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme/theme';
import DashboardLayout from './layouts/DashboardLayout';

import ModelOverviewPage from './pages/ModelOverviewPage';
import AnomalyHistoryPage from './pages/AnomalyHistoryPage';
import MultilevelSystemPage from './pages/MultilevelSystemPage';
import TaskStatusPage from './pages/TaskStatusPage';
import NotFoundPage from './pages/NotFoundPage'; 

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<ModelOverviewPage />} /> 
            <Route path="history" element={<AnomalyHistoryPage />} />
            <Route path="multilevel" element={<MultilevelSystemPage />} />
            <Route path="tasks" element={<TaskStatusPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
