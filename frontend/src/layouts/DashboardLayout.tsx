import React from 'react';
import { Link as RouterLink, Outlet, useLocation } from 'react-router-dom';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import CssBaseline from '@mui/material/CssBaseline';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import DashboardIcon from '@mui/icons-material/Dashboard';
import HistoryIcon from '@mui/icons-material/History';
import PendingActionsIcon from '@mui/icons-material/PendingActions';
import LayersIcon from '@mui/icons-material/Layers';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import { alpha, Theme } from '@mui/material/styles';

interface NavItem {
  text: string;
  path: string;
  icon: React.ReactElement;
}

const navItems: NavItem[] = [
  {
    text: 'Обзор моделей',
    path: '/',
    icon: <DashboardIcon fontSize="small" />,
  },
  {
    text: 'История аномалий',
    path: '/history',
    icon: <HistoryIcon fontSize="small" />,
  },
  {
    text: 'Многоуровневая система',
    path: '/multilevel',
    icon: <LayersIcon fontSize="small" />,
  },
  {
    text: 'Статус Задач',
    path: '/tasks',
    icon: <PendingActionsIcon fontSize="small" />,
  },
];

export default function DashboardLayout() {
  const location = useLocation();

  return (
    <Box sx={{ display: 'block' }}>
      <CssBaseline />
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h6" noWrap component="div" sx={{ mr: 4, color: 'text.primary' }}>
            AnomaLens
          </Typography>

          <Stack direction="row" spacing={1.5}>
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Button
                  key={item.text}
                  component={RouterLink}
                  to={item.path}
                  startIcon={item.icon}
                  sx={{
                    color: isActive ? 'primary.main' : 'text.secondary',
                    backgroundColor: isActive ? (theme: Theme) => alpha(theme.palette.primary.main, 0.1) : 'transparent',
                    fontWeight: isActive ? 600 : 500,
                    borderRadius: 2,
                    px: 2,
                    py: 0.75,
                    minWidth: 'auto',
                    textTransform: 'none',
                    transition: (theme: Theme) => theme.transitions.create(['background-color', 'color'], {
                         duration: theme.transitions.duration.short,
                    }),
                    '&:hover': {
                      backgroundColor: isActive ? (theme: Theme) => alpha(theme.palette.primary.main, 0.15) : (theme: Theme) => alpha(theme.palette.text.secondary, 0.05),
                      color: isActive ? 'primary.main' : 'text.primary',
                    }
                  }}
                >
                  {item.text}
                </Button>
              );
            })}
          </Stack>
        </Toolbar>
      </AppBar>

      <Box component="main" sx={{ flexGrow: 1, p: 3, bgcolor: 'background.default' }}>
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
} 