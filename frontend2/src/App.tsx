import React from 'react';
import { createBrowserRouter, RouterProvider, Link, Outlet } from 'react-router-dom';
import type { RouteObject } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import DashboardPage from './pages/DashboardPage';
import AnomaliesPage from './pages/AnomaliesPage';
import NotFoundPage from './pages/NotFoundPage';
import MultilevelStatusPage from './pages/MultilevelStatusPage';
import MultilevelConfigPage from './pages/MultilevelConfigPage';
import MultilevelTrainPage from './pages/MultilevelTrainPage';
import MultilevelDetectPage from './pages/MultilevelDetectPage';
import ModelTrainPage from './pages/ModelTrainPage';
import ModelDetectPage from './pages/ModelDetectPage';
import OrdersPage from './pages/OrdersPage';
import ProductsPage from './pages/ProductsPage';
import AppSettingsPage from './pages/AppSettingsPage';
import GeolocationPage from './pages/GeolocationPage';
import CustomersPage from './pages/CustomersPage';
import SellersPage from './pages/SellersPage';
import ReviewsPage from './pages/ReviewsPage';
import TranslationsPage from './pages/TranslationsPage';
import TasksPage from './pages/TasksPage';

// Определяем тип для handle, чтобы TypeScript был доволен
export interface RouteHandle {
  crumb?: (data?: any) => React.ReactNode; // data - необязательные данные из loader/action
  title?: string | (() => string);
}

// Определяем тип для наших объектов маршрутов с handle
export interface AppRouteObject extends Omit<RouteObject, 'handle' | 'children'> {
  handle?: RouteHandle;
  children?: AppRouteObject[];
}

const routeConfig: AppRouteObject[] = [
  {
    path: '/',
    element: <MainLayout />,
    handle: {
      crumb: () => <Link to="/">Home</Link>,
      title: 'Home'
    },
    children: [
      {
        index: true,
        element: <DashboardPage />,
        handle: {
          crumb: () => <span>Dashboard</span>,
          title: 'Dashboard'
        },
      },
      {
        path: 'anomalies',
        element: <AnomaliesPage />,
        handle: {
          crumb: () => <Link to="/anomalies">Anomalies</Link>,
          title: 'Anomalies'
        },
      },
      {
        path: 'multilevel',
        element: <Outlet />, // Родительский маршрут для группы, рендерит дочерние
        handle: {
          // Ссылка на первую дочернюю страницу или просто текст
          crumb: () => <Link to="/multilevel/status">Multilevel System</Link>,
          title: 'Multilevel System'
        },
        children: [
          {
            path: 'status',
            element: <MultilevelStatusPage />,
            handle: {
              crumb: () => <span>Status</span>, // Последний элемент не ссылка
              title: 'System Status'
            },
          },
          {
            path: 'config',
            element: <MultilevelConfigPage />,
            handle: {
              crumb: () => <span>Configuration</span>,
              title: 'System Configuration'
            },
          },
          {
            path: 'train',
            element: <MultilevelTrainPage />,
            handle: {
              crumb: () => <span>Train All</span>,
              title: 'Train All Models'
            },
          },
          {
            path: 'detect',
            element: <MultilevelDetectPage />,
            handle: {
              crumb: () => <span>Detect All</span>,
              title: 'Detect All Anomalies'
            },
          },
        ],
      },
      {
        path: 'model',
        element: <Outlet />,
        handle: {
          crumb: () => <Link to="/model/train">Single Model</Link>,
          title: 'Single Model Operations'
        },
        children: [
          {
            path: 'train',
            element: <ModelTrainPage />,
            handle: {
              crumb: () => <span>Train Model</span>,
              title: 'Train Model'
            },
          },
          {
            path: 'detect', // Изменено с detect-anomalies для соответствия пути меню
            element: <ModelDetectPage />,
            handle: {
              crumb: () => <span>Detect Anomalies</span>,
              title: 'Detect Anomalies'
            },
          },
        ],
      },
      {
        path: 'data',
        element: <Outlet />,
        handle: {
          crumb: () => <Link to="/data/orders">Data</Link>,
          title: 'Data Explorer'
        },
        children: [
          {
            path: 'orders',
            element: <OrdersPage />,
            handle: {
              crumb: () => <span>Orders</span>,
              title: 'Orders'
            },
          },
          {
            path: 'products',
            element: <ProductsPage />,
            handle: {
              crumb: () => <span>Products</span>,
              title: 'Products'
            },
          },
          {
            path: 'geolocation',
            element: <GeolocationPage />,
            handle: {
              crumb: () => <span>Geolocation</span>,
              title: 'Geolocation Data'
            },
          },
          {
            path: 'customers',
            element: <CustomersPage />,
            handle: {
              crumb: () => <span>Customers</span>,
              title: 'Customers Data'
            },
          },
          {
            path: 'sellers',
            element: <SellersPage />,
            handle: {
              crumb: () => <span>Sellers</span>,
              title: 'Sellers Data'
            },
          },
          {
            path: 'reviews',
            element: <ReviewsPage />,
            handle: {
              crumb: () => <span>Reviews</span>,
              title: 'Product Reviews'
            },
          },
          {
            path: 'translations',
            element: <TranslationsPage />,
            handle: {
              crumb: () => <span>Translations</span>,
              title: 'Category Translations'
            },
          },
          // Добавьте customers, sellers и т.д. по аналогии
        ],
      },
      {
        path: 'settings', // Совпадает с ключом меню 'app-settings'
        element: <AppSettingsPage />,
        handle: {
          crumb: () => <span>Application Settings</span>,
          title: 'Application Settings'
        },
      },
      { path: 'tasks', element: <TasksPage />, handle: { crumb: () => <span>Tasks</span>, title: 'Task Status'} },
      { path: '*', element: <NotFoundPage /> }, // Страница не найдена
    ],
  },
];

// Создаем data router
const router = createBrowserRouter(routeConfig as RouteObject[]);

// Убираем AppContent, App теперь напрямую использует RouterProvider
function App() {
  return <RouterProvider router={router} />;
}

export default App;