import React from 'react';
import { Link, Outlet, useLocation, useMatches } from 'react-router-dom';
import {
  DesktopOutlined,
  FileTextOutlined,
  SettingOutlined,
  GoldOutlined,
  UserOutlined,
  TeamOutlined,
  AppstoreOutlined,
  DeploymentUnitOutlined,
  SolutionOutlined,
  EyeOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  ProfileOutlined,
  HomeOutlined,
  GlobalOutlined,
  MessageOutlined,
  TranslationOutlined,
} from '@ant-design/icons';
import { Layout, Menu, theme, Typography, Row, Col, Breadcrumb, Grid } from 'antd';
import type { AppRouteObject, RouteHandle } from '../App';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

type MenuItemAntD = {
  key: string;
  icon?: React.ReactNode;
  label: React.ReactNode;
  path?: string;
  children?: MenuItemAntD[];
};

function getItem(
  label: React.ReactNode,
  key: string,
  icon?: React.ReactNode,
  path?: string,
  children?: MenuItemAntD[],
): MenuItemAntD {
  return {
    key,
    icon,
    children,
    label: path && !children ? <Link to={path}>{label}</Link> : label,
    path,
  } as MenuItemAntD;
}

const menuItems: MenuItemAntD[] = [
  getItem('Dashboard', 'dashboard', <HomeOutlined />, '/'),
  getItem('Anomalies', 'anomalies', <EyeOutlined />, '/anomalies'),
  getItem('Multilevel System', 'multilevel', <LineChartOutlined />, undefined, [
    getItem('Status', 'multilevel-status', <DeploymentUnitOutlined />, '/multilevel/status'),
    getItem('Configuration', 'multilevel-config', <SettingOutlined />, '/multilevel/config'),
    getItem('Train All', 'multilevel-train', <ExperimentOutlined />, '/multilevel/train'),
    getItem('Detect All', 'multilevel-detect', <SolutionOutlined />, '/multilevel/detect'),
  ]),
  getItem('Single Model', 'single-model', <AppstoreOutlined />, undefined, [
    getItem('Train Model', 'train-model', <ExperimentOutlined />, '/model/train'),
    getItem('Detect Anomalies', 'model-detect', <SolutionOutlined />, '/model/detect'),
  ]),
  getItem('Data', 'data', <GoldOutlined />, undefined, [
    getItem('Orders', 'orders', <ProfileOutlined />, '/data/orders'),
    getItem('Products', 'products', <DesktopOutlined />, '/data/products'),
    getItem('Geolocation', 'geolocation', <GlobalOutlined />, '/data/geolocation'),
    getItem('Customers', 'customers', <UserOutlined />, '/data/customers'),
    getItem('Sellers', 'sellers', <TeamOutlined />, '/data/sellers'),
    getItem('Reviews', 'reviews', <MessageOutlined />, '/data/reviews'),
    getItem('Translations', 'translations', <TranslationOutlined />, '/data/translations'),
  ]),
  getItem('Settings', 'app-settings', <SettingOutlined />, '/settings'),
  getItem('Tasks', 'tasks', <FileTextOutlined />, '/tasks'),
];

const MainLayout: React.FC = () => {
  const location = useLocation();
  const matches = useMatches() as Array<{ id: string; pathname: string; params: any; data: any; handle: RouteHandle }>;
  const screens = Grid.useBreakpoint();

  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const getSelectedKeys = () => {
    let bestMatch = '';
    let bestMatchLength = 0;
    const findKeyRecursive = (items: MenuItemAntD[], currentPath: string) => {
      for (const item of items) {
        if (item.path && currentPath.startsWith(item.path)) {
          if (item.path.length > bestMatchLength) {
            bestMatch = item.key;
            bestMatchLength = item.path.length;
          } else if (item.path.length === bestMatchLength && currentPath === item.path) {
            bestMatch = item.key;
          }
        }
        if (item.children) {
          findKeyRecursive(item.children, currentPath);
        }
      }
    };
    findKeyRecursive(menuItems, location.pathname);
    return bestMatch ? [bestMatch] : (location.pathname === '/' ? ['dashboard'] : []);
  };

  const crumbs = matches
    .filter((match) => Boolean(match.handle?.crumb))
    .map((match, index, array) => {
      const crumbNode = match.handle.crumb!(match.data);
      return {
        key: match.id,
        title: index === array.length - 1 ? (typeof crumbNode === 'string' ? <span>{crumbNode}</span> : crumbNode) : crumbNode,
      };
    });
  
  const currentTitleMatch = matches.slice().reverse().find(match => Boolean(match.handle?.title));
  const pageTitle = currentTitleMatch?.handle?.title
    ? (typeof currentTitleMatch.handle.title === 'function' ? currentTitleMatch.handle.title() : currentTitleMatch.handle.title)
    : 'AnomaLens';

  const headerPaddingHorizontal = screens.md ? 20 : 10;
  const contentPaddingHorizontal = screens.md ? 24 : 16;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', padding: `0 ${headerPaddingHorizontal}px`, background: '#fff', borderBottom: '1px solid #f0f0f0', position: 'sticky', top: 0, zIndex:10 }}>
        <Row justify="space-between" align="middle" style={{ width: '100%' }}>
          <Col flex="none">
            <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
              <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>AnomaLens</Link>
              </Title>
            </div>
          </Col>
          <Col flex="auto">
            <Menu
              theme="light"
              mode="horizontal"
              selectedKeys={getSelectedKeys()}
              items={menuItems}
              style={{ lineHeight: '62px', borderBottom: 'none', justifyContent: 'center' }}
            />
          </Col>
          </Row>
      </Header>
      <Content style={{ 
        padding: `0 ${contentPaddingHorizontal}px`, 
        marginTop: '24px',
        display: 'flex',
        flexDirection: 'column' 
      }}>
        <Breadcrumb items={crumbs} style={{ marginBottom: '16px' }} />
        <Title level={2} style={{marginBottom: 24}}>{pageTitle}</Title>
        <div
          style={{
            padding: 24,
            background: colorBgContainer,
            borderRadius: borderRadiusLG,
            flex: 1, 
          }}
        >
          <Outlet />
        </div>
      </Content>
      <Footer style={{ textAlign: 'center', padding: '24px 0' }}>
        AnomaLens Frontend ©{new Date().getFullYear()}
      </Footer>
    </Layout>
  );
};

export default MainLayout;