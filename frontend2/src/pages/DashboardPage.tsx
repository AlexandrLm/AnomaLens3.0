import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Row, Col, Card, Statistic, Spin, Alert, Tag, Table, Empty, Tooltip, List, Progress, Divider, Space, Button } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, PieChart, Pie, Cell, Sector, ResponsiveContainer } from 'recharts';
import HighlightAndZoomLineChart from '../components/common/HighlightAndZoomLineChart';
import { ApiOutlined, BugOutlined, DeploymentUnitOutlined, CheckCircleOutlined, CloseCircleOutlined, WarningOutlined, EyeOutlined, ExclamationCircleOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { fetchApiStatus } from '../services/apiStatusService';
import { fetchAnomalies } from '../services/anomalyService';
import { fetchMultilevelStatus } from '../services/multilevelService';
import type { ApiStatusResponse, RootAnomalySchema, MultilevelStatus, DetectorStatus, DetectorLevelStatus } from '../types/api';
import dayjs from 'dayjs';
import { Link } from 'react-router-dom';
import { notification } from 'antd';

const { Title, Paragraph, Text } = Typography;

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const renderActiveShape = (props: any) => {
  const RADIAN = Math.PI / 180;
  const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value } = props;
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(-RADIAN * midAngle);
  const sx = cx + (outerRadius + 10) * cos;
  const sy = cy + (outerRadius + 10) * sin;
  const mx = cx + (outerRadius + 30) * cos;
  const my = cy + (outerRadius + 30) * sin;
  const ex = mx + (cos >= 0 ? 1 : -1) * 22;
  const ey = my;
  const textAnchor = cos >= 0 ? 'start' : 'end';

  return (
    <g>
      <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>
        {payload.name}
      </text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
      <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
      <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
      <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333">{`${value} (${(percent * 100).toFixed(0)}%)`}</text>
    </g>
  );
};

const MultilevelLevelStatusCard: React.FC<{ levelName: string; levelData?: DetectorLevelStatus; loading: boolean; error?: string | null }> = ({
  levelName,
  levelData,
  loading,
  error
}) => {
  if (loading) {
    return <Card size="small" title={levelName}><Spin size="small" /></Card>;
  }
  if (error) {
     return <Card size="small" title={levelName} extra={<Tooltip title={error}><WarningOutlined style={{color: 'red'}}/></Tooltip>}><Text type="danger">Ошибка</Text></Card>;
  }
  if (!levelData || Object.keys(levelData).length === 0) {
    return <Card size="small" title={levelName}><Text type="secondary">Нет детекторов</Text></Card>;
  }

  let trainedCount = 0;
  let loadableCount = 0;
  let problematicDetectors: string[] = [];
  const totalDetectors = Object.keys(levelData).length;

  Object.entries(levelData).forEach(([name, status]) => {
    if (status.is_trained) trainedCount++;
    if (status.can_load) loadableCount++;
    if (!status.is_trained || !status.can_load || status.error_message) {
      problematicDetectors.push(name);
    }
  });

  const overallOk = trainedCount === totalDetectors && loadableCount === totalDetectors && problematicDetectors.length === 0;
  const someProblems = problematicDetectors.length > 0;

  const getStatusIndicator = () => {
    if (overallOk) return <CheckCircleOutlined style={{ color: 'green' }} />;
    if (someProblems) return <ExclamationCircleOutlined style={{ color: 'orange' }} />;
    return <InfoCircleOutlined style={{ color: 'blue' }} />;
  };

  return (
    <Card 
        size="small" 
        title={<Space>{getStatusIndicator()} {levelName}</Space>}
        extra={<Text type="secondary">{`${loadableCount} / ${totalDetectors} готовы`}</Text>}
    >
      <Progress 
        percent={totalDetectors > 0 ? (loadableCount / totalDetectors) * 100 : 0} 
        status={overallOk ? 'success' : (someProblems ? 'exception' : 'normal')} 
        size="small" 
        showInfo={false}
        style={{ marginBottom: problematicDetectors.length > 0 ? 8 : 0 }}
      />
      {problematicDetectors.length > 0 && (
        <Tooltip title={`Проблемные детекторы: ${problematicDetectors.join(', ')}`}>
            <Text type="danger" style={{fontSize: '12px', display: 'block', textAlign: 'center'}}>
                {problematicDetectors.length} {problematicDetectors.length === 1 ? 'детектор' : (problematicDetectors.length < 5 ? 'детектора' : 'детекторов')} требуют внимания
            </Text>
        </Tooltip>
      )}
       <Link to="/multilevel/status" style={{fontSize: '12px', display: 'block', textAlign: 'right', marginTop: '8px'}}>
            Подробнее...
       </Link>
    </Card>
  );
};

const DashboardPage: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<ApiStatusResponse | null>(null);
  const [totalAnomalies, setTotalAnomalies] = useState<number | null>(null);
  const [recentAnomalies, setRecentAnomalies] = useState<RootAnomalySchema[]>([]);
  const [anomaliesByDay, setAnomaliesByDay] = useState<Array<{ date: string; count: number }>>([]);
  const [anomaliesByType, setAnomaliesByType] = useState<Array<{ name: string; value: number }>>([]);
  const [anomaliesByScore, setAnomaliesByScore] = useState<Array<{ range: string; count: number }>>([]);
  const [multilevelStatus, setMultilevelStatus] = useState<MultilevelStatus | null>(null);
  
  const [loading, setLoading] = useState({
    api: true,
    anomalies: true,
    multilevel: true,
  });
  const [error, setError] = useState<{
    api: string | null;
    anomalies: string | null;
    multilevel: string | null;
  }>({
    api: null,
    anomalies: null,
    multilevel: null,
  });

  const [activePieIndex, setActivePieIndex] = useState(0);
  const onPieEnter = useCallback((_: any, index: number) => {
    setActivePieIndex(index);
  }, [setActivePieIndex]);

  const loadDashboardData = useCallback(async () => {
    setLoading(prev => ({ ...prev, api: true }));
    try {
      const statusData = await fetchApiStatus();
      setApiStatus(statusData);
      setError(prev => ({ ...prev, api: null }));
    } catch (err: any) {
      setError(prev => ({ ...prev, api: err.message }));
      notification.error({ message: 'Ошибка загрузки статуса API', description: err.message });
    } finally {
      setLoading(prev => ({ ...prev, api: false }));
    }

    setLoading(prev => ({ ...prev, anomalies: true }));
    try {
      const endDate = dayjs();
      const startDate30Days = endDate.subtract(30, 'day');

      const [anomaliesData, anomaliesForCharts] = await Promise.all([
        fetchAnomalies({ limit: 5, skip: 0 }),
        fetchAnomalies({ limit: 1000, start_date: startDate30Days.toISOString(), end_date: endDate.toISOString() })
      ]);
      
      setTotalAnomalies(anomaliesData.total);
      setRecentAnomalies(anomaliesData.items);

      const dailyCounts: Record<string, number> = {};
      anomaliesForCharts.items.forEach(anomaly => {
        const date = dayjs(anomaly.detection_date).format('YYYY-MM-DD');
        dailyCounts[date] = (dailyCounts[date] || 0) + 1;
      });
      const sortedDailyData = Object.entries(dailyCounts)
        .map(([date, count]) => ({ date, count }))
        .sort((a, b) => dayjs(a.date).valueOf() - dayjs(b.date).valueOf());
      setAnomaliesByDay(sortedDailyData);

      const typeCounts: Record<string, number> = {};
      anomaliesForCharts.items.forEach(anomaly => {
        typeCounts[anomaly.detector_type] = (typeCounts[anomaly.detector_type] || 0) + 1;
      });
      setAnomaliesByType(Object.entries(typeCounts).map(([name, value]) => ({ name, value })));
      
      if (anomaliesForCharts.items.length > 0) {
        const scoreCounts: Record<string, number> = {};
        // Инициализируем 100 корзин + 'N/A'
        for (let i = 0; i < 100; i++) {
          const lowerBound = i / 100;
          const upperBound = (i + 1) / 100;
          scoreCounts[`${lowerBound.toFixed(2)}-${upperBound.toFixed(2)}`] = 0;
        }
        scoreCounts['N/A'] = 0;

        anomaliesForCharts.items.forEach(anomaly => {
          const score = anomaly.anomaly_score;
          if (score === null || score === undefined) {
            scoreCounts['N/A']++;
          } else {
            let binIndex = Math.floor(score * 100);
            if (binIndex >= 100) binIndex = 99; 
            if (binIndex < 0) binIndex = 0; 

            const lowerBound = binIndex / 100;
            const upperBound = (binIndex + 1) / 100;
            const rangeKey = `${lowerBound.toFixed(2)}-${upperBound.toFixed(2)}`;
            
            if (scoreCounts.hasOwnProperty(rangeKey)) {
                 if (score >= lowerBound && (score < upperBound || (binIndex === 99 && score <= upperBound))) {
                    scoreCounts[rangeKey]++;
                } else if (binIndex === 99 && score === 1.0) { 
                     scoreCounts['0.99-1.00']++;
                } 
            }
          }
        });
        setAnomaliesByScore(Object.entries(scoreCounts)
          .filter(([range, _]) => range !== 'N/A' || scoreCounts['N/A'] > 0) 
          .map(([range, count]) => ({ range, count }))
          .sort((a,b) => {
              if(a.range === 'N/A') return 1;
              if(b.range === 'N/A') return -1;
              return parseFloat(a.range.split('-')[0]) - parseFloat(b.range.split('-')[0]);
          })
        );
      }

      setError(prev => ({ ...prev, anomalies: null }));
    } catch (err: any) {
      setError(prev => ({ ...prev, anomalies: err.message }));
      notification.error({ message: 'Ошибка загрузки данных аномалий', description: err.message });
    } finally {
      setLoading(prev => ({ ...prev, anomalies: false }));
    }

    setLoading(prev => ({ ...prev, multilevel: true }));
    try {
      const mlStatusData = await fetchMultilevelStatus();
      setMultilevelStatus(mlStatusData);
      setError(prev => ({ ...prev, multilevel: null }));
    } catch (err: any) {
      setError(prev => ({ ...prev, multilevel: err.message }));
      if ((err.message as string).includes("503")) {
        notification.warning({ message: 'Многоуровневая система', description: 'Сервис инициализируется или временно недоступен.' });
      } else {
        notification.error({ message: 'Ошибка загрузки статуса многоуровневой системы', description: err.message });
      }
    } finally {
      setLoading(prev => ({ ...prev, multilevel: false }));
    }
  }, []);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const recentAnomaliesColumns: ColumnsType<RootAnomalySchema> = [
    { title: 'ID', dataIndex: 'id', key: 'id', render: (id: number, record: RootAnomalySchema) => <Link to={`/anomalies?anomaly_id_highlight=${id}`}>{id}</Link>},
    { title: 'Дата', dataIndex: 'detection_date', key: 'detection_date', render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm') },
    { title: 'Тип', dataIndex: 'detector_type', key: 'detector_type' },
    { title: 'Оценка', dataIndex: 'anomaly_score', key: 'anomaly_score', render: (score?: number | null) => score?.toFixed(2) || 'N/A' },
    { title: 'Действие', key: 'action', render: (_: any, record: RootAnomalySchema) => <Link to={`/anomalies?anomaly_id_view=${record.id}`}><Tooltip title="Просмотреть детали"><EyeOutlined /></Tooltip></Link>}
  ];

  // Определяем состояния для карточки статуса API
  const isOverallApiLoading = loading.api || loading.anomalies;
  // Приоритет у ошибки загрузки аномалий, затем у ошибки API, если данные по аномалиям еще не определены
  const displayErrorCatched = error.anomalies || (!totalAnomalies && error.api);
  const displayErrorText = error.anomalies ? String(error.anomalies) : (error.api ? String(error.api) : 'Неизвестная ошибка');
  
  const isPositiveStatus = totalAnomalies !== null && !error.anomalies;

  let statusValueForStatistic: string;
  if (isOverallApiLoading) {
    statusValueForStatistic = 'Загрузка...';
  } else if (displayErrorCatched) {
    statusValueForStatistic = `Ошибка: ${displayErrorText.substring(0,50)}`;
  } else if (isPositiveStatus) {
    statusValueForStatistic = apiStatus && !error.api ? `API: ${apiStatus.environment} / Данные: OK` : 'Данные загружены';
  } else if (apiStatus && !error.api) {
    statusValueForStatistic = `API: ${apiStatus.environment}`;
  } else {
    statusValueForStatistic = 'Статус не определен';
  }

  return (
    <Spin spinning={loading.api || loading.anomalies || loading.multilevel} tip="Загрузка данных дашборда...">
      <Paragraph style={{marginBottom: 24}}>
        Добро пожаловать на дашборд AnomaLens. Здесь вы найдете обзор ключевых показателей системы.
      </Paragraph>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Статус API"
              value={statusValueForStatistic}
              prefix={isOverallApiLoading ? <Spin size="small" /> : <ApiOutlined />}
              valueStyle={{ 
                color: displayErrorCatched ? '#cf1322' : isPositiveStatus ? '#3f8600' : undefined 
              }}
              suffix={
                displayErrorCatched ? <Tooltip title={displayErrorText}><CloseCircleOutlined style={{color: '#cf1322'}}/></Tooltip> : 
                isPositiveStatus ? <CheckCircleOutlined /> : 
                null
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8}>
          <Card>
            <Statistic
              title="Всего аномалий"
              value={totalAnomalies ?? 'Загрузка...'}
              prefix={<BugOutlined />}
              valueStyle={{ color: error.anomalies ? '#cf1322' : undefined }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8}>
            <Card>
                <Statistic
                title="Задачи (пример)"
                value={"Нет активных"}
                prefix={<DeploymentUnitOutlined />}
                />
            </Card>
        </Col>
      </Row>

      <Title level={4} style={{marginTop: 24, marginBottom: 16}}>Статус Многоуровневой Системы Детекции</Title>
      {loading.multilevel && <Card><Spin tip="Загрузка статуса ML системы..."/></Card>}
      {error.multilevel && !loading.multilevel && 
        <Alert 
            message="Ошибка загрузки статуса многоуровневой системы" 
            description={error.multilevel} 
            type="error" showIcon 
            action={
                <Button size="small" danger onClick={loadDashboardData}>
                    Попробовать снова
                </Button>
            }
            style={{ marginBottom: 24 }}
        />
      }
      {!loading.multilevel && !error.multilevel && multilevelStatus && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          {Object.entries(multilevelStatus).map(([levelKey, levelData]) => {
            if (!levelKey.endsWith('_level') && levelKey !== 'transaction_level' && levelKey !== 'behavior_level' && levelKey !== 'time_series_level') {
                return null;
            }
            const levelNameMapping: Record<string, string> = {
                transaction_level: "Транзакционный",
                behavior_level: "Поведенческий",
                time_series_level: "Временных рядов"
            };
            const displayName = levelNameMapping[levelKey] || levelKey.replace('_level', '').replace(/^\w/, c => c.toUpperCase());

            return (
              <Col xs={24} sm={12} md={8} key={levelKey}>
                <MultilevelLevelStatusCard
                  levelName={displayName}
                  levelData={levelData as DetectorLevelStatus}
                  loading={false} 
                  error={null}
                />
              </Col>
            );
          })}
        </Row>
      )}
      {!loading.multilevel && !error.multilevel && !multilevelStatus && (
        <Card style={{ marginBottom: 24 }}><Empty description="Данные о статусе многоуровневой системы отсутствуют или сервис недоступен." /></Card>
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Динамика обнаружения аномалий (за последние 30 дней)">
            {loading.anomalies && <Spin />}
            {!loading.anomalies && error.anomalies && <Alert message="Ошибка загрузки данных для графика" type="error" />}
            {!loading.anomalies && !error.anomalies && anomaliesByDay.length > 0 && (
              <HighlightAndZoomLineChart
                data={anomaliesByDay}
                lines={[
                  {
                    key: 'anomaliesCount',
                    name: 'Кол-во аномалий',
                    dataKey: 'count',
                    stroke: '#8884d8',
                    activeDot: { r: 8 },
                    type: 'monotone'
                  }
                ]}
                xAxisProps={{
                  dataKey: 'date',
                  tickFormatter: (tick) => dayjs(tick).format('DD.MM')
                }}
                yAxisProps={{
                  tickFormatter: (value: any) => typeof value === 'number' ? value.toFixed(2) : value
                }}
                cartesianGridProps={{
                  strokeDasharray: '3 3'
                }}
                tooltipProps={{
                  formatter: (value: any, name: string) => [
                    typeof value === 'number' ? value.toFixed(2) : value,
                    name
                  ]
                }}
                legendProps={{
                  // настройки для легенды
                }}
                height={300} // Высота для ResponsiveContainer
                brushProps={{
                  dataKey: 'date',
                  height: 30,
                  stroke: '#8884d8'
                }}
              />
            )}
             {!loading.anomalies && !error.anomalies && anomaliesByDay.length === 0 && <Empty description="Нет данных об аномалиях за период"/>}
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Распределение аномалий по типам детекторов">
            {loading.anomalies && <Spin />}
            {!loading.anomalies && error.anomalies && <Alert message="Ошибка загрузки данных для графика" type="error" />}
            {!loading.anomalies && !error.anomalies && anomaliesByType.length > 0 && (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    activeIndex={activePieIndex}
                    activeShape={renderActiveShape}
                    data={anomaliesByType}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    onMouseEnter={onPieEnter}
                  >
                    {anomaliesByType.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}
            {!loading.anomalies && !error.anomalies && anomaliesByType.length === 0 && <Empty description="Нет данных о типах аномалий"/>}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Распределение аномалий по оценке">
            {loading.anomalies && <Spin />}
            {!loading.anomalies && error.anomalies && <Alert message="Ошибка загрузки данных для графика" type="error" />}
            {!loading.anomalies && !error.anomalies && anomaliesByScore.length > 0 && (
              <HighlightAndZoomLineChart
                data={anomaliesByScore}
                lines={[
                  {
                    key: 'anomaliesByScoreCount',
                    name: 'Кол-во аномалий',
                    dataKey: 'count',
                    stroke: '#82ca9d',
                    type: 'monotone'
                  }
                ]}
                xAxisProps={{
                  dataKey: 'range'
                }}
                yAxisProps={{
                  tickFormatter: (value: any) => typeof value === 'number' ? value.toFixed(2) : value
                }}
                cartesianGridProps={{
                  strokeDasharray: '3 3'
                }}
                tooltipProps={{
                  formatter: (value: any, name: string) => [
                    typeof value === 'number' ? value.toFixed(2) : value,
                    name
                  ]
                }}
                legendProps={{
                  // настройки для легенды, если нужны
                }}
                height={300} // Высота для ResponsiveContainer
                brushProps={{
                  dataKey: 'range',
                  height: 30,
                  stroke: '#82ca9d'
                }}
              />
            )}
            {!loading.anomalies && !error.anomalies && anomaliesByScore.length === 0 && <Empty description="Нет данных для распределения по оценкам"/>}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <Card title="Последние обнаруженные аномалии">
            {loading.anomalies && <Spin />}
            {!loading.anomalies && error.anomalies && <Alert message="Ошибка загрузки недавних аномалий" type="error" />}
            {!loading.anomalies && !error.anomalies && (
              <Table
                columns={recentAnomaliesColumns}
                dataSource={recentAnomalies}
                rowKey="id"
                pagination={false}
                size="small"
                locale={{ emptyText: <Empty description="Нет недавних аномалий" /> }}
              />
            )}
          </Card>
        </Col>
      </Row>
    </Spin>
  );
};

// HMR refresh trigger
export default DashboardPage;