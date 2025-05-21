import React, { useState, useEffect, useCallback } from 'react';
import {
  Typography,
  Table,
  Button,
  Modal,
  Form,
  Input,
  DatePicker,
  Space,
  Tag,
  Alert,
  InputNumber,
  Select,
  Card,
  Spin,
  Row,
  Col,
  notification,
  Tooltip,
  Popconfirm,
  Segmented,
  Pagination,
  Descriptions,
  Collapse,
  List,
  Timeline
} from 'antd';
import { EyeOutlined, DeleteOutlined, ExperimentOutlined, ReloadOutlined, SearchOutlined, AppstoreOutlined, TableOutlined, WarningOutlined, ShoppingCartOutlined, UserOutlined, CheckCircleOutlined, ClockCircleOutlined, SyncOutlined, DeploymentUnitOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import type {
  RootAnomalySchema,
  FetchAnomaliesParams,
  AnomalyDetails as AnomalyDetailsType,
  AnomalyLevelScores as AnomalyLevelScoresType,
  OrderSchema,
  OrderItemSchema,
  DetectorExplanationEntry,
  ContributingDetectorsExplanations,
  ProductSchema,
  SellerSchema
} from '../types/api';
import {
  fetchAnomalies,
  deleteAnomaly,
  deleteAllAnomalies,
  fetchLlmExplanation
} from '../services/anomalyService';
import { fetchOrderById } from '../services/orderService';
import { fetchProductById } from '../services/productService';
import { fetchSellerById } from '../services/sellerService';
import dayjs from 'dayjs';
import AnomalyCard from '../components/AnomalyCard';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import AnomalyRelationshipGraph from '../components/common/AnomalyRelationshipGraph';

const { Title, Paragraph, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { Panel } = Collapse;

const DETECTOR_TYPES = ["isolation_forest", "statistical", "rules_based", "autoencoder", "llm_heuristic"];

const orderStatusColors: Record<string, string> = {
  delivered: 'success',
  shipped: 'processing',
  canceled: 'error',
  invoiced: 'cyan',
  processing: 'blue',
  created: 'default',
  unavailable: 'magenta',
  approved: 'green',
};

const formatDate = (dateString?: string | null) => {
  return dateString ? dayjs(dateString).format('YYYY-MM-DD HH:mm:ss') : 'N/A';
};

const AnomaliesPage: React.FC = () => {
  const [anomalies, setAnomalies] = useState<RootAnomalySchema[]>([]);
  const [totalAnomalies, setTotalAnomalies] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [filters, setFilters] = useState<Omit<FetchAnomaliesParams, 'skip' | 'limit'>>({});
  const [formFilters] = Form.useForm();

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedAnomaly, setSelectedAnomaly] = useState<RootAnomalySchema | null>(null);

  const [llmExplanation, setLlmExplanation] = useState<string | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'card'>('table');
  const [deleteAllLoading, setDeleteAllLoading] = useState(false);

  const [selectedAnomalyOrderDetails, setSelectedAnomalyOrderDetails] = useState<OrderSchema | null>(null);
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState<string | null>(null);

  const [selectedAnomalyProductDetails, setSelectedAnomalyProductDetails] = useState<ProductSchema | null>(null);
  const [productLoading, setProductLoading] = useState(false);
  const [productError, setProductError] = useState<string | null>(null);

  const [selectedAnomalySellerDetails, setSelectedAnomalySellerDetails] = useState<SellerSchema | null>(null);
  const [sellerLoading, setSellerLoading] = useState(false);
  const [sellerError, setSellerError] = useState<string | null>(null);

  const loadAnomalies = useCallback(async (
    currentPagination: TablePaginationConfig,
    currentFilters: Omit<FetchAnomaliesParams, 'skip' | 'limit'>
  ) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchAnomaliesParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
        ...currentFilters,
      };
      const data = await fetchAnomalies(params);
      setAnomalies(data.items);
      setTotalAnomalies(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize }));
    } catch (err) {
      setError((err as Error).message);
      notification.error({
        message: 'Ошибка загрузки аномалий',
        description: (err as Error).message,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAnomalies(pagination, filters);
  }, [loadAnomalies, pagination.current, pagination.pageSize, filters]);

  const handleTableChange = (
    newPagination: TablePaginationConfig,
  ) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const handleFilterSubmit = (values: any) => {
    const newFilters: Omit<FetchAnomaliesParams, 'skip' | 'limit'> = {
      start_date: values.dateRange?.[0]?.toISOString(),
      end_date: values.dateRange?.[1]?.toISOString(),
      min_score: values.min_score,
      max_score: values.max_score,
      detector_type: values.detector_type,
    };
    Object.keys(newFilters).forEach(key => newFilters[key as keyof typeof newFilters] === undefined && delete newFilters[key as keyof typeof newFilters]);

    setFilters(newFilters);
    setPagination(prev => ({ ...prev, current: 1 })); 
  };
  
  const resetFilters = () => {
    formFilters.resetFields();
    setFilters({});
    setPagination(prev => ({ ...prev, current: 1 }));
  };

  const fetchAndSetOrderDetails = async (orderId: string) => {
    setOrderLoading(true);
    setOrderError(null);
    setSelectedAnomalyOrderDetails(null);
    try {
      const orderData = await fetchOrderById(orderId);
      setSelectedAnomalyOrderDetails(orderData);
    } catch (err) {
      setOrderError((err as Error).message);
      notification.error({
          message: `Ошибка загрузки деталей заказа ${orderId}`,
          description: (err as Error).message
      });
    } finally {
      setOrderLoading(false);
    }
  };

  const fetchAndSetProductDetails = async (productId: string) => {
    setProductLoading(true);
    setProductError(null);
    setSelectedAnomalyProductDetails(null);
    try {
      const productData = await fetchProductById(productId);
      setSelectedAnomalyProductDetails(productData);
    } catch (err) {
      setProductError((err as Error).message);
      notification.error({
        message: `Ошибка загрузки деталей продукта ${productId}`,
        description: (err as Error).message,
      });
    } finally {
      setProductLoading(false);
    }
  };

  const fetchAndSetSellerDetails = async (sellerId: string) => {
    setSellerLoading(true);
    setSellerError(null);
    setSelectedAnomalySellerDetails(null);
    try {
      const sellerData = await fetchSellerById(sellerId);
      setSelectedAnomalySellerDetails(sellerData);
    } catch (err) {
      setSellerError((err as Error).message);
      notification.error({
        message: `Ошибка загрузки деталей продавца ${sellerId}`,
        description: (err as Error).message,
      });
    } finally {
      setSellerLoading(false);
    }
  };

  const showViewModal = (record: RootAnomalySchema) => {
    setSelectedAnomaly(record);
    setLlmExplanation(null); 
    setLlmLoading(false);
    setSelectedAnomalyOrderDetails(null);
    setOrderError(null);
    if (record.order_id) {
      fetchAndSetOrderDetails(record.order_id);
    }
    if (record.details?.product_id) {
      fetchAndSetProductDetails(record.details.product_id as string);
    }
    if (record.details?.seller_id) {
      fetchAndSetSellerDetails(record.details.seller_id as string);
    }
    setIsModalVisible(true);
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedAnomaly(null);
    setSelectedAnomalyOrderDetails(null);
    setOrderLoading(false);
    setOrderError(null);
    setLlmExplanation(null);
    setLlmLoading(false);
    setSelectedAnomalyProductDetails(null);
    setProductLoading(false);
    setProductError(null);
    setSelectedAnomalySellerDetails(null);
    setSellerLoading(false);
    setSellerError(null);
  };

  const handleDeleteAnomaly = async (id: number) => {
    try {
      setLoading(true); 
      await deleteAnomaly(id);
      notification.success({ message: `Аномалия ID ${id} успешно удалена` });
      const newTotal = totalAnomalies -1;
      const newCurrentPage = (anomalies.length === 1 && (pagination.current || 1) > 1) 
                       ? (pagination.current || 1) - 1 
                       : pagination.current || 1;
      const newPaginationState = { ...pagination, current: newCurrentPage, total: newTotal };
      setPagination(newPaginationState); 
      loadAnomalies(newPaginationState, filters); 
    } catch (err) {
      notification.error({
        message: 'Ошибка удаления аномалии',
        description: (err as Error).message,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFetchLlmExplanation = async (anomalyId: number) => {
    setLlmLoading(true);
    setLlmExplanation(null);
    
    const anomalyToShow = anomalies.find(anom => anom.id === anomalyId) || selectedAnomaly;
    if (!anomalyToShow || anomalyToShow.id !== anomalyId) {
        const foundAnomaly = anomalies.find(anom => anom.id === anomalyId);
        if (foundAnomaly) {
            showViewModal(foundAnomaly);
        } else {
            notification.error({ message: 'Аномалия не найдена для отображения деталей перед запросом LLM.'});
        setLlmLoading(false);
        return;
        }
    } else if (!isModalVisible) {
        setIsModalVisible(true);
    }

    try {
      const response = await fetchLlmExplanation(anomalyId);
      setLlmExplanation(response.llm_explanation);
      notification.success({ message: 'Объяснение LLM получено' });
    } catch (err) {
      notification.error({
        message: 'Ошибка получения объяснения LLM',
        description: (err as Error).message,
      });
      setLlmExplanation("Не удалось загрузить объяснение.");
    } finally {
      setLlmLoading(false);
    }
  };

  const handleDeleteAllAnomalies = async () => {
    setDeleteAllLoading(true);
    setError(null);
    try {
      const response = await deleteAllAnomalies();
      notification.success({
        message: 'Успех',
        description: response.message || 'Все аномалии успешно удалены.',
      });
      setAnomalies([]);
      setTotalAnomalies(0);
      setPagination(prev => ({ ...prev, current: 1, total: 0 }));
    } catch (err) {
      setError((err as Error).message);
      notification.error({
        message: 'Ошибка при удалении всех аномалий',
        description: (err as Error).message,
      });
    } finally {
      setDeleteAllLoading(false);
    }
  };

  const columns: ColumnsType<RootAnomalySchema> = [
    { title: 'ID', dataIndex: 'id', key: 'id', sorter: (a, b) => a.id - b.id, defaultSortOrder: 'descend' },
    { title: 'Order ID', dataIndex: 'order_id', key: 'order_id' },
    { title: 'Order Item ID', dataIndex: 'order_item_id', key: 'order_item_id' },
    {
      title: 'Дата обнаружения',
      dataIndex: 'detection_date',
      key: 'detection_date',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a,b) => dayjs(a.detection_date).unix() - dayjs(b.detection_date).unix()
    },
    {
      title: 'Оценка',
      dataIndex: 'anomaly_score',
      key: 'anomaly_score',
      render: (score?: number | null) => score !== null && score !== undefined ? <Tag color={score > 0.7 ? 'volcano' : score > 0.5 ? 'orange' : 'green'}>{score.toFixed(2)}</Tag> : <Tag>N/A</Tag>,
      sorter: (a,b) => (a.anomaly_score ?? -1) - (b.anomaly_score ?? -1)
    },
    { title: 'Тип детектора', dataIndex: 'detector_type', key: 'detector_type' },
    {
      title: 'Действия',
      key: 'actions',
      fixed: 'right', 
      width: 150,    
      render: (_, record) => (
        <Space size="small"> 
          <Tooltip title="Просмотр деталей">
            <Button size="small" icon={<EyeOutlined />} onClick={() => showViewModal(record)} />
          </Tooltip>
          <Tooltip title="Получить объяснение LLM">
            <Button size="small" icon={<ExperimentOutlined />} onClick={() => handleFetchLlmExplanation(record.id)} loading={llmLoading && selectedAnomaly?.id === record.id} />
          </Tooltip>
          <Tooltip title="Удалить аномалию">
            <Popconfirm 
                title="Удалить аномалию?"
                description={`Вы уверены, что хотите удалить аномалию ID ${record.id}?`}
                onConfirm={() => handleDeleteAnomaly(record.id)}
                okText="Да"
                cancelText="Нет"
              >
              <Button size="small" icon={<DeleteOutlined />} danger />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ width: '100%' }}>
      <Card style={{ marginBottom: 24 }}>
        <Form
            form={formFilters}
            layout="vertical"
            onFinish={handleFilterSubmit}
        >
            <Row gutter={16}>
                <Col xs={24} sm={12} md={8} lg={6}>
                    <Form.Item name="dateRange" label="Диапазон дат обнаружения">
                        <RangePicker style={{ width: '100%' }} />
                    </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8} lg={4}>
                    <Form.Item name="min_score" label="Минимальная оценка">
                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} placeholder="напр. 0.5"/>
                    </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8} lg={4}>
                    <Form.Item name="max_score" label="Максимальная оценка">
                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} placeholder="напр. 0.9"/>
                    </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8} lg={6}>
                    <Form.Item name="detector_type" label="Тип детектора">
                        <Select allowClear placeholder="Выберите или введите">
                            {DETECTOR_TYPES.map(type => <Option key={type} value={type}>{type}</Option>)}
                        </Select>
                    </Form.Item>
                </Col>
                <Col xs={24} sm={24} md={24} lg={4} style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '24px' }}>
                     <Space>
                        <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>Фильтр</Button>
                        <Button onClick={resetFilters} icon={<ReloadOutlined />}>Сброс</Button>
                    </Space>
                </Col>
            </Row>
        </Form>
      </Card>

      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }} gutter={[16,16]}>
        <Col>
          <Paragraph style={{ margin: 0 }}>
              Управление и просмотр обнаруженных аномалий.
          </Paragraph>
        </Col>
        <Col>
          <Space wrap> 
            <Segmented
              options={[
                { value: 'table', icon: <TableOutlined /> },
                { value: 'card', icon: <AppstoreOutlined /> },
              ]}
              value={viewMode}
              onChange={(value) => setViewMode(value as 'table' | 'card')}
            />
            <Popconfirm
              title="Удалить ВСЕ аномалии?"
              description="Это действие необратимо и удалит все записи об аномалиях из системы. Вы уверены?"
              onConfirm={handleDeleteAllAnomalies}
              okText="Да, удалить все"
              cancelText="Отмена"
              okButtonProps={{ danger: true, loading: deleteAllLoading }}
              icon={<WarningOutlined style={{ color: 'red' }} />}
            >
              <Button danger icon={<DeleteOutlined />} loading={deleteAllLoading}>
                Удалить все (DEV)
              </Button>
            </Popconfirm>
          </Space>
        </Col>
      </Row>

      {error && <Alert message="Ошибка" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      {viewMode === 'table' ? (
        <Table
          columns={columns}
          dataSource={anomalies}
          rowKey="id"
          loading={loading}
          pagination={{...pagination, total: totalAnomalies}}
          onChange={handleTableChange}
          scroll={{ x: 'max-content' }}
        />
      ) : (
        <Spin spinning={loading}> 
            {anomalies.length > 0 ? (
                <Row gutter={[16, 16]}> 
                {anomalies.map(anomaly => (
                    <Col xs={24} sm={12} md={8} lg={6} key={anomaly.id}>
                    <AnomalyCard 
                        anomaly={anomaly} 
                        onViewDetails={showViewModal} 
                        onDelete={() => handleDeleteAnomaly(anomaly.id)} 
                        onGetLlmExplanation={() => handleFetchLlmExplanation(anomaly.id)} 
                        isLlmLoading={llmLoading && selectedAnomaly?.id === anomaly.id}
                    />
                    </Col>
                ))}
                </Row>
            ) : (
                !loading && totalAnomalies === 0 && <Paragraph>Нет данных для отображения.</Paragraph> 
            )}
            {totalAnomalies > (pagination.pageSize || 10) && (
                 <Row justify="center" style={{marginTop: 24}}>
                    <Pagination
                        current={pagination.current}
                        pageSize={pagination.pageSize}
                        total={totalAnomalies}
                        showSizeChanger={pagination.showSizeChanger}
                        pageSizeOptions={pagination.pageSizeOptions}
                        onChange={(page: number, pageSize?: number) => handleTableChange({current: page, pageSize: pageSize || pagination.pageSize})}                        
                    />
                 </Row>
            )}
        </Spin>
      )}

      <Modal
        title={`Детали аномалии (ID: ${selectedAnomaly?.id})`}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={[
            <Button key="back" onClick={handleModalCancel}>Закрыть</Button>,
        ]}
        width={800}
      >
        {selectedAnomaly && (
          <div>
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="ID Аномалии">{selectedAnomaly.id}</Descriptions.Item>
              <Descriptions.Item label="ID Заказа">{selectedAnomaly.order_id || 'N/A'}</Descriptions.Item>
              {selectedAnomaly.order_item_id && <Descriptions.Item label="ID Элемента Заказа">{selectedAnomaly.order_item_id}</Descriptions.Item>}
              <Descriptions.Item label="Дата Обнаружения">{dayjs(selectedAnomaly.detection_date).format('YYYY-MM-DD HH:mm:ss')}</Descriptions.Item>
              <Descriptions.Item label="Оценка Аномалии">
                {selectedAnomaly.anomaly_score !== null && selectedAnomaly.anomaly_score !== undefined 
                  ? <Tag color={selectedAnomaly.anomaly_score > 0.7 ? 'volcano' : selectedAnomaly.anomaly_score > 0.5 ? 'orange' : 'green'}>
                      {selectedAnomaly.anomaly_score.toFixed(2)}
                    </Tag> 
                  : <Tag>N/A</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="Тип Детектора">{selectedAnomaly.detector_type}</Descriptions.Item>
            </Descriptions>

            {selectedAnomaly.details && (
              <Card title="Технические детали аномалии" size="small" style={{ marginTop: 16, borderColor: '#e8e8e8' }} headStyle={{background: '#fafafa'}} >
                <Descriptions bordered column={1} size="small" layout="horizontal">
                  {selectedAnomaly.details.product_id && <Descriptions.Item label="Product ID (из деталей)">{selectedAnomaly.details.product_id}</Descriptions.Item>}
                  {selectedAnomaly.details.seller_id && <Descriptions.Item label="Seller ID (из деталей)">{selectedAnomaly.details.seller_id}</Descriptions.Item>}
                  {selectedAnomaly.details.final_threshold_used !== undefined && selectedAnomaly.details.final_threshold_used !== null &&
                    <Descriptions.Item label="Использованный порог">{selectedAnomaly.details.final_threshold_used.toFixed(2)}</Descriptions.Item>}
                </Descriptions>
                
                {selectedAnomaly.details.level_scores && (
                  <Card title="Оценки по уровням" type="inner" size="small" style={{ marginTop: 12, borderColor: '#f0f0f0' }}>
                    <Descriptions bordered column={1} size="small">
                      {Object.entries(selectedAnomaly.details.level_scores).map(([level, score]) => (
                        <Descriptions.Item label={level.split('_').map(s => s.charAt(0).toUpperCase() + s.substring(1)).join(' ')} key={level}>
                          {typeof score === 'number' ? score.toFixed(4) : String(score)}
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </Card>
                )}
                
                {(() => {
                  const { product_id, seller_id, final_threshold_used, level_scores, contributing_detectors_explanations, ...otherDetails } = selectedAnomaly.details as AnomalyDetailsType;
                  
                  const hasContributions = contributing_detectors_explanations && Object.values(contributing_detectors_explanations).some(level => level && level.length > 0);

                  if (!hasContributions) return null;

                  return (
                    <Collapse ghost size="small" style={{marginTop: 8}} defaultActiveKey={hasContributions ? ['contributingDetectors'] : []}>
                      {hasContributions && (
                        <Panel header={<Text type="secondary">Объяснения от Детекторов</Text>} key="contributingDetectors">
                          {Object.entries(contributing_detectors_explanations || {}).map(([levelName, explanations]) => {
                            if (!explanations || explanations.length === 0) return null;
                            const formattedLevelName = levelName.replace(/_level$/, '').split('_').map(s => s.charAt(0).toUpperCase() + s.substring(1)).join(' ');
                            return (
                              <Card key={levelName} title={`Уровень: ${formattedLevelName}`} type="inner" size="small" style={{marginBottom: 10, borderColor: '#f0f0f0'}} headStyle={{background: '#f9f9f9'}}>
                                {explanations.map((exp: DetectorExplanationEntry, index: number) => {
                                  let displayContent: string | object | null = null;
                                  let displayAsJson = false;

                                  const topLevelExpText = exp.explanation_text;
                                  const mainExplanation = exp.explanation;

                                  // 1. Top-level exp.explanation_text is a non-empty string
                                  if (typeof topLevelExpText === 'string' && topLevelExpText.trim() !== '') {
                                    displayContent = topLevelExpText;
                                  }
                                  // 2. mainExplanation is an object
                                  else if (mainExplanation && typeof mainExplanation === 'object') {
                                    const mainExplanationObj = mainExplanation as Record<string, unknown>;
                                    // 2a. Nested explanation_text directly under mainExplanation
                                    if (typeof mainExplanationObj.explanation_text === 'string' && (mainExplanationObj.explanation_text as string).trim() !== '') {
                                      displayContent = mainExplanationObj.explanation_text as string;
                                    }
                                    // 2b. Nested explanation_text under mainExplanation.detector_specific_info
                                    else if (mainExplanationObj.detector_specific_info && typeof mainExplanationObj.detector_specific_info === 'object') {
                                      const detectorInfo = mainExplanationObj.detector_specific_info as Record<string, unknown>;
                                      if (typeof detectorInfo.explanation_text === 'string' && (detectorInfo.explanation_text as string).trim() !== '') {
                                        displayContent = detectorInfo.explanation_text as string;
                                      } else {
                                        // Fallback: mainExplanation is an object, but no usable nested explanation_text found. Display mainExplanation as JSON.
                                        displayContent = mainExplanation;
                                        displayAsJson = true;
                                      }
                                    } else {
                                      // Fallback: mainExplanation is an object, but no detector_specific_info or no usable explanation_text there. Display mainExplanation as JSON.
                                      displayContent = mainExplanation;
                                      displayAsJson = true;
                                    }
                                  }
                                  // 3. mainExplanation itself is a non-empty string (and Check 1 & 2 failed)
                                  else if (typeof mainExplanation === 'string' && mainExplanation.trim() !== '') {
                                    displayContent = mainExplanation;
                                  }
                                  // 4. Top-level explanation_text is an object (and all above failed)
                                  else if (topLevelExpText && typeof topLevelExpText === 'object') {
                                    displayContent = topLevelExpText;
                                    displayAsJson = true;
                                  }
                                  
                                  return (
                                    <div key={index} style={{marginBottom: 8, paddingBottom: 8, borderBottom: index < explanations.length - 1 ? '1px dashed #e8e8e8' : 'none'}}>
                                      <Descriptions column={1} size="small">
                                        <Descriptions.Item label={<Text strong>{exp.detector_name}</Text>} labelStyle={{width: '150px'}}>
                                          <Tag>Оценка: {exp.score.toFixed(4)}</Tag>
                                        </Descriptions.Item>
                                        {displayContent !== null && (
                                          <Descriptions.Item label="Объяснение">
                                            {displayAsJson
                                              ? <SyntaxHighlighter language="json" style={atomOneDark} customStyle={{ maxHeight: '150px', overflowY: 'auto', fontSize: '12px', padding: '10px', borderRadius: '4px', background: '#282c34', margin: 0 }}>
                                                  {JSON.stringify(displayContent, null, 2)}
                                                </SyntaxHighlighter>
                                              : <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{displayContent as string}</Paragraph>
                                            }
                                          </Descriptions.Item>
                                        )}
                                      </Descriptions>
                                    </div>
                                  );
                                })}
                              </Card>
                            );
                          })}
                        </Panel>
                      )}
                    </Collapse>
                  );
                })()}
              </Card>
            )}

            {/* Новый блок для LLM объяснения */}
            {selectedAnomaly && (
              <div style={{ marginTop: 16 }}>
                {!llmExplanation && !llmLoading && (
                  <Button 
                    type="dashed" 
                    icon={<ExperimentOutlined />} 
                    onClick={() => handleFetchLlmExplanation(selectedAnomaly.id)}
                    block
                  >
                    Получить объяснение от LLM
                  </Button>
                )}
                {llmLoading && (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Spin tip="Загрузка объяснения LLM..." />
                  </div>
                )}
                {llmExplanation && !llmLoading && (
                  <Card title="Объяснение LLM" headStyle={{ background: '#e6f7ff' }} style={{borderColor: '#91d5ff'}}>
                    <Paragraph style={{ whiteSpace: 'pre-wrap' }}>{llmExplanation}</Paragraph>
                    <Button 
                        type="link" 
                        onClick={() => { setLlmExplanation(null); setLlmLoading(false); /* Опционально: сбросить ошибку LLM если есть */ }} 
                        style={{float: 'right', padding: 0}}
                        danger
                    >
                        Скрыть объяснение
                    </Button>
                  </Card>
                )}
                {/* Можно добавить обработку ошибок LLM здесь, если необходимо */}
              </div>
            )}

            {/* Карточка для Графа Связей */}
            {selectedAnomaly && (
              <Card title={<Space><DeploymentUnitOutlined />Связи Сущностей</Space>} style={{marginTop: 16, borderColor: '#e8e8e8'}} headStyle={{background: '#fafafa'}} size="small">
                <AnomalyRelationshipGraph 
                  anomaly={selectedAnomaly}
                  order={selectedAnomalyOrderDetails}
                  detailedProduct={selectedAnomalyProductDetails}
                  detailedSeller={selectedAnomalySellerDetails}
                />
              </Card>
            )}

            {selectedAnomaly.order_id && (
              <Card title={<Space><ShoppingCartOutlined />Информация о Заказе ({selectedAnomaly.order_id})</Space>} style={{marginTop: 16, borderColor: '#e8e8e8'}} headStyle={{background: '#fafafa'}}>
                {orderLoading && <div style={{textAlign: 'center', padding: '20px'}}><Spin tip="Загрузка деталей заказа..." /></div>}
                {orderError && <Alert message="Ошибка загрузки заказа" description={orderError} type="error" showIcon />}
                {selectedAnomalyOrderDetails && !orderLoading && !orderError && (
                  <div>
                    <Timeline mode="left" style={{marginTop: 8, marginLeft: -10}}>
                        {selectedAnomalyOrderDetails.order_status && (
                             <Timeline.Item 
                                dot={selectedAnomalyOrderDetails.order_status === 'delivered' ? <CheckCircleOutlined /> : <SyncOutlined />}
                                color={orderStatusColors[selectedAnomalyOrderDetails.order_status] || 'default'}
                              >
                                <Text strong>Статус заказа: </Text>
                                <Tag color={orderStatusColors[selectedAnomalyOrderDetails.order_status] || 'default'} style={{marginLeft: 8}}>
                                    {selectedAnomalyOrderDetails.order_status}
                                </Tag>
                            </Timeline.Item>
                        )}
                        {selectedAnomalyOrderDetails.order_purchase_timestamp && (
                          <Timeline.Item 
                            label={formatDate(selectedAnomalyOrderDetails.order_purchase_timestamp)} 
                            dot={<ClockCircleOutlined />}
                            color="blue"
                          >
                            <Text strong>Заказ создан / Дата покупки</Text>
                          </Timeline.Item>
                        )}
                        {selectedAnomalyOrderDetails.order_approved_at && (
                          <Timeline.Item 
                            label={formatDate(selectedAnomalyOrderDetails.order_approved_at)}
                            dot={<CheckCircleOutlined />}
                            color="green"
                          >
                            Заказ оплачен/подтвержден
                          </Timeline.Item>
                        )}
                        {selectedAnomalyOrderDetails.order_delivered_carrier_date && (
                          <Timeline.Item 
                            label={formatDate(selectedAnomalyOrderDetails.order_delivered_carrier_date)} 
                            dot={<SyncOutlined spin/>}
                            color="processing"
                          >
                            Передан в доставку
                          </Timeline.Item>
                        )}
                        {selectedAnomalyOrderDetails.order_delivered_customer_date && (
                          <Timeline.Item 
                            label={formatDate(selectedAnomalyOrderDetails.order_delivered_customer_date)} 
                            dot={<CheckCircleOutlined />}
                            color="success"
                           >
                            Доставлен клиенту
                          </Timeline.Item>
                        )}
                        {selectedAnomalyOrderDetails.order_status !== 'delivered' && 
                         !selectedAnomalyOrderDetails.order_delivered_customer_date && 
                         selectedAnomalyOrderDetails.order_estimated_delivery_date && (
                          <Timeline.Item 
                            label={formatDate(selectedAnomalyOrderDetails.order_estimated_delivery_date)}
                            dot={<ClockCircleOutlined style={{color: 'gray'}}/>}
                            color="gray"
                          >
                            Ожидаемая дата доставки (текущий статус: {selectedAnomalyOrderDetails.order_status})
                          </Timeline.Item>
                        )}
                      </Timeline>

                    {selectedAnomalyOrderDetails.customer && (
                        <Card title={<Space><UserOutlined />Клиент</Space>} type="inner" size="small" style={{ marginTop: 12, borderColor: '#f0f0f0' }}>
                           <Descriptions bordered column={1} size="small">
                                <Descriptions.Item label="Уникальный ID Клиента">{selectedAnomalyOrderDetails.customer.customer_unique_id}</Descriptions.Item>
                                <Descriptions.Item label="Город">{selectedAnomalyOrderDetails.customer.customer_city}</Descriptions.Item>
                                <Descriptions.Item label="Штат">{selectedAnomalyOrderDetails.customer.customer_state}</Descriptions.Item>
                           </Descriptions>
                        </Card>
                    )}
                    {selectedAnomalyOrderDetails.items && selectedAnomalyOrderDetails.items.length > 0 && (
                        <Card title={<Space><ShoppingCartOutlined />Товары в Заказе ({selectedAnomalyOrderDetails.items.length})</Space>} type="inner" size="small" style={{ marginTop: 12, borderColor: '#f0f0f0' }}>
                            <List
                                dataSource={selectedAnomalyOrderDetails.items}
                                size="small"
                                renderItem={(item: OrderItemSchema) => {
                                    const isAnomalyItem = !!(selectedAnomaly.order_item_id && item.order_item_id === selectedAnomaly.order_item_id);
                                    return (
                                        <List.Item 
                                            style={isAnomalyItem ? 
                                                {backgroundColor: '#fffbe6', border: '1px solid #ffe58f', borderRadius: '4px', margin: '4px 0', padding: '8px'} : 
                                                {padding: '8px', borderBottom: '1px solid #f0f0f0'}
                                            }
                                        >
                                            <Descriptions bordered={false} column={1} size="small" style={{width: '100%'}}>
                                                <Descriptions.Item label={<Text strong>Позиция ID</Text>} span={1}>
                                                    <Text strong={isAnomalyItem}>{item.order_item_id} (Продукт: {item.product_id})</Text> 
                                                    {isAnomalyItem && <Tag color="warning" style={{marginLeft: 8}}>Связано с аномалией</Tag>}
                                                </Descriptions.Item>
                                                <Descriptions.Item label="Цена" span={1}>{item.price.toFixed(2)}</Descriptions.Item>
                                                <Descriptions.Item label="Доставка" span={1}>{item.freight_value.toFixed(2)}</Descriptions.Item>
                                                {item.product?.product_category_name && 
                                                    <Descriptions.Item label="Категория" span={1}>{item.product.product_category_name}</Descriptions.Item>}
                                                <Descriptions.Item label="Продавец ID" span={1}>{item.seller_id}</Descriptions.Item>
                                            </Descriptions>
                                        </List.Item>
                                    );
                                }}
                            />
                        </Card>
                    )}
                  </div>
                )}
              </Card>
            )}

            {/* Карточка для деталей Продукта */}
            {selectedAnomalyProductDetails && !productLoading && (
              <Card title={<Space><AppstoreOutlined />Информация о Продукте ({selectedAnomalyProductDetails.product_id})</Space>} style={{marginTop: 16, borderColor: '#e8e8e8'}} headStyle={{background: '#fafafa'}} size="small">
                <Descriptions bordered column={1} size="small">
                  <Descriptions.Item label="ID Продукта">{selectedAnomalyProductDetails.product_id}</Descriptions.Item>
                  <Descriptions.Item label="Категория">
                    {selectedAnomalyProductDetails.product_category_name || 'N/A'}
                    {selectedAnomalyProductDetails.category_translation && selectedAnomalyProductDetails.product_category_name !== selectedAnomalyProductDetails.category_translation.product_category_name_english && 
                      <Text type="secondary"> ({selectedAnomalyProductDetails.category_translation.product_category_name_english})</Text>}
                  </Descriptions.Item>
                  <Descriptions.Item label="Длина имени">{selectedAnomalyProductDetails.product_name_lenght ?? 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Длина описания">{selectedAnomalyProductDetails.product_description_lenght ?? 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Кол-во фото">{selectedAnomalyProductDetails.product_photos_qty ?? 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Вес (г)">{selectedAnomalyProductDetails.product_weight_g ?? 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Размеры (ДxШxВ см)">
                    {[selectedAnomalyProductDetails.product_length_cm, selectedAnomalyProductDetails.product_width_cm, selectedAnomalyProductDetails.product_height_cm].filter(d => d !== null && d !== undefined).join(' x ') || 'N/A'}
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            )}
            {productLoading && <div style={{textAlign: 'center', padding: '20px'}}><Spin tip="Загрузка деталей продукта..." /></div>}
            {productError && <Alert message="Ошибка загрузки продукта" description={productError} type="error" showIcon style={{marginTop:16}}/>}

            {/* Карточка для деталей Продавца */}
            {selectedAnomalySellerDetails && !sellerLoading && (
              <Card title={<Space><UserOutlined />Информация о Продавце ({selectedAnomalySellerDetails.seller_id})</Space>} style={{marginTop: 16, borderColor: '#e8e8e8'}} headStyle={{background: '#fafafa'}} size="small">
                <Descriptions bordered column={1} size="small">
                  <Descriptions.Item label="ID Продавца">{selectedAnomalySellerDetails.seller_id}</Descriptions.Item>
                  <Descriptions.Item label="Город">{selectedAnomalySellerDetails.seller_city || 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Штат">{selectedAnomalySellerDetails.seller_state || 'N/A'}</Descriptions.Item>
                  <Descriptions.Item label="Индекс (префикс)">{selectedAnomalySellerDetails.seller_zip_code_prefix || 'N/A'}</Descriptions.Item>
                </Descriptions>
              </Card>
            )}
            {sellerLoading && <div style={{textAlign: 'center', padding: '20px'}}><Spin tip="Загрузка деталей продавца..." /></div>}
            {sellerError && <Alert message="Ошибка загрузки продавца" description={sellerError} type="error" showIcon style={{marginTop:16}}/>}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AnomaliesPage;