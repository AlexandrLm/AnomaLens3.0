import React, { useState, useEffect, useCallback } from 'react';
import {
  Typography, Table, Spin, Alert, Card, Tag, Tooltip, Button, Modal, Descriptions, DatePicker, Form, Row, Col, Space, List, Divider, Empty, notification, Tabs
} from 'antd';
import { EyeOutlined, ReloadOutlined, SearchOutlined, ShoppingCartOutlined, UserOutlined, CreditCardOutlined, MessageOutlined, ProfileOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchOrders, fetchOrderById } from '../services/orderService';
import type { OrderSchema, FetchOrdersParams, OrderItemSchema, OrderPaymentSchema, OrderReviewSchema } from '../types/api';
import dayjs from 'dayjs';

const { Title, Paragraph, Text } = Typography;
const { RangePicker } = DatePicker;

// Хелпер для форматирования дат
const formatDate = (dateString?: string | null) => {
  return dateString ? dayjs(dateString).format('YYYY-MM-DD HH:mm:ss') : 'N/A';
};

// Статусы заказов и их цвета (пример)
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


const OrdersPage: React.FC = () => {
  const [orders, setOrders] = useState<OrderSchema[]>([]);
  const [totalOrders, setTotalOrders] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [filters, setFilters] = useState<Omit<FetchOrdersParams, 'skip' | 'limit'>>({});
  const [formFilters] = Form.useForm();

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedOrder, setSelectedOrder] = useState<OrderSchema | null>(null);
  const [modalLoading, setModalLoading] = useState(false);

  const loadOrders = useCallback(async (
    currentPagination: TablePaginationConfig,
    currentFilters: Omit<FetchOrdersParams, 'skip' | 'limit'>
  ) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchOrdersParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
        ...currentFilters,
      };
      const data = await fetchOrders(params);
      setOrders(data.items);
      setTotalOrders(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize }));
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки заказов', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOrders(pagination, filters);
  }, [loadOrders, pagination.current, pagination.pageSize, filters]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const handleFilterSubmit = (values: any) => {
    const newFilters: Omit<FetchOrdersParams, 'skip' | 'limit'> = {
      start_date: values.dateRange?.[0]?.toISOString(),
      end_date: values.dateRange?.[1]?.toISOString(),
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

  const showOrderDetails = async (orderId: string) => {
    setIsModalVisible(true);
    setModalLoading(true);
    setSelectedOrder(null);
    try {
        const orderData = await fetchOrderById(orderId);
        setSelectedOrder(orderData);
    } catch (err) {
        notification.error({
            message: `Ошибка загрузки заказа ${orderId}`,
            description: (err as Error).message
        });
        setIsModalVisible(false);
    } finally {
        setModalLoading(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedOrder(null);
  };

  const columns: ColumnsType<OrderSchema> = [
    {
      title: 'ID Заказа',
      dataIndex: 'order_id',
      key: 'order_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'Статус',
      dataIndex: 'order_status',
      key: 'order_status',
      align: 'center',
      render: (status: string) => <Tag color={orderStatusColors[status] || 'default'}>{status}</Tag>,
    },
    {
      title: 'Дата покупки',
      dataIndex: 'order_purchase_timestamp',
      key: 'order_purchase_timestamp',
      render: (date: string) => formatDate(date),
      sorter: (a,b) => dayjs(a.order_purchase_timestamp).unix() - dayjs(b.order_purchase_timestamp).unix()
    },
    {
      title: 'Клиент ID',
      dataIndex: 'customer_id',
      key: 'customer_id',
      ellipsis: true,
       render: (id: string, record: OrderSchema) => (
         <Tooltip title={record.customer ? `Город: ${record.customer.customer_city}, Штат: ${record.customer.customer_state}` : 'Детали клиента не загружены'}>
           <Text copyable={{text: id}}>{id}</Text>
         </Tooltip>
       ),
    },
    {
      title: 'Кол-во позиций',
      dataIndex: 'items',
      key: 'items_count',
      align: 'center',
      render: (items: OrderItemSchema[]) => items?.length || 0,
    },
    {
      title: 'Сумма платежей',
      key: 'total_payment',
      align: 'right',
      render: (_, record: OrderSchema) => {
        const total = record.payments?.reduce((sum, p) => sum + p.payment_value, 0);
        return total ? total.toFixed(2) : 'N/A';
      },
    },
    {
      title: 'Действия',
      key: 'actions',
      align: 'center',
      width: 100,
      render: (_, record: OrderSchema) => (
        <Tooltip title="Просмотреть детали заказа">
          <Button icon={<EyeOutlined />} onClick={() => showOrderDetails(record.order_id)} />
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
      <Title level={3} style={{ margin: 0, marginBottom: 24 }}>Список Заказов</Title>
      
      <Form form={formFilters} layout="vertical" onFinish={handleFilterSubmit} style={{marginBottom: 24}}>
        <Row gutter={16} align="bottom">
            <Col xs={24} sm={12} md={8}>
                <Form.Item name="dateRange" label="Диапазон дат покупки">
                    <RangePicker style={{ width: '100%' }} />
                </Form.Item>
            </Col>
            <Col xs={24} sm={12} md={8}>
                 <Form.Item>
                    <Space>
                        <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>Фильтр</Button>
                        <Button onClick={resetFilters} icon={<ReloadOutlined />}>Сброс</Button>
                         <Button 
                            icon={<ReloadOutlined />} 
                            onClick={() => loadOrders({ ...pagination, current: 1 }, filters)} 
                            loading={loading && !error} // Показываем лоадер на кнопке только если нет глобальной ошибки
                            disabled={loading}
                        >
                            Обновить список
                        </Button>
                    </Space>
                </Form.Item>
            </Col>
        </Row>
      </Form>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={orders}
        rowKey="order_id"
        loading={loading}
        pagination={{...pagination, total: totalOrders}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />

      <Modal
        title={selectedOrder ? `Детали заказа: ${selectedOrder.order_id}` : "Загрузка деталей..."}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={<Button key="close" onClick={handleModalCancel}>Закрыть</Button>}
        width={900} // Шире для деталей заказа
      >
        {modalLoading && <div style={{textAlign: 'center', padding: '30px'}}><Spin size="large" /></div>}
        {selectedOrder && !modalLoading && (
          <Tabs defaultActiveKey="general"
            items={[
                { 
                    key: 'general', 
                    label: <><ProfileOutlined /> Общее</>, 
                    children: (
                        <Descriptions bordered column={1} size="small" style={{marginTop: 16}}>
                            <Descriptions.Item label="ID Заказа">{selectedOrder.order_id}</Descriptions.Item>
                            <Descriptions.Item label="Статус"><Tag color={orderStatusColors[selectedOrder.order_status] || 'default'}>{selectedOrder.order_status}</Tag></Descriptions.Item>
                            <Descriptions.Item label="Дата покупки">{formatDate(selectedOrder.order_purchase_timestamp)}</Descriptions.Item>
                            <Descriptions.Item label="Дата подтверждения">{formatDate(selectedOrder.order_approved_at)}</Descriptions.Item>
                            <Descriptions.Item label="Дата передачи перевозчику">{formatDate(selectedOrder.order_delivered_carrier_date)}</Descriptions.Item>
                            <Descriptions.Item label="Дата доставки клиенту">{formatDate(selectedOrder.order_delivered_customer_date)}</Descriptions.Item>
                            <Descriptions.Item label="Ожидаемая дата доставки">{formatDate(selectedOrder.order_estimated_delivery_date)}</Descriptions.Item>
                        </Descriptions>
                    )
                },
                {
                    key: 'customer',
                    label: <><UserOutlined /> Клиент</>,
                    children: selectedOrder.customer ? (
                         <Descriptions bordered column={1} size="small" style={{marginTop: 16}}>
                            <Descriptions.Item label="ID Клиента">{selectedOrder.customer.customer_id}</Descriptions.Item>
                            <Descriptions.Item label="Уникальный ID Клиента">{selectedOrder.customer.customer_unique_id}</Descriptions.Item>
                            <Descriptions.Item label="Индекс (префикс)">{selectedOrder.customer.customer_zip_code_prefix}</Descriptions.Item>
                            <Descriptions.Item label="Город">{selectedOrder.customer.customer_city}</Descriptions.Item>
                            <Descriptions.Item label="Штат">{selectedOrder.customer.customer_state}</Descriptions.Item>
                        </Descriptions>
                    ) : <Empty description="Информация о клиенте отсутствует" />
                },
                {
                    key: 'items',
                    label: <><ShoppingCartOutlined /> Позиции ({selectedOrder.items?.length || 0})</>,
                    children: (
                        <List
                            itemLayout="horizontal"
                            dataSource={selectedOrder.items}
                            renderItem={(item: OrderItemSchema) => (
                            <List.Item>
                                <List.Item.Meta
                                title={<Text>Продукт ID: <Text copyable>{item.product_id}</Text> (Поз. ID: {item.order_item_id})</Text>}
                                description={(
                                    <Space direction="vertical" size="small">
                                        <Text>Продавец ID: <Text copyable>{item.seller_id}</Text></Text>
                                        <Text>Цена: {item.price.toFixed(2)}, Доставка: {item.freight_value.toFixed(2)}</Text>
                                        <Text type="secondary">Лимит отправки: {formatDate(item.shipping_limit_date)}</Text>
                                         {item.product && <Text strong>Продукт: {item.product.product_category_name || 'Без категории'}</Text>}
                                         {item.seller && <Text strong>Продавец: {item.seller.seller_city}, {item.seller.seller_state}</Text>}
                                    </Space>
                                )}
                                />
                            </List.Item>
                            )}
                            locale={{ emptyText: <Empty description="Нет позиций в заказе" />}}
                            style={{marginTop: 16}}
                        />
                    )
                },
                {
                    key: 'payments',
                    label: <><CreditCardOutlined /> Платежи ({selectedOrder.payments?.length || 0})</>,
                    children: (
                        <List
                            dataSource={selectedOrder.payments}
                            renderItem={(payment: OrderPaymentSchema) => (
                            <List.Item>
                                <Descriptions column={2} size="small" style={{width: '100%'}}>
                                    <Descriptions.Item label="Тип">{payment.payment_type}</Descriptions.Item>
                                    <Descriptions.Item label="Сумма">{payment.payment_value.toFixed(2)}</Descriptions.Item>
                                    <Descriptions.Item label="№ Попытки">{payment.payment_sequential}</Descriptions.Item>
                                    <Descriptions.Item label="Рассрочка">{payment.payment_installments} мес.</Descriptions.Item>
                                </Descriptions>
                            </List.Item>
                            )}
                             locale={{ emptyText: <Empty description="Нет информации о платежах" />}}
                             style={{marginTop: 16}}
                        />
                    )
                },
                 {
                    key: 'reviews',
                    label: <><MessageOutlined /> Отзывы ({selectedOrder.reviews?.length || 0})</>,
                    children: (
                        <List
                            dataSource={selectedOrder.reviews}
                            renderItem={(review: OrderReviewSchema) => (
                            <List.Item>
                                <Card size="small" title={`Отзыв ID: ${review.review_id} (Оценка: ${review.review_score}/5)`} style={{width: '100%'}}>
                                    {review.review_comment_title && <Paragraph strong>{review.review_comment_title}</Paragraph>}
                                    {review.review_comment_message && <Paragraph>{review.review_comment_message}</Paragraph>}
                                    <Paragraph type="secondary">Создан: {formatDate(review.review_creation_date)}</Paragraph>
                                    <Paragraph type="secondary">Ответ: {formatDate(review.review_answer_timestamp)}</Paragraph>
                                </Card>
                            </List.Item>
                            )}
                            locale={{ emptyText: <Empty description="Нет отзывов для этого заказа" />}}
                            style={{marginTop: 16}}
                        />
                    )
                }
            ]}
          />
        )}
      </Modal>
    </Card>
  );
};

export default OrdersPage; 