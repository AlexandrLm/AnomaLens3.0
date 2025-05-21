import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Button, Row, Col, Tooltip, Modal, Descriptions, Space, notification } from 'antd';
import { UserOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchCustomers, fetchCustomerById } from '../services/customerService';
import type { CustomerSchema, FetchCustomersParams } from '../types/api';

const { Title, Paragraph, Text } = Typography;

const CustomersPage: React.FC = () => {
  const [customers, setCustomers] = useState<CustomerSchema[]>([]);
  const [totalCustomers, setTotalCustomers] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState<CustomerSchema | null>(null);
  const [modalLoading, setModalLoading] = useState(false);


  const loadCustomers = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchCustomersParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
      };
      const data = await fetchCustomers(params);
      setCustomers(data.items);
      setTotalCustomers(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize})); // Добавил current и pageSize в setPagination
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки данных клиентов', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCustomers(pagination);
  }, [loadCustomers, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const showCustomerDetails = async (customerId: string) => {
    setIsModalVisible(true);
    setModalLoading(true);
    setSelectedCustomer(null);
    try {
        const customerData = await fetchCustomerById(customerId);
        setSelectedCustomer(customerData);
    } catch (err) {
        notification.error({
            message: `Ошибка загрузки клиента ${customerId}`,
            description: (err as Error).message
        });
        setIsModalVisible(false);
    } finally {
        setModalLoading(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedCustomer(null);
  };


  const columns: ColumnsType<CustomerSchema> = [
    {
      title: 'Customer ID',
      dataIndex: 'customer_id',
      key: 'customer_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'Unique ID',
      dataIndex: 'customer_unique_id',
      key: 'customer_unique_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'ZIP Prefix',
      dataIndex: 'customer_zip_code_prefix',
      key: 'customer_zip_code_prefix',
      align: 'center',
    },
    {
      title: 'Город',
      dataIndex: 'customer_city',
      key: 'customer_city',
      ellipsis: true,
    },
    {
      title: 'Штат',
      dataIndex: 'customer_state',
      key: 'customer_state',
      align: 'center',
      width: 80,
    },
    {
      title: 'Действия',
      key: 'actions',
      align: 'center',
      width: 100,
      render: (_, record: CustomerSchema) => (
        <Tooltip title="Просмотреть детали клиента">
          <Button icon={<EyeOutlined />} onClick={() => showCustomerDetails(record.customer_id)} />
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}><UserOutlined style={{marginRight: 8}}/> Данные Покупателей</Title>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadCustomers({ ...pagination, current: 1 })} 
            loading={loading && !error}
            disabled={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Список покупателей с основной информацией и возможностью пагинации.
      </Paragraph>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={customers}
        rowKey="customer_id"
        loading={loading}
        pagination={{...pagination, total: totalCustomers}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />

      <Modal
        title={selectedCustomer ? `Детали клиента: ${selectedCustomer.customer_id}` : "Загрузка деталей..."}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={<Button key="close" onClick={handleModalCancel}>Закрыть</Button>}
        width={600}
      >
        {modalLoading && <div style={{textAlign: 'center', padding: '30px'}}><Spin size="large" /></div>}
        {selectedCustomer && !modalLoading && (
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="Customer ID">{selectedCustomer.customer_id}</Descriptions.Item>
            <Descriptions.Item label="Customer Unique ID">{selectedCustomer.customer_unique_id}</Descriptions.Item>
            <Descriptions.Item label="ZIP Code Prefix">{selectedCustomer.customer_zip_code_prefix}</Descriptions.Item>
            <Descriptions.Item label="Город">{selectedCustomer.customer_city}</Descriptions.Item>
            <Descriptions.Item label="Штат">{selectedCustomer.customer_state}</Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </Card>
  );
};

export default CustomersPage; 