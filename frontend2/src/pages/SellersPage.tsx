import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Button, Row, Col, Tooltip, Modal, Descriptions, Space, notification } from 'antd';
import { TeamOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons'; // TeamOutlined для продавцов
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchSellers, fetchSellerById } from '../services/sellerService';
import type { SellerSchema, FetchSellersParams } from '../types/api';

const { Title, Paragraph, Text } = Typography;

const SellersPage: React.FC = () => {
  const [sellers, setSellers] = useState<SellerSchema[]>([]);
  const [totalSellers, setTotalSellers] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedSeller, setSelectedSeller] = useState<SellerSchema | null>(null);
  const [modalLoading, setModalLoading] = useState(false);

  const loadSellers = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchSellersParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
      };
      const data = await fetchSellers(params);
      setSellers(data.items);
      setTotalSellers(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize})); // Добавил current и pageSize
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки данных продавцов', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSellers(pagination);
  }, [loadSellers, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const showSellerDetails = async (sellerId: string) => {
    setIsModalVisible(true);
    setModalLoading(true);
    setSelectedSeller(null);
    try {
        const sellerData = await fetchSellerById(sellerId);
        setSelectedSeller(sellerData);
    } catch (err) {
        notification.error({
            message: `Ошибка загрузки продавца ${sellerId}`,
            description: (err as Error).message
        });
        setIsModalVisible(false);
    } finally {
        setModalLoading(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedSeller(null);
  };

  const columns: ColumnsType<SellerSchema> = [
    {
      title: 'Seller ID',
      dataIndex: 'seller_id',
      key: 'seller_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'ZIP Prefix',
      dataIndex: 'seller_zip_code_prefix',
      key: 'seller_zip_code_prefix',
      align: 'center',
    },
    {
      title: 'Город',
      dataIndex: 'seller_city',
      key: 'seller_city',
      ellipsis: true,
    },
    {
      title: 'Штат',
      dataIndex: 'seller_state',
      key: 'seller_state',
      align: 'center',
      width: 80,
    },
    {
      title: 'Действия',
      key: 'actions',
      align: 'center',
      width: 100,
      render: (_, record: SellerSchema) => (
        <Tooltip title="Просмотреть детали продавца">
          <Button icon={<EyeOutlined />} onClick={() => showSellerDetails(record.seller_id)} />
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}><TeamOutlined style={{marginRight: 8}}/> Данные Продавцов</Title>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadSellers({ ...pagination, current: 1 })} 
            loading={loading && !error}
            disabled={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Список продавцов с основной информацией и возможностью пагинации.
      </Paragraph>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={sellers}
        rowKey="seller_id"
        loading={loading}
        pagination={{...pagination, total: totalSellers}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />

      <Modal
        title={selectedSeller ? `Детали продавца: ${selectedSeller.seller_id}` : "Загрузка деталей..."}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={<Button key="close" onClick={handleModalCancel}>Закрыть</Button>}
        width={600}
      >
        {modalLoading && <div style={{textAlign: 'center', padding: '30px'}}><Spin size="large" /></div>}
        {selectedSeller && !modalLoading && (
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="Seller ID">{selectedSeller.seller_id}</Descriptions.Item>
            <Descriptions.Item label="ZIP Code Prefix">{selectedSeller.seller_zip_code_prefix}</Descriptions.Item>
            <Descriptions.Item label="Город">{selectedSeller.seller_city}</Descriptions.Item>
            <Descriptions.Item label="Штат">{selectedSeller.seller_state}</Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </Card>
  );
};

export default SellersPage; 