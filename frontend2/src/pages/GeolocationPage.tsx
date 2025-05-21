import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Button, Row, Col, Tooltip, notification } from 'antd';
import { GlobalOutlined, ReloadOutlined, EnvironmentOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchGeolocations } from '../services/geolocationService';
import type { GeolocationSchema, FetchGeolocationParams } from '../types/api';

const { Title, Paragraph } = Typography;

const GeolocationPage: React.FC = () => {
  const [geolocations, setGeolocations] = useState<GeolocationSchema[]>([]);
  const [totalGeolocations, setTotalGeolocations] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100', '200'],
  });

  const loadGeolocations = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchGeolocationParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
      };
      const data = await fetchGeolocations(params);
      setGeolocations(data.items);
      setTotalGeolocations(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize }));
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки данных геолокации', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGeolocations(pagination);
  }, [loadGeolocations, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const columns: ColumnsType<GeolocationSchema> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 100,
      sorter: (a, b) => a.id - b.id,
    },
    {
      title: 'Префикс ZIP-кода',
      dataIndex: 'geolocation_zip_code_prefix',
      key: 'geolocation_zip_code_prefix',
      align: 'center',
      sorter: (a, b) => a.geolocation_zip_code_prefix - b.geolocation_zip_code_prefix,
    },
    {
      title: 'Город',
      dataIndex: 'geolocation_city',
      key: 'geolocation_city',
      ellipsis: true,
      sorter: (a, b) => a.geolocation_city.localeCompare(b.geolocation_city),
    },
    {
      title: 'Штат',
      dataIndex: 'geolocation_state',
      key: 'geolocation_state',
      align: 'center',
      width: 80,
      sorter: (a, b) => a.geolocation_state.localeCompare(b.geolocation_state),
    },
    {
      title: 'Широта',
      dataIndex: 'geolocation_lat',
      key: 'geolocation_lat',
      align: 'right',
      render: (lat: number) => lat.toFixed(4),
    },
    {
      title: 'Долгота',
      dataIndex: 'geolocation_lng',
      key: 'geolocation_lng',
      align: 'right',
      render: (lng: number) => lng.toFixed(4),
    },
    {
      title: 'Карта',
      key: 'map_link',
      align: 'center',
      width: 80,
      render: (_, record: GeolocationSchema) => (
        <Tooltip title="Посмотреть на карте">
          <a 
            href={`https://www.google.com/maps?q=${record.geolocation_lat},${record.geolocation_lng}`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            <Button icon={<EnvironmentOutlined />} type="link" />
          </a>
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
       <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}><GlobalOutlined style={{marginRight: 8}}/> Данные Геолокации</Title>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadGeolocations({ ...pagination, current: 1 })} 
            loading={loading && !error}
            disabled={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Список записей геолокации, сопоставляющих префиксы почтовых индексов с координатами, городами и штатами.
        Каждая запись имеет уникальный `id`, возвращаемый API.
      </Paragraph>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={geolocations}
        rowKey="id"
        loading={loading}
        pagination={{...pagination, total: totalGeolocations}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />
    </Card>
  );
};

export default GeolocationPage; 