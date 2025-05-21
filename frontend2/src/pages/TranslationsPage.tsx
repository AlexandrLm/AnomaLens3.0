import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Button, Row, Col, Tooltip, Input, Space } from 'antd';
import { TranslationOutlined, ReloadOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchTranslations } from '../services/translationService';
import type { ProductCategoryNameTranslationSchema, FetchTranslationsParams } from '../types/api';
import { notification } from 'antd';

const { Title, Paragraph, Text } = Typography;

const TranslationsPage: React.FC = () => {
  const [translations, setTranslations] = useState<ProductCategoryNameTranslationSchema[]>([]);
  const [totalTranslations, setTotalTranslations] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 15,
    showSizeChanger: true,
    pageSizeOptions: ['15', '30', '50', '100'],
  });

  // const [searchText, setSearchText] = useState('');

  const loadTranslations = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchTranslationsParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 15),
        limit: currentPagination.pageSize || 15,
      };
      const data = await fetchTranslations(params);
      setTranslations(data.items);
      setTotalTranslations(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize}));
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки переводов', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTranslations(pagination);
  }, [loadTranslations, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  // const filteredTranslations = translations.filter(t => 
  //   t.product_category_name.toLowerCase().includes(searchText.toLowerCase()) ||
  //   t.product_category_name_english.toLowerCase().includes(searchText.toLowerCase())
  // );

  const columns: ColumnsType<ProductCategoryNameTranslationSchema> = [
    {
      title: 'Оригинальное название категории',
      dataIndex: 'product_category_name',
      key: 'product_category_name',
      ellipsis: true,
      sorter: (a, b) => a.product_category_name.localeCompare(b.product_category_name),
    },
    {
      title: 'Перевод (English)',
      dataIndex: 'product_category_name_english',
      key: 'product_category_name_english',
      ellipsis: true,
      sorter: (a, b) => a.product_category_name_english.localeCompare(b.product_category_name_english),
    },
  ];

  return (
    <Card>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}><TranslationOutlined style={{marginRight: 8}}/> Переводы Названий Категорий</Title>
        </Col>
        <Col>
          {/* <Input.Search 
            placeholder="Поиск по названию..." 
            onChange={e => setSearchText(e.target.value)} 
            style={{ width: 250, marginRight: 16 }}
            allowClear
          /> 
          */}
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadTranslations({ ...pagination, current: 1 })} 
            loading={loading && !error}
            disabled={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Список переводов названий категорий продуктов с португальского на английский.
        Ключом для каждой записи является `product_category_name`.
      </Paragraph>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={translations}
        rowKey={(record) => record.product_category_name}
        loading={loading}
        pagination={{...pagination, total: totalTranslations}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />
    </Card>
  );
};

export default TranslationsPage; 