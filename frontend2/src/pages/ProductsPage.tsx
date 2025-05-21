import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Tag, Tooltip, Button, Modal, Descriptions, Space, Row, Col, notification } from 'antd';
import { EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchProducts, fetchProductById, type FetchProductsParams } from '../services/productService';
import type { ProductSchema } from '../types/api';
// import dayjs from 'dayjs'; // Если есть поля с датами, пока не видно. Закомментировал, т.к. dayjs не используется

const { Title, Paragraph, Text } = Typography;

const ProductsPage: React.FC = () => {
  const [products, setProducts] = useState<ProductSchema[]>([]);
  const [totalProducts, setTotalProducts] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<ProductSchema | null>(null);
  const [modalLoading, setModalLoading] = useState(false);

  const loadProducts = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchProductsParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
      };
      const data = await fetchProducts(params);
      setProducts(data.items);
      setTotalProducts(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize })); // Добавил current и pageSize в setPagination
    } catch (err) {
      setError((err as Error).message);
      notification.error({ // Добавил уведомление об ошибке
        message: 'Ошибка загрузки продуктов',
        description: (err as Error).message,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProducts(pagination);
  }, [loadProducts, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const showProductDetails = async (productId: string) => {
    setIsModalVisible(true);
    setModalLoading(true);
    setSelectedProduct(null); // Сброс предыдущего
    try {
        const productData = await fetchProductById(productId);
        setSelectedProduct(productData);
    } catch (err) {
        notification.error({
            message: `Ошибка загрузки продукта ${productId}`,
            description: (err as Error).message
        });
        setIsModalVisible(false); // Закрыть модалку если ошибка
    } finally {
        setModalLoading(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedProduct(null);
  };

  const columns: ColumnsType<ProductSchema> = [
    {
      title: 'ID Продукта',
      dataIndex: 'product_id',
      key: 'product_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'Категория',
      dataIndex: 'product_category_name',
      key: 'product_category_name',
      ellipsis: true,
      render: (category?: string, record?: ProductSchema) => { // record был ProductSchema, сделал опциональным т.к. может не приходить
        if (!category) return <Tag>N/A</Tag>;
        const englishName = record?.category_translation?.product_category_name_english;
        return englishName && englishName !== category ? (
          <Tooltip title={`Оригинал: ${category}`}>
            <span>{englishName}</span>
          </Tooltip>
        ) : (
          category
        );
      },
    },
    {
      title: 'Вес (г)',
      dataIndex: 'product_weight_g',
      key: 'product_weight_g',
      align: 'right',
      render: (weight?: number) => weight ?? 'N/A', // было weight || 'N/A', заменил на ?? для обработки 0
    },
    {
      title: 'Размеры (см)',
      key: 'dimensions',
      align: 'center',
      render: (_, record: ProductSchema) => {
        const dims = [record.product_length_cm, record.product_width_cm, record.product_height_cm];
        const validDims = dims.filter(d => typeof d === 'number');
        return validDims.length > 0 ? validDims.join(' x ') : 'N/A';
      },
    },
     {
      title: 'Кол-во фото',
      dataIndex: 'product_photos_qty',
      key: 'product_photos_qty',
      align: 'center',
      render: (qty?: number) => qty ?? 'N/A',
    },
    {
      title: 'Действия',
      key: 'actions',
      align: 'center',
      width: 100,
      render: (_, record: ProductSchema) => (
        <Tooltip title="Просмотреть детали продукта">
          <Button icon={<EyeOutlined />} onClick={() => showProductDetails(record.product_id)} />
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>Список Продуктов</Title>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadProducts({ ...pagination, current: 1 })} 
            loading={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Здесь представлен список продуктов из базы данных с возможностью пагинации.
      </Paragraph>

      {error && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={products}
        rowKey="product_id"
        loading={loading}
        pagination={{...pagination, total: totalProducts}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />

      <Modal
        title={selectedProduct ? `Детали продукта: ${selectedProduct.product_id}` : "Загрузка деталей..."}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={<Button key="close" onClick={handleModalCancel}>Закрыть</Button>}
        width={720}
      >
        {modalLoading && <div style={{textAlign: 'center', padding: '30px'}}><Spin size="large" /></div>}
        {selectedProduct && !modalLoading && (
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="ID Продукта">{selectedProduct.product_id}</Descriptions.Item>
            <Descriptions.Item label="Категория (оригинал)">{selectedProduct.product_category_name || 'N/A'}</Descriptions.Item>
            {selectedProduct.category_translation && (
                 <Descriptions.Item label="Категория (Eng)">{selectedProduct.category_translation.product_category_name_english}</Descriptions.Item>
            )}
            <Descriptions.Item label="Длина имени">{selectedProduct.product_name_lenght ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Длина описания">{selectedProduct.product_description_lenght ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Количество фото">{selectedProduct.product_photos_qty ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Вес (г)">{selectedProduct.product_weight_g ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Длина (см)">{selectedProduct.product_length_cm ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Высота (см)">{selectedProduct.product_height_cm ?? 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="Ширина (см)">{selectedProduct.product_width_cm ?? 'N/A'}</Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </Card>
  );
};

export default ProductsPage; 