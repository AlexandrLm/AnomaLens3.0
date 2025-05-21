import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Table, Spin, Alert, Card, Button, Row, Col, Tooltip, Modal, Descriptions, Rate, Tag, Space } from 'antd';
import { MessageOutlined, ReloadOutlined, EyeOutlined, FieldTimeOutlined } from '@ant-design/icons'; // MessageOutlined для отзывов
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import { fetchReviews, fetchReviewById } from '../services/reviewService';
import type { OrderReviewSchema, FetchReviewsParams } from '../types/api';
import dayjs from 'dayjs';
import { notification } from 'antd';

const { Title, Paragraph, Text } = Typography;

const formatDate = (dateString?: string | null) => {
  return dateString ? dayjs(dateString).format('YYYY-MM-DD HH:mm:ss') : 'N/A';
};

const ReviewsPage: React.FC = () => {
  const [reviews, setReviews] = useState<OrderReviewSchema[]>([]);
  const [totalReviews, setTotalReviews] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [pagination, setPagination] = useState<TablePaginationConfig>({
    current: 1,
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
  });

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedReview, setSelectedReview] = useState<OrderReviewSchema | null>(null);
  const [modalLoading, setModalLoading] = useState(false);

  const loadReviews = useCallback(async (currentPagination: TablePaginationConfig) => {
    setLoading(true);
    setError(null);
    try {
      const params: FetchReviewsParams = {
        skip: ((currentPagination.current || 1) - 1) * (currentPagination.pageSize || 10),
        limit: currentPagination.pageSize || 10,
      };
      const data = await fetchReviews(params);
      setReviews(data.items);
      setTotalReviews(data.total);
      setPagination(prev => ({...prev, total: data.total, current: currentPagination.current, pageSize: currentPagination.pageSize})); // Обновлено для сохранения current и pageSize
    } catch (err) {
      setError((err as Error).message);
      notification.error({ message: 'Ошибка загрузки отзывов', description: (err as Error).message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadReviews(pagination);
  }, [loadReviews, pagination.current, pagination.pageSize]);

  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination(prev => ({
        ...prev,
        current: newPagination.current,
        pageSize: newPagination.pageSize,
    }));
  };

  const showReviewDetails = async (reviewId: string) => {
    setIsModalVisible(true);
    setModalLoading(true);
    setSelectedReview(null);
    try {
        const reviewData = await fetchReviewById(reviewId);
        setSelectedReview(reviewData);
    } catch (err) {
        notification.error({
            message: `Ошибка загрузки отзыва ${reviewId}`,
            description: (err as Error).message
        });
        setIsModalVisible(false);
    } finally {
        setModalLoading(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedReview(null);
  };

  const columns: ColumnsType<OrderReviewSchema> = [
    {
      title: 'Review ID',
      dataIndex: 'review_id',
      key: 'review_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'Order ID',
      dataIndex: 'order_id',
      key: 'order_id',
      width: 280,
      ellipsis: true,
      render: (id: string) => <Text copyable={{text: id}}>{id}</Text>
    },
    {
      title: 'Оценка',
      dataIndex: 'review_score',
      key: 'review_score',
      align: 'center',
      render: (score: number) => <Rate disabled defaultValue={score} />,
      sorter: (a, b) => a.review_score - b.review_score,
    },
    {
      title: 'Заголовок',
      dataIndex: 'review_comment_title',
      key: 'review_comment_title',
      ellipsis: true,
      render: (title?: string) => title || <Text type="secondary">Без заголовка</Text>,
    },
    {
      title: 'Дата создания',
      dataIndex: 'review_creation_date',
      key: 'review_creation_date',
      render: (date: string) => formatDate(date),
      sorter: (a,b) => dayjs(a.review_creation_date).unix() - dayjs(b.review_creation_date).unix()
    },
    {
      title: 'Действия',
      key: 'actions',
      align: 'center',
      width: 100,
      render: (_, record: OrderReviewSchema) => (
        <Tooltip title="Просмотреть детали отзыва">
          <Button icon={<EyeOutlined />} onClick={() => showReviewDetails(record.review_id)} />
        </Tooltip>
      ),
    },
  ];

  return (
    <Card>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}><MessageOutlined style={{marginRight: 8}}/> Отзывы о Продуктах</Title>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => loadReviews({ ...pagination, current: 1 })} 
            loading={loading && !error}
            disabled={loading}
          >
            Обновить
          </Button>
        </Col>
      </Row>
      
      <Paragraph>
        Список отзывов, оставленных покупателями, с пагинацией.
      </Paragraph>

      {error && !loading && <Alert message="Ошибка загрузки данных" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Table
        columns={columns}
        dataSource={reviews}
        rowKey="review_id"
        loading={loading}
        pagination={{...pagination, total: totalReviews}}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }} 
      />

      <Modal
        title={selectedReview ? `Детали отзыва: ${selectedReview.review_id}` : "Загрузка деталей..."}
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={<Button key="close" onClick={handleModalCancel}>Закрыть</Button>}
        width={720}
      >
        {modalLoading && <div style={{textAlign: 'center', padding: '30px'}}><Spin size="large" /></div>}
        {selectedReview && !modalLoading && (
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="Review ID">{selectedReview.review_id}</Descriptions.Item>
            <Descriptions.Item label="Order ID">{selectedReview.order_id}</Descriptions.Item>
            <Descriptions.Item label="Оценка"><Rate disabled defaultValue={selectedReview.review_score} /></Descriptions.Item>
            <Descriptions.Item label="Заголовок комментария">{selectedReview.review_comment_title || <Text type="secondary">N/A</Text>}</Descriptions.Item>
            <Descriptions.Item label="Текст комментария">
              {selectedReview.review_comment_message ? 
                <Paragraph style={{maxHeight: '200px', overflowY: 'auto', whiteSpace: 'pre-wrap', margin: 0}}>
                  {selectedReview.review_comment_message}
                </Paragraph> : 
                <Text type="secondary">N/A</Text>
              }
            </Descriptions.Item>
            <Descriptions.Item label="Дата создания отзыва">
              <Space><FieldTimeOutlined />{formatDate(selectedReview.review_creation_date)}</Space>
            </Descriptions.Item>
            <Descriptions.Item label="Дата ответа на отзыв">
              <Space><FieldTimeOutlined />{formatDate(selectedReview.review_answer_timestamp)}</Space>
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </Card>
  );
};

export default ReviewsPage; 