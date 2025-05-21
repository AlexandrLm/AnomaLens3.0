import React from 'react';
import { Card, Typography, Tag, Button, Space, Tooltip } from 'antd';
import { EyeOutlined, DeleteOutlined, ExperimentOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import type { RootAnomalySchema } from '../types/api';

const { Text, Paragraph } = Typography;

interface AnomalyCardProps {
  anomaly: RootAnomalySchema;
  onViewDetails: (anomaly: RootAnomalySchema) => void;
  onDelete: (id: number) => void;
  onGetLlmExplanation: (id: number) => void;
  isLlmLoading?: boolean;
}

const AnomalyCard: React.FC<AnomalyCardProps> = ({ 
    anomaly, 
    onViewDetails, 
    onDelete, 
    onGetLlmExplanation,
    isLlmLoading 
}) => {
  const { id, order_id, order_item_id, detection_date, anomaly_score, detector_type } = anomaly;

  const scoreColor = anomaly_score !== null && anomaly_score !== undefined 
    ? anomaly_score > 0.7 ? 'volcano' : anomaly_score > 0.5 ? 'orange' : 'green' 
    : 'default';

  return (
    <Card 
      title={`Аномалия ID: ${id}`} 
      style={{ marginBottom: 16 }}
      actions={[
        <Tooltip title="Просмотр деталей">
          <Button icon={<EyeOutlined />} onClick={() => onViewDetails(anomaly)} key="view" />
        </Tooltip>,
        <Tooltip title="Получить объяснение LLM">
          <Button icon={<ExperimentOutlined />} onClick={() => onGetLlmExplanation(id)} key="llm" loading={isLlmLoading} />
        </Tooltip>,
        <Tooltip title="Удалить аномалию">
          <Button icon={<DeleteOutlined />} danger onClick={() => onDelete(id)} key="delete" />
        </Tooltip>,
      ]}
    >
      <Paragraph>
        <Text strong>ID Заказа:</Text> {order_id}
      </Paragraph>
      {order_item_id && (
        <Paragraph>
          <Text strong>ID Элемента Заказа:</Text> {order_item_id}
        </Paragraph>
      )}
      <Paragraph>
        <Text strong>Дата обнаружения:</Text> {dayjs(detection_date).format('YYYY-MM-DD HH:mm:ss')}
      </Paragraph>
      <Paragraph>
        <Text strong>Оценка:</Text>{' '}
        {anomaly_score !== null && anomaly_score !== undefined 
            ? <Tag color={scoreColor}>{anomaly_score.toFixed(2)}</Tag> 
            : <Tag>N/A</Tag>}
      </Paragraph>
      <Paragraph>
        <Text strong>Тип детектора:</Text> {detector_type}
      </Paragraph>
    </Card>
  );
};

export default AnomalyCard; 