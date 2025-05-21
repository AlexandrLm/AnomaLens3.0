import React, { useState } from 'react';
import { Typography, Button, Alert, notification, Card, Form, InputNumber, Row, Col, Space, Popconfirm } from 'antd';
import { SolutionOutlined, PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { detectMultilevelAnomalies } from '../services/multilevelService';
import TaskProgressDisplay from '../components/common/TaskProgressDisplay';
import type { TaskCreationResponse, DetectionParams } from '../types/api';

const { Title, Paragraph } = Typography;

const DEFAULT_THRESHOLDS: DetectionParams = {
  transaction_threshold: 0.6,
  behavior_threshold: 0.6,
  time_series_threshold: 0.6,
  final_threshold: 0.5,
  filter_period_days: 10000, 
};


const MultilevelDetectPage: React.FC = () => {
  const [form] = Form.useForm<DetectionParams>();
  const [taskInfo, setTaskInfo] = useState<TaskCreationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDetect = async (values: DetectionParams) => {
    setLoading(true);
    setError(null);
    setTaskInfo(null);
    try {
      const paramsToSubmit: Partial<DetectionParams> = {}; // Используем Partial для гибкости
      
      // Проверяем каждое значение и добавляем, только если оно не null и не undefined
      // Это также позволяет пользователю очистить поле, чтобы использовать значение по умолчанию на бэкенде (если API это поддерживает)
      if (values.transaction_threshold !== null && values.transaction_threshold !== undefined) paramsToSubmit.transaction_threshold = values.transaction_threshold;
      if (values.behavior_threshold !== null && values.behavior_threshold !== undefined) paramsToSubmit.behavior_threshold = values.behavior_threshold;
      if (values.time_series_threshold !== null && values.time_series_threshold !== undefined) paramsToSubmit.time_series_threshold = values.time_series_threshold;
      if (values.final_threshold !== null && values.final_threshold !== undefined) paramsToSubmit.final_threshold = values.final_threshold;
      if (values.filter_period_days !== null && values.filter_period_days !== undefined) paramsToSubmit.filter_period_days = values.filter_period_days;

      const response = await detectMultilevelAnomalies(paramsToSubmit as DetectionParams); // Приводим к DetectionParams, если уверены, что API примет Partial
      setTaskInfo(response);
      notification.success({
        message: 'Запуск детекции',
        description: response.message || `Задача детекции ${response.task_id} запущена.`,
      });
    } catch (err) {
      const errorMessage = (err as Error).message;
      setError(errorMessage);
      notification.error({
        message: 'Ошибка запуска детекции',
        description: errorMessage,
      });
    } finally {
      setLoading(false);
    }
  };
  
  const handleReset = () => {
    setTaskInfo(null);
    setError(null);
    setLoading(false);
    form.setFieldsValue(DEFAULT_THRESHOLDS);
  }


  return (
    <Card>
      <Title level={3} style={{marginBottom: '24px'}}>
        <SolutionOutlined style={{ marginRight: 8 }} />
        Детекция Аномалий Многоуровневой Системой
      </Title>
      <Paragraph>
        Эта операция запускает процесс обнаружения аномалий с использованием всех обученных моделей в многоуровневой системе.
        Вы можете настроить пороги срабатывания для каждого уровня и финальный порог, а также период для фильтрации данных.
      </Paragraph>

      {error && <Alert message="Ошибка" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}

      {!taskInfo ? (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleDetect}
          initialValues={DEFAULT_THRESHOLDS}
          style={{maxWidth: '600px', marginBottom: '24px'}}
        >
          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item name="transaction_threshold" label="Порог транзакционного уровня (0-1)" rules={[{ type: 'number', min:0, max:1}]}>
                <InputNumber step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item name="behavior_threshold" label="Порог поведенческого уровня (0-1)" rules={[{ type: 'number', min:0, max:1}]}>
                <InputNumber step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item name="time_series_threshold" label="Порог уровня временных рядов (0-1)" rules={[{ type: 'number', min:0, max:1}]}>
                <InputNumber step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item name="final_threshold" label="Финальный порог (0-1)" rules={[{ type: 'number', min:0, max:1}]}>
                <InputNumber step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item name="filter_period_days" label="Период фильтрации данных (дни)">
                <InputNumber min={1} style={{ width: '100%' }} placeholder="например, 30 (дней назад)"/>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Popconfirm
                title="Запустить детекцию аномалий?"
                description="Проверьте установленные параметры перед запуском."
                onConfirm={() => form.submit()}
                okText="Да, запустить"
                cancelText="Отмена"
                disabled={loading}
            >
                <Button 
                    type="primary" 
                    icon={<PlayCircleOutlined />} 
                    loading={loading}
                    size="large"
                    disabled={loading}
                    htmlType="button" 
                >
                    Запустить детекцию
                </Button>
            </Popconfirm>
          </Form.Item>
        </Form>
      ) : (
         <Button 
            icon={<ReloadOutlined />} 
            onClick={handleReset}
            size="large"
            style={{marginBottom: 20}}
          >
            Запустить новую задачу детекции
        </Button>
      )}
      
      {taskInfo && (
         <TaskProgressDisplay 
            taskInfo={taskInfo} 
            onReset={handleReset} 
            allowReset={true}
        />
      )}
    </Card>
  );
};

export default MultilevelDetectPage; 