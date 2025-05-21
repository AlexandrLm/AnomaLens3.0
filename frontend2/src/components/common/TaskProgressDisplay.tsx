import React, { useState, useEffect, useCallback } from 'react';
import { Alert, Progress, Typography, Button, Space, Spin, Descriptions, Tag, Card, notification } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined, InfoCircleOutlined, FieldTimeOutlined, WarningOutlined } from '@ant-design/icons';
import { fetchTaskStatus } from '../../services/taskService';
import type { TaskStatusResult, TaskCreationResponse } from '../../types/api';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import dayjs from 'dayjs';

const { Text, Paragraph } = Typography;

interface TaskProgressDisplayProps {
  taskInfo: TaskCreationResponse;
  onTaskCompleted?: (result: TaskStatusResult) => void;
  onTaskFailed?: (result: TaskStatusResult) => void;
  onReset?: () => void;
  allowReset?: boolean;
}

const POLLING_INTERVAL = 3000;

const TaskProgressDisplay: React.FC<TaskProgressDisplayProps> = ({
  taskInfo,
  onTaskCompleted,
  onTaskFailed,
  onReset,
  allowReset = true,
}) => {
  const [statusResult, setStatusResult] = useState<TaskStatusResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(true);

  const getStatus = useCallback(async (currentTaskInfo: TaskCreationResponse) => {
    if (!currentTaskInfo.task_id) return;
    setLoading(true);
    setError(null);
    try {
      const endpoint = currentTaskInfo.status_endpoint || `/api/tasks/task_status/${currentTaskInfo.task_id}`;
      const relativeEndpoint = endpoint.startsWith('/api') ? endpoint.substring(4) : endpoint;
      
      const result = await fetchTaskStatus(relativeEndpoint);
      setStatusResult(result);

      if (result.status === 'completed' || result.status === 'completed_no_data' || result.status === 'completed_with_errors') {
        setIsPolling(false);
        if (onTaskCompleted) onTaskCompleted(result);
        notification.success({
            message: `Задача ${currentTaskInfo.task_id} завершена`,
            description: `Статус: ${result.status}. ${result.details}`,
        });
      } else if (result.status === 'failed') {
        setIsPolling(false);
        if (onTaskFailed) onTaskFailed(result);
         notification.error({
            message: `Задача ${currentTaskInfo.task_id} провалена`,
            description: `Ошибка: ${result.error_type || 'Unknown error'}. ${result.details}`,
        });
      }
    } catch (err) {
      setError((err as Error).message);
      setIsPolling(false); 
       notification.error({
            message: 'Ошибка получения статуса задачи',
            description: (err as Error).message,
        });
    } finally {
      setLoading(false);
    }
  }, [onTaskCompleted, onTaskFailed]);

  useEffect(() => {
    if (taskInfo && !statusResult) {
        setStatusResult(prev => prev || {
            task_id: taskInfo.task_id,
            status: taskInfo.initial_status as string,
            start_time: new Date().toISOString(),
            details: taskInfo.message as string,
            result: undefined,
            end_time: undefined,
            error_type: undefined,
        } as TaskStatusResult );
    }

    if (isPolling && taskInfo.task_id) {
      if (!statusResult || (statusResult.status !== 'completed' && statusResult.status !== 'completed_no_data' && statusResult.status !== 'completed_with_errors' && statusResult.status !== 'failed')) {
        getStatus(taskInfo); 
        const intervalId = setInterval(() => getStatus(taskInfo), POLLING_INTERVAL);
        return () => clearInterval(intervalId);
      } else {
        setIsPolling(false);
      }
    }
  }, [taskInfo, getStatus, isPolling, statusResult]);


  const getProgressPercent = () => {
    if (!statusResult) return 0;
    switch (statusResult.status) {
      case 'pending': return 10;
      case 'processing': return 50;
      case 'completed':
      case 'completed_no_data':
      case 'completed_with_errors': return 100;
      case 'failed': return 100;
      default: return 0;
    }
  };

  const getProgressStatus = (): "success" | "exception" | "normal" | "active" => {
    if (!statusResult) return "normal";
    switch (statusResult.status) {
      case 'completed':
      case 'completed_no_data': return "success";
      case 'completed_with_errors': return "exception"; 
      case 'failed': return "exception";
      case 'processing': return "active";
      default: return "normal";
    }
  };
  
  const getOverallStatusTag = () => {
    if (!statusResult) return <Tag icon={<LoadingOutlined />} color="processing">Загрузка...</Tag>;
    switch (statusResult.status) {
        case 'pending': return <Tag icon={<LoadingOutlined />} color="default">В ожидании</Tag>;
        case 'processing': return <Tag icon={<LoadingOutlined spin />} color="processing">Выполняется</Tag>;
        case 'completed': return <Tag icon={<CheckCircleOutlined />} color="success">Завершено</Tag>;
        case 'completed_no_data': return <Tag icon={<InfoCircleOutlined />} color="warning">Завершено (нет данных)</Tag>;
        case 'completed_with_errors': return <Tag icon={<WarningOutlined />} color="warning">Завершено (с ошибками)</Tag>;
        case 'failed': return <Tag icon={<CloseCircleOutlined />} color="error">Ошибка</Tag>;
        default: return <Tag color="default">{statusResult.status}</Tag>;
    }
  };

  if (!taskInfo || !taskInfo.task_id) {
    return null;
  }

  return (
    <Card 
        title={`Статус Задачи: ${taskInfo.task_id}`} 
        style={{ marginTop: 20 }}
        extra={allowReset && onReset && (!isPolling || statusResult?.status === 'failed' || statusResult?.status === 'completed' || statusResult?.status === 'completed_no_data' || statusResult?.status === 'completed_with_errors') && (
            <Button onClick={onReset}>Запустить новую задачу</Button>
        )}
    >
      {loading && !statusResult && <Spin tip="Получение статуса..." />}
      {error && <Alert message="Ошибка" description={error} type="error" showIcon />}
      
      {statusResult && (
        <>
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="ID Задачи"><Text copyable>{statusResult.task_id}</Text></Descriptions.Item>
            <Descriptions.Item label="Текущий статус">{getOverallStatusTag()}</Descriptions.Item>
            <Descriptions.Item label="Сообщение/Детали">
              <Paragraph style={{whiteSpace: 'pre-wrap', maxHeight: '150px', overflowY: 'auto'}}>
                {statusResult.details || taskInfo.message}
              </Paragraph>
            </Descriptions.Item>
            <Descriptions.Item label="Прогресс">
                <Progress percent={getProgressPercent()} status={getProgressStatus()} strokeWidth={10}/>
            </Descriptions.Item>
            {statusResult.start_time && (
                 <Descriptions.Item label="Время начала">
                    <FieldTimeOutlined /> {dayjs(statusResult.start_time).format('YYYY-MM-DD HH:mm:ss')}
                </Descriptions.Item>
            )}
            {statusResult.end_time && (
                 <Descriptions.Item label="Время окончания">
                    <FieldTimeOutlined /> {dayjs(statusResult.end_time).format('YYYY-MM-DD HH:mm:ss')}
                </Descriptions.Item>
            )}
            {statusResult.error_type && (
                 <Descriptions.Item label="Тип ошибки">
                    <Tag color="error">{statusResult.error_type}</Tag>
                </Descriptions.Item>
            )}
          </Descriptions>
            {statusResult.result && Object.keys(statusResult.result).length > 0 && (
            <Card size="small" title="Результат выполнения" style={{marginTop: 16}}>
                <SyntaxHighlighter language="json" style={atomOneDark} customStyle={{ maxHeight: '300px', overflowY: 'auto', fontSize: '12px' }}>
                    {JSON.stringify(statusResult.result, null, 2)}
                </SyntaxHighlighter>
            </Card>
            )}
        </>
      )}
    </Card>
  );
};

export default TaskProgressDisplay; 