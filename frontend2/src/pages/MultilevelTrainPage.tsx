import React, { useState } from 'react';
import { Typography, Button, Alert, notification, Card, Popconfirm } from 'antd';
import { ExperimentOutlined, PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { trainMultilevelSystem } from '../services/multilevelService';
import TaskProgressDisplay from '../components/common/TaskProgressDisplay';
import type { TaskCreationResponse } from '../types/api';

const { Title, Paragraph } = Typography;

const MultilevelTrainPage: React.FC = () => {
  const [taskInfo, setTaskInfo] = useState<TaskCreationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setTaskInfo(null); 
    try {
      const response = await trainMultilevelSystem();
      setTaskInfo(response);
      notification.success({
        message: 'Запуск обучения',
        description: response.message || `Задача обучения ${response.task_id} запущена.`,
      });
    } catch (err) {
      const errorMessage = (err as Error).message;
      setError(errorMessage);
      notification.error({
        message: 'Ошибка запуска обучения',
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
  }

  return (
    <Card>
      <Title level={3} style={{marginBottom: '24px'}}>
        <ExperimentOutlined style={{ marginRight: 8 }} />
        Обучение Всех Моделей Многоуровневой Системы
      </Title>
      <Paragraph>
        Эта операция запускает процесс обучения для всех детекторов, сконфигурированных в многоуровневой системе.
        Обучение будет происходить на всех доступных данных в базе данных.
        Процесс может занять значительное время в зависимости от объема данных и сложности моделей.
      </Paragraph>
      <Paragraph>
        После запуска вы сможете отслеживать статус выполнения задачи ниже.
      </Paragraph>

      {error && <Alert message="Ошибка" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}

      {!taskInfo ? (
        <Popconfirm
          title="Запустить обучение всех моделей?"
          description="Это может быть длительной операцией. Вы уверены?"
          onConfirm={handleTrain}
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
          >
            Запустить обучение всех моделей
          </Button>
        </Popconfirm>
      ) : (
        <Button 
            icon={<ReloadOutlined />} 
            onClick={handleReset}
            size="large"
            style={{marginBottom: 20}}
          >
            Запустить новую задачу обучения
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

export default MultilevelTrainPage; 