import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Spin, Alert, Card, Descriptions, Tag, Button, Row, Col, Space, Tooltip, Collapse } from 'antd';
import { ReloadOutlined, CheckCircleOutlined, CloseCircleOutlined, QuestionCircleOutlined, WarningOutlined, InfoCircleOutlined, FileTextOutlined } from '@ant-design/icons';
import { fetchMultilevelStatus } from '../services/multilevelService';
import type { MultilevelStatus, DetectorStatus, DetectorLevelStatus } from '../types/api';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

const renderBooleanStatus = (status?: boolean | null, trueText = "Да", falseText = "Нет", unknownText = "N/A") => {
  if (status === true) {
    return <Tag color="success" icon={<CheckCircleOutlined />}>{trueText}</Tag>;
  }
  if (status === false) {
    return <Tag color="error" icon={<CloseCircleOutlined />}>{falseText}</Tag>;
  }
  return <Tag icon={<QuestionCircleOutlined />}>{unknownText}</Tag>;
};

const JsonViewer: React.FC<{ jsonData: Record<string, any> | null | undefined, title: string }> = ({ jsonData, title }) => {
  if (!jsonData || Object.keys(jsonData).length === 0) {
    return null;
  }
  return (
    <Collapse ghost defaultActiveKey={[]} style={{marginTop: 8}} size="small">
      <Panel header={<Text type="secondary"><InfoCircleOutlined /> {title}</Text>} key="1">
        <SyntaxHighlighter language="json" style={atomOneDark} customStyle={{ maxHeight: '250px', overflowY: 'auto', fontSize: '12px', padding: '10px', borderRadius: '4px' }}>
          {JSON.stringify(jsonData, null, 2)}
        </SyntaxHighlighter>
      </Panel>
    </Collapse>
  );
};

const DetectorStatusCard: React.FC<{ name: string, status: DetectorStatus }> = ({ name, status }) => {
  let cardBorderColor = '#d9d9d9'; // Default
  let titleIcon = <FileTextOutlined />;

  if (status.error_message) {
    cardBorderColor = '#ffccc7'; // Light red for error
    titleIcon = <WarningOutlined style={{ color: 'red' }} />;
  } else if (status.is_trained && status.can_load) {
    cardBorderColor = '#b7eb8f'; // Light green for success
    titleIcon = <CheckCircleOutlined style={{ color: 'green' }}/>;
  } else if (!status.is_trained || !status.can_load) {
    cardBorderColor = '#ffe58f'; // Light yellow for warning/not fully ready
    titleIcon = <QuestionCircleOutlined style={{ color: 'orange' }} />;
  }

  return (
    <Card 
        title={<Space>{titleIcon}<Text strong>{name}</Text></Space>} 
        size="small" 
        style={{ marginBottom: 16, border: `1px solid ${cardBorderColor}` }}
        headStyle={{ background: status.is_trained && status.can_load && !status.error_message ? '#f6ffed' : (status.error_message ? '#fff1f0' : '#fafafa') }}
    >
      <Descriptions bordered column={1} size="small" layout="horizontal">
        <Descriptions.Item label="Тип детектора" span={1}>{status.detector_type || 'N/A'}</Descriptions.Item>
        <Descriptions.Item label="Обучен" span={1}>{renderBooleanStatus(status.is_trained)}</Descriptions.Item>
        <Descriptions.Item label="Имя файла" span={1}>
          {status.model_filename ? (
            <Tooltip title={status.model_filename}><Text ellipsis>{status.model_filename}</Text></Tooltip>
          ) : 'N/A'}
        </Descriptions.Item>
        <Descriptions.Item label="Существует" span={1}>{renderBooleanStatus(status.exists)}</Descriptions.Item>
        <Descriptions.Item label="Загружаем" span={1}>{renderBooleanStatus(status.can_load)}</Descriptions.Item>
        {status.error_message && (
          <Descriptions.Item label="Ошибка" span={1}>
            <Alert message={status.error_message} type="error" showIcon banner style={{padding: '4px 8px'}}/>
          </Descriptions.Item>
        )}
      </Descriptions>
      <JsonViewer jsonData={status.params_from_config} title="Параметры (из конфигурации)" />
      <JsonViewer jsonData={status.internal_params} title="Внутренние параметры модели" />
    </Card>
  );
};


const LevelStatusDisplay: React.FC<{ levelName: string, levelStatus: DetectorLevelStatus }> = ({ levelName, levelStatus }) => {
  const detectorNames = Object.keys(levelStatus);
  const levelTitle = levelName.replace('_level', '').replace('_', ' ');

  return (
    <Card title={<Title level={3} style={{ margin: 0, textTransform: 'capitalize' }}>Уровень: {levelTitle}</Title>} style={{ marginBottom: 24 }} bordered={false} bodyStyle={{paddingTop: 16}}>
      {detectorNames.length === 0 ? (
        <Paragraph>Нет детекторов, настроенных для уровня "{levelTitle}".</Paragraph>
      ) : (
        <Row gutter={[16, 16]}>
          {detectorNames.map(detectorName => (
            <Col key={detectorName} xs={24} sm={24} md={12} lg={12} xl={12}>
              <DetectorStatusCard name={detectorName} status={levelStatus[detectorName]} />
            </Col>
          ))}
        </Row>
      )}
    </Card>
  );
};


const MultilevelStatusPage: React.FC = () => {
  const [status, setStatus] = useState<MultilevelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMultilevelStatus();
      setStatus(data);
    } catch (err) {
      setError((err as Error).message);
      setStatus(null); 
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  if (loading) {
    return <Spin tip="Загрузка статуса многоуровневой системы..." size="large" style={{ display: 'flex', justifyContent:'center', alignItems:'center', minHeight: '50vh' }} />;
  }

  return (
    <div style={{padding: '24px'}}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={2}>Статус Многоуровневой Системы Детекции</Title>
          <Paragraph type="secondary">Обзор состояния и конфигурации каждого детектора на различных уровнях системы.</Paragraph>
        </Col>
        <Col>
          <Button onClick={loadStatus} icon={<ReloadOutlined />} type="primary">
            Обновить статус
          </Button>
        </Col>
      </Row>

      {error && (
        <Alert
          message="Ошибка загрузки статуса"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 24 }}
          onClose={() => setError(null)}
        />
      )}

      {!status && !loading && !error && (
         <Alert message="Нет данных" description="Не удалось получить статус системы или система еще не сконфигурирована." type="info" showIcon />
      )}

      {status && Object.keys(status).length > 0 ? (
        <>
          {Object.entries(status).map(([levelKey, levelData]) => {
            if (Object.keys(levelData).length > 0) { // Показываем уровень, только если в нем есть детекторы
                return <LevelStatusDisplay key={levelKey} levelName={levelKey} levelStatus={levelData as DetectorLevelStatus} />;
            }
            return null;
          })}
        </>
      ) : !loading && !error && (
        <Alert message="Нет сконфигурированных уровней" description="В конфигурации системы не найдено ни одного уровня с детекторами." type="warning" showIcon />
      )}
    </div>
  );
};

export default MultilevelStatusPage;