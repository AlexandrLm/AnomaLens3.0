import React, { useState, useEffect, useCallback } from 'react';
import {
  Typography,
  Spin,
  Alert,
  Button,
  Form,
  Input,
  InputNumber,
  Select,
  Card,
  Row,
  Col,
  Space,
  Divider,
  notification,
  Tooltip,
  Popconfirm
} from 'antd';
import { ReloadOutlined, SaveOutlined, PlusOutlined, DeleteOutlined, InfoCircleOutlined } from '@ant-design/icons';
import {
  fetchMultilevelConfig,
  updateMultilevelConfig,
  fetchAvailableDetectors,
} from '../services/multilevelService';
import type {
  MultilevelConfig,
  MultilevelDetectorConfigEntry,
  AvailableDetectorsResponse,
} from '../types/api';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';

const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const LEVEL_KEYS = [
  'transaction_level',
  'behavior_level',
  'time_series_level',
] as const;

const LEVEL_NAMES_MAP: Record<string, string> = {
  transaction_level: 'Транзакционный Уровень',
  behavior_level: 'Поведенческий Уровень',
  time_series_level: 'Уровень Временных Рядов',
  combination_weights: 'Веса Комбинации Уровней'
};

interface DetectorFormFieldProps {
  levelKey: typeof LEVEL_KEYS[number];
  field: { name: number; key: number; fieldKey?: number };
  remove: (index: number) => void;
  availableDetectors: AvailableDetectorsResponse | null;
}

const DetectorFormField: React.FC<DetectorFormFieldProps> = ({
  levelKey,
  field,
  remove,
  availableDetectors,
}) => {
  const form = Form.useFormInstance<MultilevelConfig>();
  const currentDetectorType = Form.useWatch([levelKey, field.name, 'type'], form);

  const initialDetectorConfig = form.getFieldValue([levelKey, field.name]);
  let additionalParams: Record<string, any> = {};
  if (initialDetectorConfig && typeof initialDetectorConfig === 'object') {
    additionalParams = { ...initialDetectorConfig };
    delete additionalParams.type;
    delete additionalParams.model_filename;
    delete additionalParams.weight;
  }

  return (
    <Card size="small" style={{ marginBottom: 16, background: '#fafafa' }}>
      <Row gutter={16} align="middle">
        <Col xs={24} md={7}>
          <Form.Item
            label="Тип детектора"
            name={[field.name, 'type']}
            rules={[{ required: true, message: 'Выберите тип' }]}
          >
            <Select placeholder="Тип" allowClear showSearch>
              {availableDetectors && availableDetectors[levelKey]?.map((type) => (
                <Option key={type} value={type}>
                  {type}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Col>
        <Col xs={24} md={7}>
          <Form.Item
            label="Имя файла модели"
            name={[field.name, 'model_filename']}
            tooltip="Опционально. Если не указано, может использоваться имя по умолчанию или автогенерация."
          >
            <Input placeholder="например, my_model.joblib" />
          </Form.Item>
        </Col>
        <Col xs={24} md={4}>
          <Form.Item
            label="Вес"
            name={[field.name, 'weight']}
            rules={[{ type: 'number', min: 0, max: 1, message: 'От 0 до 1' }]}
          >
            <InputNumber step={0.1} placeholder="0.0-1.0" style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col xs={24} md={5}>
          <Form.Item
            label={
              <Space>
                Доп. параметры (JSON)
                <Tooltip title='Введите JSON объект с дополнительными параметрами конфигурации для этого детектора. Например: {"n_estimators": 100, "random_state": 42}'>
                  <InfoCircleOutlined style={{ color: 'rgba(0,0,0,.45)' }} />
                </Tooltip>
              </Space>
            }
            name={[field.name, 'additional_params_json']}
            initialValue={Object.keys(additionalParams).length > 0 ? JSON.stringify(additionalParams, null, 2) : ''}
          >
            <Input.TextArea rows={3} placeholder='{ "param1": "value1" }' />
          </Form.Item>
        </Col>
        <Col xs={24} md={1} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', paddingTop: '20px' }}>
          <Tooltip title="Удалить детектор">
            <Button danger onClick={() => remove(field.name)} icon={<DeleteOutlined />} type="text" />
          </Tooltip>
        </Col>
      </Row>
      {currentDetectorType && (
        <Paragraph type="secondary" style={{ fontSize: '12px', marginTop: '-10px', marginBottom: '10px' }}>
          Выбран детектор: <Text strong>{currentDetectorType}</Text>. Убедитесь, что указаны корректные доп. параметры, если они требуются.
        </Paragraph>
      )}
    </Card>
  );
};

const MultilevelConfigPage: React.FC = () => {
  const [form] = Form.useForm<MultilevelConfig>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableDetectors, setAvailableDetectors] = useState<AvailableDetectorsResponse | null>(null);
  const [initialConfigJson, setInitialConfigJson] = useState<string>(""); // Для отображения исходного JSON

  const loadConfigAndDetectors = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [configData, detectorsData] = await Promise.all([
        fetchMultilevelConfig(),
        fetchAvailableDetectors(),
      ]);
      form.setFieldsValue(configData);
      setInitialConfigJson(JSON.stringify(configData, null, 2));
      setAvailableDetectors(detectorsData);
    } catch (err) {
      setError((err as Error).message);
      notification.error({
        message: 'Ошибка загрузки конфигурации',
        description: (err as Error).message,
      });
    } finally {
      setLoading(false);
    }
  }, [form]);

  useEffect(() => {
    loadConfigAndDetectors();
  }, [loadConfigAndDetectors]);

  const handleSaveConfig = async (values: MultilevelConfig) => {
    setSaving(true);
    setError(null);
    try {
      const cleanedValues = JSON.parse(JSON.stringify(values)) as MultilevelConfig; // Глубокое копирование

      for (const levelKey of LEVEL_KEYS) {
        const detectors = cleanedValues[levelKey];
        if (detectors && Array.isArray(detectors)) {
          for (let i = 0; i < detectors.length; i++) {
            let detector = detectors[i] as MultilevelDetectorConfigEntry & { additional_params_json?: string };
            
            if (detector.additional_params_json) {
              try {
                const additionalParams = JSON.parse(detector.additional_params_json);
                
                // Создаем новый объект детектора без additional_params_json, но со всеми остальными свойствами
                const { additional_params_json, ...baseDetectorProperties } = detector;
                
                // Собираем итоговый объект детектора
                const newDetectorData: MultilevelDetectorConfigEntry = {
                  ...baseDetectorProperties, // type, model_filename, weight
                };

                // Добавляем дополнительные параметры, избегая перезаписи основных
                for (const paramKey in additionalParams) {
                  if (paramKey !== 'type' && paramKey !== 'model_filename' && paramKey !== 'weight') {
                    (newDetectorData as any)[paramKey] = additionalParams[paramKey];
                  } else {
                    notification.warning({
                      message: `Дополнительный параметр '${paramKey}' конфликтует с основным полем и будет проигнорирован.`,
                      description: `Детектор: ${detector.type}, Уровень: ${levelKey}`,
                    });
                  }
                }
                detectors[i] = newDetectorData; // Заменяем старый объект детектора новым
              } catch (e) {
                notification.error({
                  message: `Ошибка парсинга JSON для доп. параметров детектора '${detector.type || 'Unknown'}' на уровне '${levelKey}' (детектор #${i + 1})`,
                  description: 'Пожалуйста, исправьте JSON или удалите его.',
                });
                setSaving(false);
                throw new Error('Invalid JSON in additional_params_json');
              }
            } else {
              // Если additional_params_json нет или он пустой, убедимся, что его нет в итоговом объекте
              const { additional_params_json, ...rest } = detector;
              detectors[i] = rest;
            }
          }
        }
      }

      await updateMultilevelConfig(cleanedValues);
      notification.success({
        message: 'Конфигурация успешно сохранена',
      });
      setInitialConfigJson(JSON.stringify(cleanedValues, null, 2));
    } catch (err) {
      if ((err as Error).message !== 'Invalid JSON in additional_params_json') {
        setError((err as Error).message);
        notification.error({
          message: 'Ошибка сохранения конфигурации',
          description: (err as Error).message,
        });
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <Spin tip="Загрузка конфигурации..." size="large" style={{ display: 'block', marginTop: 50 }} />;
  }

  if (error && !form.getFieldsValue().combination_weights) { // Если ошибка и форма пуста
    return (
        <Alert
        message="Ошибка загрузки данных"
        description={error}
        type="error"
        showIcon
        action={
            <Button size="small" danger onClick={loadConfigAndDetectors}>
            Попробовать снова
            </Button>
        }
        />
    );
  }

  return (
    <Form form={form} layout="vertical" onFinish={handleSaveConfig} initialValues={{ combination_weights: { transaction: 0.4, behavior: 0.4, time_series: 0.2 }}}> 
      <Space style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <Title level={3} style={{margin: 0}}>Конфигурация Многоуровневой Системы</Title>
        <Space>
          <Button onClick={loadConfigAndDetectors} icon={<ReloadOutlined />} loading={loading}>
            Обновить/Сбросить
          </Button>
          <Popconfirm
            title="Сохранить конфигурацию?"
            description="Вы уверены, что хотите сохранить текущие изменения?"
            onConfirm={form.submit}
            okText="Да, сохранить"
            cancelText="Отмена"
          >
            <Button type="primary" icon={<SaveOutlined />} loading={saving}>
              Сохранить конфигурацию
            </Button>
          </Popconfirm>
        </Space>
      </Space>

      {error && ( // Показываем ошибку, если она есть, даже если форма загружена с предыдущими данными
        <Alert
          message="Ошибка"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={24}>
        <Col xs={24} md={16}>
            {LEVEL_KEYS.map((levelKey) => (
                <Card 
                  key={levelKey} 
                  title={LEVEL_NAMES_MAP[levelKey] || levelKey} 
                  style={{ marginBottom: 24 }}  
                  styles={{ header: {background: '#f0f2f5'} }}
                  variant="outlined"
                >
                <Form.List name={levelKey}>
                    {(fields, { add, remove }) => (
                    <>
                        {fields.map((field) => (
                          <DetectorFormField
                            key={field.key}
                            levelKey={levelKey}
                            field={field}
                            remove={remove}
                            availableDetectors={availableDetectors}
                          />
                        ))}
                        <Form.Item>
                        <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                            Добавить детектор на уровень "{LEVEL_NAMES_MAP[levelKey]}"
                        </Button>
                        </Form.Item>
                    </>
                    )}
                </Form.List>
                </Card>
            ))}

            <Card 
              title={LEVEL_NAMES_MAP['combination_weights'] || 'Веса комбинации'} 
              styles={{ header: {background: '#f0f2f5'} }}
              variant="outlined"
            >
                <Row gutter={16}>
                {Object.keys(form.getFieldValue('combination_weights') || { transaction:0, behavior:0, time_series:0 }).map((weightKey) => (
                    <Col xs={24} sm={8} key={weightKey}>
                    <Form.Item
                        label={<Text style={{textTransform: 'capitalize'}}>{weightKey}</Text>}
                        name={['combination_weights', weightKey]}
                        rules={[
                            { required: true, message: 'Обязательное поле' },
                            { type: 'number', min: 0, max: 1, message: 'От 0 до 1' }
                        ]}
                    >
                        <InputNumber step={0.1} placeholder="0.0-1.0" style={{ width: '100%' }} />
                    </Form.Item>
                    </Col>
                ))}
                </Row>
                 <Paragraph type="secondary">
                    Сумма весов должна быть равна 1 для корректной нормализации, но система может работать и с другими значениями, интерпретируя их как относительные вклады.
                </Paragraph>
            </Card>
        </Col>
        <Col xs={24} md={8}>
             <Card 
                title="Исходная/Текущая Конфигурация (JSON)" 
                style={{position: 'sticky', top: '88px'}}
                variant="outlined"
              >
                <Paragraph type="secondary">
                    Это JSON представление конфигурации, которое будет отправлено на сервер. Полезно для отладки.
                </Paragraph>
                <div style={{maxHeight: 'calc(100vh - 200px)', overflowY: 'auto', background: '#282c34', borderRadius: '4px', padding: '10px' }}>
                    <SyntaxHighlighter language="json" style={atomOneDark} showLineNumbers customStyle={{ margin: 0 }}>
                        {initialConfigJson || "{}"}
                    </SyntaxHighlighter>
                </div>
             </Card>
        </Col>
      </Row>
    </Form>
  );
};

export default MultilevelConfigPage;