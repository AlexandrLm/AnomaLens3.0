import React, { useState, useEffect, useMemo } from 'react'; // Добавлен useMemo
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Timeline from '@mui/lab/Timeline';
import TimelineItem, { timelineItemClasses } from '@mui/lab/TimelineItem';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineContent from '@mui/lab/TimelineContent';
import TimelineDot from '@mui/lab/TimelineDot';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LabelList,
  CartesianGrid  // Для отображения значений на столбцах
} from 'recharts';
import { Anomaly, AnomalyContextResponse, OrderDetails } from './../types/anomaly';
import { getAnomalyContext } from './../services/api';

interface AnomalyDetailViewProps {
  anomaly: Anomaly | null;
  onClose?: () => void;
}

// ... (generateExplanation и OrderContextDisplay остаются без изменений) ...
const generateExplanation = (anomaly: Anomaly | null): string => {
  if (!anomaly) return "Данные аномалии отсутствуют.";
  let explanation = `Аномалия обнаружена детектором '${anomaly.detector_type || 'Неизвестный детектор'}'`;
  if (typeof anomaly.anomaly_score === 'number') { 
    explanation += ` со скором ${anomaly.anomaly_score.toFixed(4)}.`;
  } else {
    explanation += ` (скор недоступен).`;
  }
  const parsedDetails = anomaly.parsed_details;
  if (parsedDetails && Object.keys(parsedDetails).length > 0 && !parsedDetails.raw_details && !parsedDetails.parse_error) {
    const features = Object.keys(parsedDetails).filter(k => 
        !['order_id', 'order_item_id', 'product_id', 'seller_id', 'timestamp', 'detection_date', 'anomaly_score', 'detector_type', 'id'].includes(k)
    );
    if (features.length > 0) {
      explanation += ` Ключевые признаки: ${features.map(f => `${f}=${parsedDetails[f]}`).join(', ')}.`;
    }
  } else if (parsedDetails?.raw_details || parsedDetails?.parse_error) {
     explanation += ` Не удалось разобрать детали или они не предоставлены в структурированном виде.`
  } else {
     explanation += ` Специфичные признаки в деталях отсутствуют или не распарсены.`
  }
  return explanation;
};

const OrderContextDisplay: React.FC<{ order: OrderDetails }> = ({ order }) => {
  const formatDate = (dateString: string | Date | null | undefined): string => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };
  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6">Детали Заказа ({order.order_id})</Typography>
      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid size={{ xs: 12, md: 5 }}> 
          <Paper sx={{ p: 1.5, height: '100%' }}>
            <Typography variant="subtitle2">Статус и Даты:</Typography>
            <Timeline 
              sx={{
                  [`& .${timelineItemClasses.root}:before`]: { flex: 0, padding: 0, },
                  p: 0, mt: 1
              }}
            >
               <TimelineItem>
                <TimelineSeparator><TimelineDot color="primary" /><TimelineConnector /></TimelineSeparator>
                <TimelineContent sx={{py: '6px'}}><Typography variant="body2">Оформлен</Typography><Typography variant="caption">{formatDate(order.order_purchase_timestamp)}</Typography></TimelineContent>
              </TimelineItem>
              {order.order_approved_at && (
                <TimelineItem>
                  <TimelineSeparator><TimelineDot color="secondary" /><TimelineConnector /></TimelineSeparator>
                  <TimelineContent sx={{py: '6px'}}><Typography variant="body2">Одобрен</Typography><Typography variant="caption">{formatDate(order.order_approved_at)}</Typography></TimelineContent>
                </TimelineItem>
              )}
              {order.order_delivered_carrier_date && (
                <TimelineItem>
                  <TimelineSeparator><TimelineDot color="info" /><TimelineConnector /></TimelineSeparator>
                  <TimelineContent sx={{py: '6px'}}><Typography variant="body2">Передан перевозчику</Typography><Typography variant="caption">{formatDate(order.order_delivered_carrier_date)}</Typography></TimelineContent>
                </TimelineItem>
              )}
               {order.order_delivered_customer_date && (
                <TimelineItem>
                  <TimelineSeparator><TimelineDot color="success" /><TimelineConnector /></TimelineSeparator>
                  <TimelineContent sx={{py: '6px'}}><Typography variant="body2">Доставлен</Typography><Typography variant="caption">{formatDate(order.order_delivered_customer_date)}</Typography></TimelineContent>
                </TimelineItem>
              )}
              <TimelineItem>
                  <TimelineSeparator><TimelineDot color="warning" variant={!order.order_delivered_customer_date ? "filled" : "outlined"} /></TimelineSeparator>
                  <TimelineContent sx={{py: '6px'}}><Typography variant="body2">Плановая доставка</Typography><Typography variant="caption">{formatDate(order.order_estimated_delivery_date)}</Typography></TimelineContent>
                </TimelineItem>
            </Timeline>
            <Typography variant="body2" sx={{mt: 1}}>Текущий статус: <b>{order.order_status}</b></Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 7 }}> 
          <Paper sx={{ p: 1.5, mb: 2 }}>
              <Typography variant="subtitle2">Клиент:</Typography>
              <Typography>ID: {order.customer.customer_unique_id}</Typography>
              <Typography>Город: {order.customer.customer_city}, {order.customer.customer_state}</Typography>
              <Typography>Индекс: {order.customer.customer_zip_code_prefix}</Typography>
          </Paper>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Товары:</Typography>
          {order.items.map(item => (
            <Paper key={`${item.product?.product_id || 'unknown_product'}-${item.order_item_id}`} sx={{ p: 1.5, mb: 1 }}>
               <Box>
                   <Typography><b>{item.product?.product_category_name || 'Продукт'}</b> (ID: {item.product?.product_id || 'N/A'})</Typography>
                   <Typography>Индекс позиции: {item.order_item_id}</Typography> 
                   <Typography>Цена: {item.price.toFixed(2)} + Доставка: {item.freight_value.toFixed(2)} = <b>{(item.price + item.freight_value).toFixed(2)}</b></Typography>
                   <Typography>Продавец: {item.seller?.seller_id || 'N/A'} ({item.seller?.seller_city || 'N/A'}, {item.seller?.seller_state || 'N/A'})</Typography>
               </Box>
            </Paper>
          ))}
           {order.payments && order.payments.length > 0 && (
                <Paper sx={{ p: 1.5, mt: 2, mb: 2 }}>
                    <Typography variant="subtitle2">Платежи:</Typography>
                    {order.payments.map(p => (
                        <Typography key={p.payment_sequential}>{p.payment_sequential}. {p.payment_type} ({p.payment_installments}x) - Сумма: {p.payment_value.toFixed(2)}</Typography>
                    ))}
                </Paper>
           )}
           {order.reviews && order.reviews.length > 0 && (
                   <Paper sx={{ p: 1.5 }}>
                       <Typography variant="subtitle2">Отзывы:</Typography>
                       {order.reviews.map(r => (
                           <Box key={r.review_id ?? Math.random()} sx={{ mb: 1}}>
                               <Typography>Оценка: {r.review_score} {r.review_comment_title && `(${r.review_comment_title})`}</Typography>
                               {r.review_comment_message && <Typography variant="body2" sx={{ fontStyle: 'italic' }}>{r.review_comment_message}</Typography>}
                           </Box>
                       ))}
                   </Paper>
           )}
        </Grid>
      </Grid>
    </Box>
  );
};


const AnomalyDetailView: React.FC<AnomalyDetailViewProps> = ({ anomaly, onClose }) => {
  const [contextData, setContextData] = useState<AnomalyContextResponse | null>(null);
  const [loadingContext, setLoadingContext] = useState<boolean>(false);
  const [contextError, setContextError] = useState<string | null>(null);

  useEffect(() => {
    setContextData(null);
    setContextError(null);
    setLoadingContext(false);
    if (anomaly?.id) {
      const fetchContext = async () => {
        setLoadingContext(true);
        setContextError(null);
        try {
          const context = await getAnomalyContext(anomaly.id);
          setContextData(context);
        } catch (err) {
          setContextError(err instanceof Error ? err.message : 'Не удалось загрузить контекст');
        } finally {
          setLoadingContext(false);
        }
      };
      fetchContext();
    }
  }, [anomaly]);

  // Данные для графика числовых деталей
  const numericDetailsData = useMemo(() => {
    if (!anomaly?.parsed_details || anomaly.parsed_details.raw_details || anomaly.parsed_details.parse_error) {
      return [];
    }
    const data: { name: string; value: number }[] = [];
    const excludedKeys = ['order_id', 'order_item_id', 'product_id', 'seller_id', 'timestamp', 'detection_date', 'anomaly_score', 'detector_type', 'id'];
    
    for (const key in anomaly.parsed_details) {
      if (!excludedKeys.includes(key) && typeof anomaly.parsed_details[key] === 'number') {
        // Округляем для лучшего отображения, если нужно
        const value = parseFloat(Number(anomaly.parsed_details[key]).toFixed(4));
        data.push({ name: key, value: value });
      }
    }
    // Сортируем по значению для наглядности, например
    return data.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 10); // Показать топ-10
  }, [anomaly?.parsed_details]);


  if (!anomaly) {
    return null;
  }

  const explanation = generateExplanation(anomaly);
  const parsedDetails = anomaly.parsed_details;

  const renderDetectorDetailsText = (details: Record<string, any> | null | undefined) => {
    if (!details) return <Typography color="text.secondary">Дополнительные детали отсутствуют.</Typography>;
    if (details.raw_details || details.parse_error) return <Typography sx={{ fontFamily: 'monospace', wordBreak: 'break-all', fontSize: '0.8rem' }}>{details.raw_details || "Ошибка парсинга деталей."}</Typography>;
    
    const entries = Object.entries(details).filter(([key]) =>
      !['order_id', 'order_item_id', 'product_id', 'seller_id', 'timestamp', 'detection_date', 'anomaly_score', 'detector_type', 'id'].includes(key) &&
      typeof details[key] !== 'number' // Отображаем только нечисловые здесь, числа пойдут в график
    );

    if (entries.length === 0 && numericDetailsData.length === 0) { // Проверяем и числовые данные
      return <Typography color="text.secondary">Специфичные метрики детектора не найдены в деталях.</Typography>;
    }

    return entries.map(([key, value]) => (
      <Typography key={key} sx={{ wordBreak: 'break-all' }}>
        <b>{key}:</b> {String(value)}
      </Typography>
    ));
  };

  return (
    <Card sx={{ position: 'relative', overflow: 'visible' }}>
      {onClose && (
        <IconButton
          aria-label="Закрыть"
          onClick={onClose}
          sx={{ position: 'absolute', right: 8, top: 8, zIndex: 1, color: 'text.secondary' }}
        >
          <CloseIcon />
        </IconButton>
      )}
      <CardHeader
        title={`Детали Аномалии #${anomaly.id}`}
        titleTypographyProps={{ variant: 'h5' }}
        sx={{ pb: 1 }}
      />
      <CardContent sx={{ pt: 0 }}>
          <Divider sx={{ mb: 2 }} />
          <Grid container spacing={3}> {/* Увеличил spacing для лучшего вида с графиком */}
            <Grid size={{ xs: 12, md: 6 }}>
              <Typography variant="subtitle1" gutterBottom>Основная Информация:</Typography>
              <Typography><b>ID Заказа:</b> {anomaly.order_id || 'N/A'}</Typography>
              <Typography><b>ID Позиции:</b> {anomaly.order_item_id ?? 'N/A'}</Typography>
              <Typography><b>Время Детекции:</b> {anomaly.detection_date ? new Date(anomaly.detection_date).toLocaleString() : 'N/A'}</Typography>
              <Typography><b>Детектор:</b> {anomaly.detector_type || 'N/A'}</Typography>
              <Typography><b>Общий Скор:</b> {typeof anomaly.anomaly_score === 'number' ? anomaly.anomaly_score.toFixed(4) : 'N/A'}</Typography>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <Typography variant="subtitle1" gutterBottom>Детали Детектора (текст):</Typography>
              {renderDetectorDetailsText(parsedDetails)}
            </Grid>

            {numericDetailsData.length > 0 && (
              <Grid size={{ xs: 12 }}> {/* График на всю ширину в модалке */}
                <Paper sx={{ p: 2, mt: 2, height: '300px' }}> {/* Задаем высоту для графика */}
                  <Typography variant="subtitle1" gutterBottom>Ключевые числовые показатели:</Typography>
                  <ResponsiveContainer width="100%" height="90%">
                    <BarChart data={numericDetailsData} layout="vertical" margin={{ top: 5, right: 30, left: 50, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={150} /* Увеличил ширину для длинных имен */ />
                      <Tooltip formatter={(value: number) => [value.toFixed(4), "Значение"]} />
                      {/* <Legend /> */}
                      <Bar dataKey="value" fill="#8884d8" barSize={20}>
                        <LabelList dataKey="value" position="right" formatter={(value: number) => value.toFixed(3)} style={{ fill: 'black' }} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            )}
          </Grid>

          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Объяснение:</Typography>
          <Alert severity="info" icon={false} sx={{ mt: 1, mb: 2 }}>{explanation}</Alert>
          
          <Typography variant="subtitle1" sx={{ mb: 1 }}>Полный Контекст Заказа:</Typography>
          {loadingContext && <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}><CircularProgress /></Box>}
          {contextError && <Alert severity="error" sx={{ mb: 2 }}>{contextError}</Alert>}
          {contextData ? (
             <OrderContextDisplay order={contextData.order_details} />
          ) : (
             !loadingContext && !contextError &&
             <Typography color="text.secondary" sx={{ mb: 2 }}>Контекст не загружен или недоступен.</Typography>
          )}
      </CardContent>
    </Card>
  );
};

export default AnomalyDetailView;