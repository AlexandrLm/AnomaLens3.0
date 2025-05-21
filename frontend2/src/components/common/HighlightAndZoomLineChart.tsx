import React from 'react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import type { 
    LineProps as RechartsLineProps, // Переименовываем, чтобы избежать конфликта с нашим интерфейсом
    XAxisProps,
    YAxisProps,
    TooltipProps,
    LegendProps as RechartsLegendProps, // Переименовываем
    CartesianGridProps,
    BrushProps as RechartsBrushProps // Переименуем, чтобы наш BrushProps мог быть специфичнее
} from 'recharts';

// Определим наш интерфейс для BrushProps, выбирая только нужные поля
// чтобы избежать проблем с типами при spread
interface CustomBrushProps {
  dataKey: RechartsBrushProps['dataKey']; // Обязательно
  height?: number;
  stroke?: string;
  startIndex?: number;
  endIndex?: number;
  onChange?: RechartsBrushProps['onChange'];
  // Можно добавить другие нужные свойства из RechartsBrushProps
  fill?: string;
  travellerWidth?: number;
  gap?: number;
  padding?: RechartsBrushProps['padding'];
  x?: number;
  y?: number;
  alwaysShowText?: boolean;
}

// Используем React.ComponentProps для получения пропсов RechartsLineChart
export interface HighlightAndZoomLineChartProps extends Omit<React.ComponentProps<typeof RechartsLineChart>, 'children'> {
  lines: Array<LineConfig>; 
  xAxisProps?: XAxisProps;
  yAxisProps?: YAxisProps;
  tooltipProps?: TooltipProps<any, any>; 
  legendProps?: RechartsLegendProps; // Используем переименованный тип
  cartesianGridProps?: CartesianGridProps;
  brushProps?: CustomBrushProps; // Используем наш CustomBrushProps
  showResponsiveContainer?: boolean; 
}

// Определим отдельный тип для конфигурации линии, чтобы избежать слишком вложенных типов
interface LineConfig extends Omit<RechartsLineProps, 'key'> {
    key: string | number; // React key
    name: string; // name для Legend
    dataKey: RechartsLineProps['dataKey']; // dataKey обязателен для Line
    color?: string; // Наш кастомный цвет
    // Здесь можно добавить другие часто используемые LineProps, если нужно
}

const HighlightAndZoomLineChart: React.FC<HighlightAndZoomLineChartProps> = ({
  data,
  lines,
  xAxisProps,
  yAxisProps,
  tooltipProps,
  legendProps, // Этот проп теперь не используется напрямую для <Legend .../>
  cartesianGridProps,
  brushProps, // Получаем brushProps
  showResponsiveContainer = true,
  ...rest 
}) => {
  const chartContent = (
    <RechartsLineChart data={data} {...rest}>
      {cartesianGridProps && <CartesianGrid {...cartesianGridProps} />}
      {xAxisProps && <XAxis {...xAxisProps} />}
      {yAxisProps && <YAxis {...yAxisProps} />}
      {tooltipProps && <RechartsTooltip {...tooltipProps} />}
      {/* Убираем {...legendProps} чтобы избежать ошибки типов, легенда будет использовать значения по умолчанию */} 
      {lines.length > 0 && <Legend />}
      {lines.map((lineConfig) => (
        <Line
          key={lineConfig.key} // React key
          type={lineConfig.type || "monotone"}
          dataKey={lineConfig.dataKey}
          name={lineConfig.name} // Для Legend
          stroke={lineConfig.color || lineConfig.stroke || "#8884d8"} 
          activeDot={lineConfig.activeDot || { r: 8 }}
          connectNulls={lineConfig.connectNulls}
          // Передаем другие известные пропсы LineProps, если они есть в lineConfig
          dot={lineConfig.dot}
          isAnimationActive={lineConfig.isAnimationActive}
          animationDuration={lineConfig.animationDuration}
          animationEasing={lineConfig.animationEasing}
          // Можно добавить еще, если нужно, но стараемся избегать полного spread, если он вызывает проблемы
        />
      ))}
      {/* Добавляем Brush если переданы brushProps */} 
      {brushProps && (
        <Brush 
          dataKey={brushProps.dataKey} 
          height={brushProps.height || 30} 
          stroke={brushProps.stroke || '#8884d8'}
          startIndex={brushProps.startIndex}
          endIndex={brushProps.endIndex}
          onChange={brushProps.onChange}
          fill={brushProps.fill}
          travellerWidth={brushProps.travellerWidth}
          gap={brushProps.gap}
          padding={brushProps.padding}
          x={brushProps.x}
          y={brushProps.y}
          alwaysShowText={brushProps.alwaysShowText}
        />
      )}
    </RechartsLineChart>
  );

  if (showResponsiveContainer) {
    return (
      <ResponsiveContainer width="100%" height={rest.height || 300}>
        {chartContent}
      </ResponsiveContainer>
    );
  }

  return chartContent;
};

export default HighlightAndZoomLineChart; 