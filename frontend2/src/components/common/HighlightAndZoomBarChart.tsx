import React from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type {
    BarProps as RechartsBarProps,
    XAxisProps,
    YAxisProps,
    TooltipProps,
    LegendProps as RechartsLegendProps,
    CartesianGridProps
} from 'recharts';

// Определяем тип для конфигурации одного "Bar" элемента
interface BarConfig extends Omit<RechartsBarProps, 'key'> {
    key: string | number; // React key
    name: string;         // Имя для легенды
    dataKey: RechartsBarProps['dataKey']; // dataKey обязателен
    fill?: string;       // Цвет заливки
    // Добавляем другие пропсы, которые могут понадобиться для Bar
    label?: RechartsBarProps['label'];
    onMouseEnter?: RechartsBarProps['onMouseEnter'];
    onMouseLeave?: RechartsBarProps['onMouseLeave'];
    onClick?: RechartsBarProps['onClick'];
    // и т.д.
}

export interface HighlightAndZoomBarChartProps extends Omit<React.ComponentProps<typeof RechartsBarChart>, 'children'> {
  bars: Array<BarConfig>; // Массив для описания столбцов (Bar)
  xAxisProps?: XAxisProps;
  yAxisProps?: YAxisProps;
  tooltipProps?: TooltipProps<any, any>;
  legendProps?: RechartsLegendProps; // Этот проп не будет использоваться для <Legend .../>
  cartesianGridProps?: CartesianGridProps;
  showResponsiveContainer?: boolean;
}

const HighlightAndZoomBarChart: React.FC<HighlightAndZoomBarChartProps> = ({
  data,
  bars,
  xAxisProps,
  yAxisProps,
  tooltipProps,
  legendProps, // Не используется для spread
  cartesianGridProps,
  showResponsiveContainer = true,
  ...rest
}) => {
  const chartContent = (
    <RechartsBarChart data={data} {...rest}>
      {cartesianGridProps && <CartesianGrid {...cartesianGridProps} />}
      {xAxisProps && <XAxis {...xAxisProps} />}
      {yAxisProps && <YAxis {...yAxisProps} />}
      {tooltipProps && <RechartsTooltip {...tooltipProps} />}
      {/* Убираем {...legendProps}, легенда будет по умолчанию, если bars существуют */}
      {bars.length > 0 && <Legend />}
      {bars.map((barConfig) => {
        // Явно передаем известные пропсы, чтобы избежать проблем с типами при spread
        return (
          <Bar
            key={barConfig.key}
            dataKey={barConfig.dataKey}
            name={barConfig.name}
            fill={barConfig.fill || "#82ca9d"} // Цвет по умолчанию, если не указан
            label={barConfig.label}
            onMouseEnter={barConfig.onMouseEnter}
            onMouseLeave={barConfig.onMouseLeave}
            onClick={barConfig.onClick}
            // Другие специфичные BarProps можно добавить сюда, если они есть в BarConfig
            // background={barConfig.background} // пример
            // radius={barConfig.radius} // пример
          />
        );
      })}
    </RechartsBarChart>
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

export default HighlightAndZoomBarChart; 