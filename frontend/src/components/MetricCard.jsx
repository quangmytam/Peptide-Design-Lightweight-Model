import React from 'react';
import Card from './Card';

const MetricCard = ({ title, value, delta }) => {
  const isPositive = delta.startsWith('+');
  const deltaColor = isPositive ? 'text-green-600 dark:text-green-500' : 'text-red-600 dark:text-red-500';

  return (
    <Card className="flex-col gap-2">
      <p className="text-slate-700 dark:text-slate-300 text-base font-medium leading-normal">{title}</p>
      <p className="text-slate-900 dark:text-slate-50 tracking-tight text-3xl font-bold leading-tight">{value}</p>
      <p className={`${deltaColor} text-sm font-medium leading-normal`}>{delta}</p>
    </Card>
  );
};

export default MetricCard;
