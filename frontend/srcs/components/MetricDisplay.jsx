import React from 'react';

const MetricDisplay = ({ label, value }) => {
  return (
    <div className="bg-background-light dark:bg-background-dark p-4 rounded-lg">
      <p className="text-xs uppercase text-text-light/60 dark:text-text-dark/60 font-semibold tracking-wider">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value}</p>
    </div>
  );
};

export default MetricDisplay;
