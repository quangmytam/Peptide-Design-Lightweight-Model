import React from 'react';
import { motion } from 'framer-motion';

const DatasetListItem = ({ dataset, selectedDataset, setSelectedDataset }) => {
  const getStatusIndicator = (status) => {
    switch (status) {
      case 'Analyzed':
        return (
          <div className="shrink-0 flex items-center gap-2 text-sm font-medium text-green-700 dark:text-green-400">
            <div className="size-2.5 rounded-full bg-green-500"></div>
            <span>Analyzed</span>
          </div>
        );
      case 'Processing':
        return (
          <div className="shrink-0 flex items-center gap-2 text-sm font-medium text-amber-600 dark:text-amber-400">
            <div className="size-2.5 rounded-full bg-amber-500"></div>
            <span>Processing</span>
          </div>
        );
      default:
        return (
          <div className="shrink-0 flex items-center gap-2 text-sm font-medium text-slate-500 dark:text-slate-400">
            <div className="size-2.5 rounded-full bg-slate-400"></div>
            <span>Raw</span>
          </div>
        );
    }
  };

  return (
    <motion.div
      onClick={() => setSelectedDataset(dataset)}
      whileHover={{ backgroundColor: 'rgba(203, 213, 225, 0.5)' }}
      className={`flex items-center gap-4 px-4 min-h-[72px] py-2 justify-between rounded-lg cursor-pointer transition-colors ${selectedDataset.name === dataset.name ? 'bg-primary/20 dark:bg-primary/30 border border-primary' : 'hover:bg-slate-200/50 dark:hover:bg-slate-800/50'}`}
    >
      <div className="flex items-center gap-4">
        <div className="text-primary dark:text-white flex items-center justify-center rounded-lg bg-white/50 dark:bg-slate-700/50 shrink-0 size-12">
          <span className="material-symbols-outlined text-2xl">
            {dataset.status === 'Processing' ? 'hourglass_top' : 'biotech'}
          </span>
        </div>
        <div className="flex flex-col justify-center">
          <p className="text-[#0d141b] dark:text-white text-base font-medium leading-normal line-clamp-1">{dataset.name}</p>
          <p className="text-slate-600 dark:text-slate-300 text-sm font-normal leading-normal line-clamp-2">Uploaded: {dataset.date}</p>
        </div>
      </div>
      {getStatusIndicator(dataset.status)}
    </motion.div>
  );
};

export default DatasetListItem;
