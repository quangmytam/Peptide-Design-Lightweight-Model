import React from 'react';
import { motion } from 'framer-motion';
import Button from './Button';

const PeptideResultCard = ({ peptide, index }) => {
  return (
    <motion.div
      key={index}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.07)' }}
      className="flex flex-col gap-5 p-5 rounded-xl bg-card-light dark:bg-card-dark backdrop-blur-xl border border-white/50 dark:border-white/10 shadow-lg shadow-gray-500/5 dark:shadow-black/20"
    >
      <div className="flex flex-col gap-2">
        <p className="text-xs font-medium uppercase tracking-wider text-subtext-light dark:text-subtext-dark">Sequence</p>
        <p className="font-mono text-lg font-medium tracking-wider">{peptide.sequence}</p>
      </div>
      <div className="flex flex-col gap-3">
        <div className="flex justify-between items-baseline">
          <p className="text-sm font-medium text-subtext-light dark:text-subtext-dark">Stability Score</p>
          <p className="text-2xl font-bold text-accent-teal">{peptide.stability}</p>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div className="bg-accent-teal h-2 rounded-full" style={{ width: `${peptide.stability * 100}%` }}></div>
        </div>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <Button variant="secondary" className="flex-1">
          <span className="material-symbols-outlined text-base">content_copy</span>
          Copy
        </Button>
        <Button variant="secondary" className="flex-1 bg-gray-500/10 text-text-light dark:text-text-dark hover:bg-gray-500/20">
          View Details
        </Button>
      </div>
    </motion.div>
  );
};

export default PeptideResultCard;
