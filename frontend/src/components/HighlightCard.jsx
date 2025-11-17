import React from 'react';
import { motion } from 'framer-motion';

const HighlightCard = ({ icon, title, children }) => {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      className="flex flex-col rounded-xl border border-slate-900/10 bg-card-light p-6 shadow-soft backdrop-blur-md transition-transform duration-200 dark:border-slate-50/10 dark:bg-card-dark"
    >
      <div className="flex size-12 items-center justify-center rounded-lg bg-accent/10 text-accent dark:bg-accent/20">
        <span className="material-symbols-outlined text-3xl">{icon}</span>
      </div>
      <h3 className="mt-4 text-lg font-bold">{title}</h3>
      <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">{children}</p>
    </motion.div>
  );
};

export default HighlightCard;
