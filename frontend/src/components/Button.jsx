import React from 'react';
import { motion } from 'framer-motion';

const Button = ({ children, onClick, disabled, variant = 'primary', className = '' }) => {
  const baseClasses = 'flex items-center justify-center gap-2 rounded-lg h-12 text-base font-bold tracking-wide transition-colors';

  const variants = {
    primary: 'bg-primary text-white hover:bg-primary/90 flex-1',
    icon: 'w-12 bg-slate-200 dark:bg-slate-700 text-text-light dark:text-text-dark hover:bg-slate-300 dark:hover:bg-slate-600',
    secondary: 'h-10 px-4 bg-primary/20 dark:bg-primary/30 text-primary text-sm font-bold hover:bg-primary/30 dark:hover:bg-primary/40',
    destructive: 'h-10 px-4 bg-red-500/20 text-red-500 text-sm font-bold hover:bg-red-500/30',
    small: 'h-10 px-2 bg-primary text-white text-sm font-medium hover:bg-primary/90',
  };

  const combinedClasses = `${baseClasses} ${variants[variant]} ${className}`;

  return (
    <motion.button
      whileHover={{ scale: variant === 'icon' ? 1.1 : 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      disabled={disabled}
      className={combinedClasses}
    >
      {children}
    </motion.button>
  );
};

export default Button;
