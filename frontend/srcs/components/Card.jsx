import React from 'react';
import { motion } from 'framer-motion';

const Card = ({ children, className }) => {
  return (
    <motion.div
      whileHover={{ y: -4, boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.07)' }}
      className={`glass-card rounded-xl shadow-card p-6 ${className}`}
    >
      {children}
    </motion.div>
  );
};

export default Card;
