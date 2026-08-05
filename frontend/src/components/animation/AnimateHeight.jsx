import React from 'react';
import { motion } from 'framer-motion';
import { TRANSITIONS } from '../../utils/motion';

export default function AnimateHeight({ children, isOpen, className = '', style }) {
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ 
        height: isOpen ? 'auto' : 0, 
        opacity: isOpen ? 1 : 0 
      }}
      transition={TRANSITIONS.calm}
      className={className}
      style={{ overflow: 'hidden', ...style }}
    >
      {children}
    </motion.div>
  );
}
