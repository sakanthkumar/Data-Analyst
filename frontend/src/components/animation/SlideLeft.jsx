import React from 'react';
import { motion } from 'framer-motion';
import { TRANSITIONS, VARIANTS } from '../../utils/motion';

export default function SlideLeft({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={VARIANTS.slideLeft}
      transition={{ ...TRANSITIONS.calm, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
