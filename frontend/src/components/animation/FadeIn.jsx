import React from 'react';
import { motion } from 'framer-motion';
import { TRANSITIONS, VARIANTS } from '../../utils/motion';

export default function FadeIn({ children, delay = 0, duration, className = '' }) {
  const transition = {
    ...TRANSITIONS.calm,
    delay,
    ...(duration !== undefined && { duration })
  };

  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={VARIANTS.fadeIn}
      transition={transition}
      className={className}
    >
      {children}
    </motion.div>
  );
}
