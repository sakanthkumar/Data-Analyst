import React from 'react';
import { motion } from 'framer-motion';

export default function StaggerContainer({ children, staggerDelay = 0.05, delayChildren = 0, className = '', style }) {
  const containerVariants = {
    initial: {},
    animate: {
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: delayChildren
      }
    }
  };

  return (
    <motion.div
      initial="initial"
      animate="animate"
      variants={containerVariants}
      className={className}
      style={style}
    >
      {children}
    </motion.div>
  );
}
