import React from 'react';
import StaggerContainer from './StaggerContainer';

export default function StaggerGrid({ children, className = '', staggerDelay = 0.04, delayChildren = 0, style }) {
  return (
    <StaggerContainer
      staggerDelay={staggerDelay}
      delayChildren={delayChildren}
      className={`grid ${className}`}
      style={style}
    >
      {children}
    </StaggerContainer>
  );
}
