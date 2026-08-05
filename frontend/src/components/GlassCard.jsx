import React from 'react';

export default function GlassCard({ children, className = '', onClick, style }) {
  return (
    <div 
      className={`glass-card ${className}`} 
      onClick={onClick} 
      style={style}
    >
      {children}
    </div>
  );
}
