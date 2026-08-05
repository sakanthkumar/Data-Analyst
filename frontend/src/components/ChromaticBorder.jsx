import React from 'react';

export default function ChromaticBorder({ children, className = '' }) {
  return (
    <div className={`chromatic-border ${className}`}>
      <div className="chromatic-inner">
        {children}
      </div>
    </div>
  );
}
