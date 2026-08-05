import React from 'react';

export default function EmptyCopilotState({ onTabChange }) {
  return (
    <div 
      className="glass-card fade-in" 
      style={{ 
        maxWidth: '560px', 
        margin: '6rem auto', 
        padding: '3rem 2rem', 
        textAlign: 'center',
        background: 'var(--bg-card)',
        backdropFilter: 'blur(20px)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px'
      }}
    >
      <div 
        style={{ 
          width: '64px', 
          height: '64px', 
          borderRadius: '50%', 
          backgroundColor: 'rgba(59, 130, 246, 0.08)', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          margin: '0 auto 1.5rem auto'
        }}
      >
        <span 
          className="material-symbols-outlined" 
          style={{ 
            fontSize: '36px', 
            color: 'var(--primary-color)' 
          }}
        >
          database_off
        </span>
      </div>

      <h2 
        style={{ 
          fontFamily: 'Geist', 
          fontSize: '1.75rem', 
          color: 'var(--text-main)', 
          marginBottom: '0.75rem' 
        }}
      >
        No Active Dataset
      </h2>
      
      <p 
        style={{ 
          color: 'var(--text-muted)', 
          fontSize: '0.95rem', 
          lineHeight: '1.6', 
          marginBottom: '2rem' 
        }}
      >
        Upload a dataset or choose one from your library to start interacting with the Dataset Intelligence Assistant.
      </p>

      <div 
        style={{ 
          backgroundColor: 'var(--bg-input)', 
          borderRadius: '8px', 
          padding: '1rem', 
          marginBottom: '2rem', 
          border: '1px solid var(--border-color)', 
          textAlign: 'left' 
        }}
      >
        <div 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.05em', 
            color: 'var(--text-muted)', 
            marginBottom: '0.5rem', 
            fontWeight: 600 
          }}
        >
          Supported Formats
        </div>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {['CSV', 'Excel', 'JSON', 'Parquet'].map((fmt) => (
            <div 
              key={fmt} 
              style={{ 
                fontSize: '0.8rem', 
                color: 'var(--text-main)', 
                background: 'rgba(255, 255, 255, 0.04)', 
                padding: '4px 10px', 
                borderRadius: '4px', 
                border: '1px solid var(--border-color)',
                fontFamily: 'JetBrains Mono, monospace'
              }}
            >
              {fmt}
            </div>
          ))}
        </div>
      </div>

      <button 
        className="primary-btn" 
        onClick={() => onTabChange('logs')}
        style={{ 
          padding: '12px 24px', 
          fontSize: '0.95rem', 
          fontWeight: 600, 
          borderRadius: '8px', 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px', 
          margin: '0 auto',
          boxShadow: '0 4px 12px rgba(59, 130, 246, 0.25)' 
        }}
      >
        <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
          folder_open
        </span>
        Open Dataset Library
      </button>
    </div>
  );
}
