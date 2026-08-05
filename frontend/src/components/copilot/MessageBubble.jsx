import React from 'react';

export default function MessageBubble({ message }) {
  return (
    <div 
      className="fade-in" 
      style={{ 
        display: 'flex', 
        justifyContent: 'flex-end', 
        gap: '12px',
        margin: '12px 0 24px 0',
        alignItems: 'flex-start',
        boxSizing: 'border-box',
        width: '100%'
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', maxWidth: '75%' }}>
        <span 
          style={{ 
            fontSize: '0.72rem', 
            color: 'var(--text-muted)', 
            textTransform: 'uppercase', 
            letterSpacing: '0.06em',
            marginBottom: '4px',
            fontWeight: 600,
            fontFamily: 'JetBrains Mono, monospace'
          }}
        >
          You
        </span>
        <div 
          style={{ 
            background: 'var(--bg-sidebar)',
            border: '1px solid var(--border-color)',
            color: 'var(--text-main)',
            padding: '12px 16px',
            borderRadius: '12px 12px 2px 12px',
            fontSize: '0.925rem',
            lineHeight: '1.5',
            boxShadow: 'var(--shadow-sm)',
            wordBreak: 'break-word',
            whiteSpace: 'pre-wrap'
          }}
        >
          {message.content}
        </div>
      </div>
      
      <div 
        style={{ 
          width: '32px', 
          height: '32px', 
          borderRadius: '50%', 
          backgroundColor: 'var(--bg-card)', 
          border: '1px solid var(--border-color)',
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: 'var(--text-muted)',
          marginTop: '18px'
        }}
      >
        <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
          person
        </span>
      </div>
    </div>
  );
}
