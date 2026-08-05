import React from 'react';

export default function SuggestedPrompts({ prompts = [], onSelectPrompt }) {
  if (!prompts || prompts.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
      <div 
        style={{ 
          fontSize: '0.72rem', 
          color: 'var(--text-muted)', 
          fontWeight: 600, 
          textTransform: 'uppercase', 
          letterSpacing: '0.06em',
          fontFamily: 'JetBrains Mono, monospace'
        }}
      >
        Suggested Actions & Follow-ups
      </div>
      <div 
        style={{ 
          display: 'flex', 
          gap: '8px', 
          flexWrap: 'wrap', 
          paddingBottom: '4px'
        }}
      >
        {prompts.map((promptText, idx) => (
          <button
            key={idx}
            onClick={() => onSelectPrompt(promptText)}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '16px',
              padding: '6px 14px',
              color: 'var(--text-main)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              fontFamily: 'Inter, sans-serif',
              textAlign: 'left'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.08)';
              e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
              e.currentTarget.style.borderColor = 'var(--border-color)';
            }}
          >
            {promptText}
          </button>
        ))}
      </div>
    </div>
  );
}
