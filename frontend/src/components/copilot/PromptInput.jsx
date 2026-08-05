import React, { useState } from 'react';

export default function PromptInput({ onSend, loading, onAttachDataset }) {
  const [value, setValue] = useState('');

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!value.trim() || loading) return;
    onSend(value.trim());
    setValue('');
  };

  return (
    <form onSubmit={handleSubmit} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          background: 'rgba(11, 28, 48, 0.4)',
          backdropFilter: 'blur(20px)',
          border: '1px solid var(--border-color)',
          borderRadius: '9999px',
          padding: '4px 6px 4px 12px',
          boxShadow: 'var(--shadow-sm)',
          transition: 'all 0.3s ease',
          width: '100%',
          boxSizing: 'border-box'
        }}
        onFocusCapture={(e) => {
          e.currentTarget.style.borderColor = 'var(--border-active)';
          e.currentTarget.style.boxShadow = '0 0 20px rgba(173, 198, 255, 0.1)';
        }}
        onBlurCapture={(e) => {
          e.currentTarget.style.borderColor = 'var(--border-color)';
          e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
        }}
      >
        {/* Left Add Attachment Trigger */}
        <button
          type="button"
          onClick={onAttachDataset}
          title="Summarize Dataset Health"
          disabled={loading}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            padding: '8px',
            borderRadius: '50%',
            transition: 'color 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--primary-color)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>
            add
          </span>
        </button>

        {/* Text Input */}
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask your analyst anything..."
          disabled={loading}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: 'var(--text-main)',
            fontSize: '0.925rem',
            padding: '8px 12px',
            fontFamily: 'Inter, sans-serif'
          }}
        />

        {/* Action icons & Send button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', paddingRight: '2px' }}>
          <button
            type="button"
            title="Voice input"
            disabled={true}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              opacity: 0.4,
              cursor: 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              padding: '8px',
              borderRadius: '50%'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>
              mic
            </span>
          </button>

          <button
            type="submit"
            disabled={loading || !value.trim()}
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '50%',
              backgroundColor: value.trim() ? 'var(--primary-color)' : 'rgba(255, 255, 255, 0.03)',
              color: value.trim() ? 'white' : 'var(--text-muted)',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: value.trim() && !loading ? 'pointer' : 'default',
              transition: 'all 0.2s',
              padding: 0,
              boxShadow: value.trim() ? '0 2px 8px rgba(59, 130, 246, 0.2)' : 'none'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '18px', fontVariationSettings: "'FILL' 1" }}>
              arrow_forward
            </span>
          </button>
        </div>
      </div>
    </form>
  );
}
