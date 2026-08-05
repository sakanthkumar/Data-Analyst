import React from 'react';

export default function Drawer({ isOpen, onClose, title, children, footer }) {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop overlay */}
      <div 
        className="fixed inset-0 z-[70]"
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(4px)',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0
        }}
        onClick={onClose}
      />
      {/* Drawer panel */}
      <aside 
        className="fixed right-0 top-0 h-screen shadow-2xl z-[80] flex flex-col"
        style={{
          width: '400px',
          backgroundColor: 'var(--bg-sidebar)',
          borderLeft: '1px solid var(--border-color)',
          color: 'var(--text-main)',
          position: 'fixed',
          right: 0,
          top: 0,
          bottom: 0,
          overflow: 'hidden'
        }}
      >
        {/* Header */}
        <div 
          className="px-6 h-16 flex items-center justify-between"
          style={{ 
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'between'
          }}
        >
          <h2 style={{ fontSize: '1.125rem', fontWeight: 'bold', margin: 0, color: 'var(--text-main)' }}>
            {title || 'Details'}
          </h2>
          <button 
            className="p-1 rounded-lg transition-colors border-none bg-transparent cursor-pointer flex items-center justify-center text-muted hover:text-main"
            style={{ 
              color: 'var(--text-muted)',
              outline: 'none'
            }}
            onClick={onClose}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>close</span>
          </button>
        </div>
        
        {/* Content */}
        <div 
          className="flex-1 overflow-y-auto p-6"
          style={{ overflowY: 'auto' }}
        >
          {children}
        </div>

        {/* Footer */}
        {footer && (
          <div 
            className="p-6"
            style={{ 
              borderTop: '1px solid var(--border-color)',
              backgroundColor: 'rgba(0, 0, 0, 0.05)'
            }}
          >
            {footer}
          </div>
        )}
      </aside>
    </>
  );
}
