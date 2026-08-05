import React from 'react';

export default function Modal({ isOpen, onClose, title, titleColor = 'var(--primary-color)', maxWidth = '600px', children }) {
  if (!isOpen) return null;
  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{ maxWidth }}>
        <header className="modal-header">
          <h2 style={{ color: titleColor }}>{title}</h2>
          <button className="close-btn" onClick={onClose}>Close</button>
        </header>
        {children}
      </div>
    </div>
  );
}
