import React from 'react';
import Modal from './Modal';

export default function TargetModal({
  isOpen,
  onClose,
  detectedTarget,
  confidence,
  candidateTargets,
  selectedTarget,
  setSelectedTarget,
  onConfirm
}) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="🎯 Confirm Target Column" maxWidth="600px">
      <div style={{ padding: '1.5rem 0' }}>
        <p style={{ marginBottom: '1.25rem' }}>
          Analyst.AI has scanned your dataset and detected the most likely target variable for failure/churn analysis:
        </p>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', background: 'var(--bg-secondary)', padding: '15px', borderRadius: '6px', marginBottom: '1.5rem', border: '1px solid var(--border-color)' }}>
          <div>
            <strong>Detected Target Variable:</strong> <span className="badge" style={{ fontSize: '0.95rem' }}>{detectedTarget}</span>
          </div>
          <div>
            <strong>Guessed Confidence Score:</strong> <span style={{ fontWeight: 'bold', color: confidence >= 0.9 ? 'var(--success-color)' : 'var(--accent-color)' }}>{Math.round(confidence * 100)}%</span>
          </div>
        </div>

        <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
          Select Target Column:
        </label>
        <select
          value={selectedTarget}
          onChange={(e) => setSelectedTarget(e.target.value)}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid var(--border-color)',
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)'
          }}
        >
          {candidateTargets.map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '1rem' }}>
        <button className="secondary-btn" onClick={onClose}>Cancel</button>
        <button className="primary-btn" onClick={onConfirm}>Confirm and Proceed</button>
      </div>
    </Modal>
  );
}
