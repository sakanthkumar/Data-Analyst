import React from 'react';
import Modal from './Modal';

export default function AcronymModal({
  isOpen,
  onClose,
  unknownAcronyms,
  acronymInputs,
  setAcronymInputs,
  onSubmit,
  onSkip
}) {
  return (
    <Modal isOpen={isOpen} onClose={onSkip} title="🧠 Define Custom Terms" maxWidth="600px">
      <div style={{ padding: '1rem 0' }}>
        <p>The system detected unrecognized terms in the dataset. Please define them for accurate analysis.</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '1rem' }}>
          {unknownAcronyms.map(acronym => (
            <div key={acronym} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <strong style={{ minWidth: '60px' }}>{acronym}:</strong>
              <input
                type="text"
                placeholder={`Definition of ${acronym}...`}
                value={acronymInputs[acronym] || ''}
                style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                onChange={(e) => setAcronymInputs(prev => ({ ...prev, [acronym]: e.target.value }))}
              />
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '1rem' }}>
        <button className="secondary-btn" onClick={onSkip}>Skip Definitions</button>
        <button className="primary-btn" onClick={onSubmit}>Save & Start Analysis</button>
      </div>
    </Modal>
  );
}
