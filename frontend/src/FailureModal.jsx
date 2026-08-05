import React from 'react';
import Modal from './Modal';

export default function FailureModal({
  isOpen,
  onClose,
  failures
}) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`⚠️ Highlighted Records (${failures.length})`} titleColor="var(--danger-color)" maxWidth="900px">
      <div className="table-wrapper" style={{ maxHeight: '400px', overflowY: 'auto' }}>
        <table>
          <thead>
            <tr>
              {failures.length > 0 && Object.keys(failures[0]).map(k => <th key={k}>{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {failures.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((v, j) => <td key={j}>{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ textAlign: 'right', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
        Report automatically saved to history.
      </p>
    </Modal>
  );
}
