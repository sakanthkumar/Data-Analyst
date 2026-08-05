import React from 'react';

function formatBytes(bytes, decimals = 2) {
  if (!bytes) return 'N/A';
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export default function DatasetContextPanel({ activeDataset, domainProfile }) {
  if (!activeDataset) return null;

  const rows = activeDataset.shape?.[0] || 0;
  const cols = activeDataset.shape?.[1] || 0;
  const fileSize = formatBytes(activeDataset.file_size_bytes);
  const targetCol = activeDataset.target_column || domainProfile?.target_column || 'Not Defined';
  
  // Quality analysis
  const missingCount = Object.values(activeDataset.missing_values || {}).reduce((a, b) => a + b, 0);
  const duplicateCount = activeDataset.duplicate_rows || 0;
  const outlierCount = Object.values(activeDataset.outliers || {}).reduce((a, b) => a + b, 0);
  const totalCells = rows * cols;
  const qualityScore = totalCells > 0
    ? Math.max(0, Math.min(100, Math.round((1 - (missingCount + duplicateCount) / totalCells) * 100)))
    : 100;

  const getQualityColor = (score) => {
    if (score >= 90) return 'var(--accent-color)'; // success emerald
    if (score >= 70) return 'var(--warning-color)'; // warning amber
    return 'var(--danger-color)'; // danger rose
  };

  const status = domainProfile?.status || 'idle';
  const statusLabels = {
    idle: 'Idle',
    running: 'Analyzing',
    completed: 'Active',
    failed: 'Error'
  };
  const statusColors = {
    idle: '#8c909f',
    running: '#3b82f6',
    completed: '#10b981',
    failed: '#f43f5e'
  };

  return (
    <aside 
      className="glass-card" 
      style={{ 
        width: '300px', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '20px', 
        padding: '20px',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-color)',
        borderLeft: 'none',
        borderRadius: 0,
        height: '100%',
        boxSizing: 'border-box',
        overflowY: 'auto'
      }}
    >
      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '12px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Dataset Context
        </h3>
        
        {/* Dataset Basic info block */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Name</span>
            <span 
              style={{ 
                fontSize: '0.85rem', 
                color: 'var(--text-main)', 
                fontWeight: 600,
                maxWidth: '180px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
              title={activeDataset.filename}
            >
              {activeDataset.filename}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>File Size</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 500 }}>
              {fileSize}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Rows × Cols</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 500, fontFamily: 'JetBrains Mono, monospace' }}>
              {rows.toLocaleString()} × {cols}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Type</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 500 }}>
              {activeDataset.target_column ? 'Labeled Segment' : 'Tabular CSV'}
            </span>
          </div>
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: 0 }} />

      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '12px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Domain Profile
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Domain Context</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 600 }}>
              {domainProfile?.domain || 'Identifying...'}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Target Column</span>
            <span 
              className="label-technical"
              style={{ 
                color: 'var(--primary-color)', 
                background: 'rgba(59, 130, 246, 0.08)', 
                padding: '2px 6px', 
                borderRadius: '4px',
                border: '1px solid rgba(59, 130, 246, 0.15)',
                fontWeight: 600,
                maxWidth: '160px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
              title={targetCol}
            >
              {targetCol}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Analysis Class</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', textTransform: 'capitalize' }}>
              {domainProfile?.analysis_type || 'Scanning...'}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Analysis Status</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: statusColors[status], fontSize: '10px' }}>●</span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 600 }}>
                {statusLabels[status]}
              </span>
            </div>
          </div>
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: 0 }} />

      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '12px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Data Health & Quality
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {/* Quality Progress bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Quality Index</span>
              <span style={{ fontSize: '0.85rem', color: getQualityColor(qualityScore), fontWeight: 700 }}>
                {qualityScore}%
              </span>
            </div>
            <div style={{ height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
              <div 
                style={{ 
                  height: '100%', 
                  width: `${qualityScore}%`, 
                  backgroundColor: getQualityColor(qualityScore),
                  borderRadius: '3px',
                  transition: 'width 0.5s ease-out'
                }} 
              />
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Missing Values</span>
            <span style={{ fontSize: '0.85rem', color: missingCount > 0 ? 'var(--warning-color)' : 'var(--text-main)', fontWeight: 500 }}>
              {missingCount.toLocaleString()}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Duplicate Rows</span>
            <span style={{ fontSize: '0.85rem', color: duplicateCount > 0 ? 'var(--warning-color)' : 'var(--text-main)', fontWeight: 500 }}>
              {duplicateCount.toLocaleString()}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Outliers Detected</span>
            <span style={{ fontSize: '0.85rem', color: outlierCount > 0 ? 'var(--warning-color)' : 'var(--text-main)', fontWeight: 500 }}>
              {outlierCount.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
