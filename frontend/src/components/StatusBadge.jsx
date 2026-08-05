import React from 'react';

export default function StatusBadge({ status, label }) {
  const normalizedStatus = status ? status.toUpperCase() : 'UNKNOWN';
  const displayLabel = label || normalizedStatus;

  let dotColor = '#9ca3af'; // default muted gray
  let badgeClass = 'bg-white/5 text-on-surface-variant';

  if (normalizedStatus === 'SYNCED' || normalizedStatus === 'ACTIVE' || normalizedStatus === 'STABLE' || normalizedStatus === 'SUCCESS' || normalizedStatus === 'COMPLETED' || normalizedStatus === 'ONLINE') {
    dotColor = '#10b981'; // emerald-500
    badgeClass = 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
  } else if (normalizedStatus === 'SYNCING' || normalizedStatus === 'PENDING' || normalizedStatus === 'RUNNING' || normalizedStatus === 'PROCESSING') {
    dotColor = '#f59e0b'; // amber-500
    badgeClass = 'bg-amber-500/10 text-amber-400 border border-amber-500/20';
  } else if (normalizedStatus === 'FAILED' || normalizedStatus === 'ERROR' || normalizedStatus === 'CRITICAL' || normalizedStatus === 'OFFLINE') {
    dotColor = '#ef4444'; // rose-500
    badgeClass = 'bg-rose-500/10 text-rose-400 border border-rose-500/20';
  }

  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-label-md tracking-wider font-semibold uppercase ${badgeClass}`} style={{ display: 'inline-flex', verticalAlign: 'middle', whiteSpace: 'nowrap' }}>
      <span 
        className="w-1.5 h-1.5 rounded-full" 
        style={{ 
          backgroundColor: dotColor, 
          boxShadow: `0 0 6px ${dotColor}`,
          display: 'inline-block'
        }} 
      />
      {displayLabel}
    </div>
  );
}
