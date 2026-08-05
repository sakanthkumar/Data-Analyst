import React from 'react';
import GlassCard from '../GlassCard';

export default function ReportSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
      {/* Header skeleton */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '60%' }}>
          <div className="skeleton-shimmer" style={{ height: '36px', width: '200px' }} />
          <div className="skeleton-shimmer" style={{ height: '18px', width: '400px' }} />
        </div>
        <div className="skeleton-shimmer" style={{ height: '38px', width: '120px', borderRadius: '6px' }} />
      </div>

      {/* Grid of Report Cards skeleton */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px' }}>
        {[1, 2, 3].map(i => (
          <GlassCard key={i} style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="skeleton-shimmer" style={{ width: '220px', height: '20px' }} />
              <div className="skeleton-shimmer" style={{ width: '80px', height: '32px', borderRadius: '6px' }} />
            </div>
            <div className="skeleton-shimmer" style={{ width: '150px', height: '12px' }} />
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div className="skeleton-shimmer" style={{ width: '100%', height: '14px' }} />
              <div className="skeleton-shimmer" style={{ width: '95%', height: '14px' }} />
              <div className="skeleton-shimmer" style={{ width: '80%', height: '14px' }} />
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
