import React from 'react';
import GlassCard from '../GlassCard';

export default function DatasetSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
      {/* Header skeleton */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '60%' }}>
          <div className="skeleton-shimmer" style={{ height: '36px', width: '220px' }} />
          <div className="skeleton-shimmer" style={{ height: '18px', width: '450px' }} />
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <div className="skeleton-shimmer" style={{ height: '38px', width: '100px', borderRadius: '6px' }} />
          <div className="skeleton-shimmer" style={{ height: '38px', width: '110px', borderRadius: '6px' }} />
        </div>
      </div>

      {/* Grid of dataset cards skeleton */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px' }}>
        {/* Upload card placeholder */}
        <div className="skeleton-shimmer" style={{ height: '270px', borderRadius: '12px' }} />
        
        {/* Dataset cards placeholders */}
        {[1, 2, 3].map(i => (
          <GlassCard key={i} style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="skeleton-shimmer" style={{ height: '128px', width: '100%' }} />
            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="skeleton-shimmer" style={{ width: '65%', height: '18px' }} />
              <div style={{ display: 'flex', gap: '8px' }}>
                <div className="skeleton-shimmer" style={{ width: '70px', height: '16px', borderRadius: '4px' }} />
                <div className="skeleton-shimmer" style={{ width: '60px', height: '16px', borderRadius: '4px' }} />
              </div>
              <div style={{ display: 'flex', gap: '24px', margin: '10px 0' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div className="skeleton-shimmer" style={{ width: '40px', height: '10px' }} />
                  <div className="skeleton-shimmer" style={{ width: '80px', height: '14px' }} />
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div className="skeleton-shimmer" style={{ width: '40px', height: '10px' }} />
                  <div className="skeleton-shimmer" style={{ width: '50px', height: '14px' }} />
                </div>
              </div>
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between' }}>
                <div className="skeleton-shimmer" style={{ width: '50px', height: '12px' }} />
                <div className="skeleton-shimmer" style={{ width: '60px', height: '12px' }} />
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
