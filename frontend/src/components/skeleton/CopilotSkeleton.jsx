import React from 'react';
import GlassCard from '../GlassCard';

export default function CopilotSkeleton() {
  return (
    <div style={{ display: 'flex', height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Suggestions Sidebar Skeleton (Hidden on Mobile) */}
      <div 
        style={{ 
          width: '320px', 
          borderRight: '1px solid var(--border-color)', 
          padding: '24px', 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '24px',
          boxSizing: 'border-box'
        }}
      >
        <div className="skeleton-shimmer" style={{ width: '180px', height: '16px' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {[1, 2, 3].map(i => (
            <GlassCard key={i} style={{ padding: '16px', height: '60px' }}>
              <div className="skeleton-shimmer" style={{ width: '100%', height: '12px' }} />
              <div className="skeleton-shimmer" style={{ width: '80%', height: '12px', marginTop: '6px' }} />
            </GlassCard>
          ))}
        </div>
        <div style={{ marginTop: 'auto' }}>
          <GlassCard style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div className="skeleton-shimmer" style={{ width: '100px', height: '18px' }} />
            <div className="skeleton-shimmer" style={{ width: '100%', height: '12px' }} />
            <div className="skeleton-shimmer" style={{ width: '100%', height: '32px', borderRadius: '6px', marginTop: '4px' }} />
          </GlassCard>
        </div>
      </div>

      {/* Main Chat Feed Skeleton */}
      <div style={{ flex: 1, padding: '32px', display: 'flex', flexDirection: 'column', gap: '32px', boxSizing: 'border-box', position: 'relative' }}>
        <div style={{ maxWidth: '768px', margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: '32px' }}>
          
          {/* Welcome robot logo */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', padding: '32px 0' }}>
            <div className="skeleton-shimmer" style={{ width: '64px', height: '64px', borderRadius: '50%' }} />
            <div className="skeleton-shimmer" style={{ width: '280px', height: '24px' }} />
            <div className="skeleton-shimmer" style={{ width: '400px', height: '16px' }} />
          </div>

          {/* User Message Skeleton */}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <GlassCard style={{ padding: '16px', width: '60%', borderRadius: '16px' }}>
              <div className="skeleton-shimmer" style={{ width: '100%', height: '14px' }} />
              <div className="skeleton-shimmer" style={{ width: '70%', height: '14px', marginTop: '6px' }} />
            </GlassCard>
          </div>

          {/* AI Response Skeleton */}
          <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
            <div className="skeleton-shimmer" style={{ width: '32px', height: '32px', borderRadius: '50%', flexShrink: 0 }} />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="skeleton-shimmer" style={{ width: '90%', height: '14px' }} />
              
              {/* Embedded Chart Block Skeleton */}
              <GlassCard style={{ padding: '24px', height: '180px', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around' }}>
                <div className="skeleton-shimmer" style={{ width: '40px', height: '80px', borderRadius: '4px' }} />
                <div className="skeleton-shimmer" style={{ width: '40px', height: '110px', borderRadius: '4px' }} />
                <div className="skeleton-shimmer" style={{ width: '40px', height: '50px', borderRadius: '4px' }} />
                <div className="skeleton-shimmer" style={{ width: '40px', height: '130px', borderRadius: '4px' }} />
              </GlassCard>
              
              <div className="skeleton-shimmer" style={{ width: '70%', height: '14px' }} />
            </div>
          </div>
        </div>

        {/* Input Bar Placeholder */}
        <div style={{ position: 'absolute', bottom: '32px', left: '32px', right: '32px', display: 'flex', justifyContent: 'center' }}>
          <div className="skeleton-shimmer" style={{ width: '100%', maxWidth: '672px', height: '52px', borderRadius: '26px' }} />
        </div>
      </div>
    </div>
  );
}
