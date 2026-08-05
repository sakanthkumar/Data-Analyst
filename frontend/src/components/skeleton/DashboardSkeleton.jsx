import React from 'react';
import GlassCard from '../GlassCard';

export default function DashboardSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
      {/* Welcome Header Skeleton */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '60%' }}>
          <div className="skeleton-shimmer" style={{ height: '36px', width: '250px' }} />
          <div className="skeleton-shimmer" style={{ height: '18px', width: '400px' }} />
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <div className="skeleton-shimmer" style={{ height: '38px', width: '140px', borderRadius: '6px' }} />
          <div className="skeleton-shimmer" style={{ height: '38px', width: '120px', borderRadius: '6px' }} />
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
        {[1, 2, 3].map(i => (
          <GlassCard key={i} style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div className="skeleton-shimmer" style={{ width: '40px', height: '40px', borderRadius: '8px' }} />
              <div className="skeleton-shimmer" style={{ width: '50px', height: '16px' }} />
            </div>
            <div className="skeleton-shimmer" style={{ width: '120px', height: '14px' }} />
            <div className="skeleton-shimmer" style={{ width: '80px', height: '32px' }} />
            <div className="skeleton-shimmer" style={{ width: '100%', height: '4px', marginTop: '4px' }} />
          </GlassCard>
        ))}
      </div>

      {/* Layout panels */}
      <div style={{ display: 'grid', gridTemplateColumns: '8fr 4fr', gap: '24px' }}>
        {/* Left Side: Summary and Health */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Executive Summary skeleton */}
          <GlassCard style={{ padding: '32px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="skeleton-shimmer" style={{ width: '240px', height: '24px' }} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div className="skeleton-shimmer" style={{ width: '100%', height: '18px' }} />
                <div className="skeleton-shimmer" style={{ width: '100%', height: '16px' }} />
                <div className="skeleton-shimmer" style={{ width: '90%', height: '16px' }} />
                <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                  <div className="skeleton-shimmer" style={{ width: '100px', height: '36px', borderRadius: '6px' }} />
                  <div className="skeleton-shimmer" style={{ width: '80px', height: '36px', borderRadius: '6px' }} />
                </div>
              </div>
              <div className="skeleton-shimmer" style={{ height: '140px', borderRadius: '12px' }} />
            </div>
          </GlassCard>

          {/* Health Ecosystem skeleton */}
          <GlassCard style={{ padding: '32px', height: '300px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="skeleton-shimmer" style={{ width: '200px', height: '24px' }} />
            <div className="skeleton-shimmer" style={{ width: '300px', height: '14px' }} />
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
              <div className="skeleton-shimmer" style={{ width: '180px', height: '180px', borderRadius: '50%' }} />
            </div>
          </GlassCard>
        </div>

        {/* Right Side: Timeline skeleton */}
        <GlassCard style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="skeleton-shimmer" style={{ width: '180px', height: '24px' }} />
          <div className="skeleton-shimmer" style={{ width: '120px', height: '14px', marginBottom: '8px' }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {[1, 2, 3, 4].map(i => (
              <div key={i} style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                <div className="skeleton-shimmer" style={{ width: '40px', height: '40px', borderRadius: '50%' }} />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
                  <div className="skeleton-shimmer" style={{ width: '70%', height: '14px' }} />
                  <div className="skeleton-shimmer" style={{ width: '40%', height: '10px' }} />
                </div>
                <div className="skeleton-shimmer" style={{ width: '8px', height: '8px', borderRadius: '50%' }} />
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
