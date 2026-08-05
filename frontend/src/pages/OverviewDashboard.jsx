import React from 'react';
import GlassCard from '../components/GlassCard';
import ChromaticBorder from '../components/ChromaticBorder';
import StatusBadge from '../components/StatusBadge';
import EnterpriseGlobe3D from '../three/EnterpriseGlobe3D';
import FadeIn from '../components/animation/FadeIn';
import StaggerContainer from '../components/animation/StaggerContainer';
import SlideUp from '../components/animation/SlideUp';
import MagneticButton from '../components/animation/MagneticButton';

// Simulated latency data
const latencyPoints = [25, 40, 35, 60, 55, 85, 95, 75, 60, 50, 45, 65, 80, 90, 75, 85];

// Markdown/Plaintext formatter
const ReportView = ({ text }) => {
  if (!text) return null;
  return (
    <div className="report-content" style={{ fontSize: '0.9rem', lineHeight: '1.6', color: 'var(--text-main)' }}>
      {text.split('\n').map((line, i) => {
        if (line.startsWith('###')) {
          return <h4 key={i} style={{ color: 'var(--primary-color)', fontSize: '1rem', marginTop: '14px', marginBottom: '6px', fontFamily: 'Geist' }}>{line.replace('###', '')}</h4>;
        }
        if (line.startsWith('**')) {
          return <strong key={i} style={{ display: 'block', marginTop: '10px', color: 'var(--text-main)' }}>{line.replace(/\*\*/g, '')}</strong>;
        }
        return <p key={i} style={{ marginBottom: '8px', color: 'var(--text-muted)' }}>{line}</p>;
      })}
    </div>
  );
};

export default function OverviewDashboard({
  data,
  domainProfile,
  runAnalysis,
  loadFailures,
  reports,
  reportLoading,
  downloadPDF,
  onNavigateToUpload
}) {
  const hasData = !!data;
  
  // Custom dashboard values reflecting uploaded dataset OR standard Stitch defaults
  const activeDatasets = hasData ? "1" : "1,284";
  const datasetsSubtext = hasData ? (data.filename || "active file") : "connected";
  const confidenceScore = hasData ? `${data.failure_rate}%` : "98.4%";
  const confidenceLabel = hasData ? "Calculated anomaly rate" : "Stable";
  const insightsCount = hasData && data.missing_values
    ? Object.values(data.missing_values).reduce((a, b) => a + b, 0).toLocaleString()
    : "42,910";
  const insightsLabel = hasData ? "missing data values detected" : "Processing 484 nodes / sec";

  return (
    <FadeIn>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* Welcome Header & Action Buttons Row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)', margin: '0 0 6px 0', fontFamily: 'Geist', letterSpacing: '-0.02em' }}>
              Enterprise Overview
            </h1>
            <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              Real-time health indicators and intelligence synthesis across your organization's data infrastructure.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              className="secondary-btn" 
              style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '6px' }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>calendar_month</span>
              Last 24 Hours
            </button>
            <MagneticButton 
              className="primary-btn" 
              onClick={downloadPDF}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px', borderRadius: '6px', color: 'var(--text-on-primary, #fff)', backgroundColor: 'var(--primary-color)' }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
              Export PDF
            </MagneticButton>
          </div>
        </div>

        {/* Banner if no active dataset loaded */}
        {!hasData && (
          <div 
            style={{ 
              background: 'rgba(59, 130, 246, 0.08)',
              border: '1px solid rgba(59, 130, 246, 0.2)',
              borderRadius: '8px',
              padding: '16px 20px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px'
            }}
          >
            <div>
              <strong style={{ color: 'var(--primary-color)', fontSize: '0.9rem', display: 'block' }}>⚡ Connect Your Own Dataset</strong>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                You are currently viewing mock system telemetry. Upload a CSV file to evaluate custom data fields.
              </span>
            </div>
            <button 
              className="primary-btn" 
              onClick={onNavigateToUpload}
              style={{ fontSize: '0.8rem', padding: '6px 14px', borderRadius: '6px' }}
            >
              Start New Analysis
            </button>
          </div>
        )}

        {/* KPI Layout Grid (Layout density matching Stitch) */}
        <StaggerContainer 
          staggerDelay={0.05} 
          className="cards-grid" 
          style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}
        >
          
          {/* KPI 1 - Active Datasets */}
          <SlideUp>
            <GlassCard style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div style={{ padding: '6px', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '6px', color: 'var(--primary-color)', display: 'flex' }}>
                  <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>database</span>
                </div>
                <div style={{ display: 'flex', color: 'var(--accent-color)', fontSize: '0.75rem', fontWeight: 600, alignItems: 'center', gap: '2px' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>trending_up</span>
                  +12%
                </div>
              </div>
              <div className="label-technical" style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Active Datasets
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginTop: '4px' }}>
                <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>
                  {activeDatasets}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {datasetsSubtext}
                </span>
              </div>
              <div style={{ height: '3px', backgroundColor: 'var(--bg-input)', borderRadius: '1.5px', marginTop: '16px', overflow: 'hidden' }}>
                <div style={{ width: '75%', height: '100%', backgroundColor: 'var(--primary-color)' }} />
              </div>
            </GlassCard>
          </SlideUp>

          {/* KPI 2 - AI Confidence Score */}
          <SlideUp>
            <GlassCard style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div style={{ padding: '6px', backgroundColor: 'rgba(139, 92, 246, 0.1)', borderRadius: '6px', color: 'var(--secondary-color)', display: 'flex' }}>
                  <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>verified_user</span>
                </div>
                <StatusBadge status={hasData ? "COMPLETED" : "STABLE"} />
              </div>
              <div className="label-technical" style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                AI Confidence Score
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', marginTop: '4px' }}>
                <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>
                  {confidenceScore}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '4px' }}>
                  {confidenceLabel}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '3px', marginTop: '16px' }}>
                <div style={{ height: '3px', flex: 1, backgroundColor: 'var(--secondary-color)', borderRadius: '1.5px' }} />
                <div style={{ height: '3px', flex: 1, backgroundColor: 'var(--secondary-color)', borderRadius: '1.5px' }} />
                <div style={{ height: '3px', flex: 1, backgroundColor: 'var(--secondary-color)', borderRadius: '1.5px' }} />
                <div style={{ height: '3px', flex: 1, backgroundColor: 'var(--secondary-color)', borderRadius: '1.5px' }} />
                <div style={{ height: '3px', flex: 1, backgroundColor: 'var(--bg-input)', borderRadius: '1.5px' }} />
              </div>
            </GlassCard>
          </SlideUp>

          {/* KPI 3 - Insights Generated */}
          <SlideUp>
            <GlassCard style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div style={{ padding: '6px', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderRadius: '6px', color: 'var(--accent-color)', display: 'flex' }}>
                  <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>lightbulb</span>
                </div>
                <StatusBadge status="ACTIVE" label="REAL-TIME" />
              </div>
              <div className="label-technical" style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Insights Generated
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginTop: '4px' }}>
                <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>
                  {insightsCount}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '16px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ display: 'inline-block' }}></span>
                <span>{insightsLabel}</span>
              </div>
            </GlassCard>
          </SlideUp>
        </StaggerContainer>

        {/* 12-Column Layout Grid matching Stitch */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '24px' }}>
          
          {/* LEFT 8-COLUMN COLUMN */}
          <div style={{ gridColumn: 'span 8', display: 'flex', flexDirection: 'column', gap: '24px' }} className="col-span-8-responsive">
            
            {/* Executive Summary Card (ChromaticBorder) */}
            <ChromaticBorder>
              <div style={{ padding: '28px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                  <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)', fontVariationSettings: "'FILL' 1" }}>
                    auto_awesome
                  </span>
                  <h3 style={{ margin: 0, fontSize: '1.125rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                    Executive Intelligence Summary
                  </h3>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }} className="grid-cols-2-responsive">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <p style={{ margin: 0, fontSize: '1.05rem', color: 'var(--text-main)', lineHeight: 1.5, fontFamily: 'Inter' }}>
                      Our analysis indicates a <span style={{ color: 'var(--primary-color)', fontWeight: 600 }}>14.2% structural shift</span> in customer acquisition data over the last quarter.
                    </p>
                    <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, fontFamily: 'Inter' }}>
                      Primary drivers include optimized algorithmic bids and a significant reduction in churn across the North American sector. Infrastructure health remains high at 99.9% with no forecasted bottlenecks.
                    </p>
                    
                    {/* Action buttons */}
                    <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                      <button 
                        className="primary-btn" 
                        onClick={() => runAnalysis('why')}
                        style={{ padding: '8px 16px', borderRadius: '6px', backgroundColor: 'var(--secondary-color)', color: 'white', border: 'none', fontSize: '0.8rem', fontWeight: 600 }}
                      >
                        Deep Dive
                      </button>
                      <button 
                        className="secondary-btn" 
                        style={{ padding: '8px 16px', borderRadius: '6px', fontSize: '0.8rem' }}
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                  
                  {/* Latency Map Sparkline Container */}
                  <div style={{ padding: '20px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-main)' }}>Network Latency Map</span>
                      <span className="label-technical" style={{ opacity: 0.6 }}>Global Cluster</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '3px', height: '80px', position: 'relative' }}>
                      {latencyPoints.map((val, i) => (
                        <div 
                          key={i} 
                          style={{ 
                            flex: 1, 
                            backgroundColor: i === 6 ? 'var(--primary-color)' : 'rgba(59, 130, 246, 0.2)', 
                            height: `${val}%`,
                            borderRadius: '1.5px'
                          }} 
                        />
                      ))}
                      {/* Pulse circle on peak */}
                      <div 
                        style={{ 
                          position: 'absolute', 
                          top: '5%', 
                          left: '37.5%', 
                          transform: 'translateX(-50%)',
                          width: '10px', 
                          height: '10px', 
                          backgroundColor: 'var(--primary-color)', 
                          borderRadius: '50%', 
                          filter: 'blur(3px)',
                          opacity: 0.5,
                          animation: 'pulse 2s infinite' 
                        }} 
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      <span>EU-West</span>
                      <span>US-East</span>
                      <span>Asia-Pac</span>
                    </div>
                  </div>
                </div>
              </div>
            </ChromaticBorder>

            {/* Generated Reports & Target Scan outputs (placed inside Left panel) */}
            {Object.keys(reports).length > 0 && (
              <GlassCard style={{ padding: '24px', borderLeft: '4px solid var(--primary-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>📝 Generated Executive Reports</h3>
                  {reportLoading && <span className="spinner-small" />}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {Object.entries(reports).map(([title, content]) => (
                    <div key={title} className="report-card fade-in" style={{ backgroundColor: 'rgba(0,0,0,0.1)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                      <h4 style={{ margin: '0 0 12px 0', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', color: 'var(--primary-color)', fontFamily: 'Geist' }}>
                        {title}
                      </h4>
                      <ReportView text={content} />
                    </div>
                  ))}
                </div>
              </GlassCard>
            )}

            {/* Dataset Health Radar Chart panel */}
            <GlassCard style={{ padding: '28px', display: 'flex', flexDirection: 'column' }}>
              <div>
                <h3 style={{ margin: '0 0 4px 0', fontSize: '1.125rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                  Dataset Health Ecosystem
                </h3>
                <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Aggregated metrics across 5 critical validation vectors.
                </p>
              </div>
              
              {/* Radar visualization layout */}
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1, padding: '32px 0', position: 'relative' }}>
                <svg style={{ width: '100%', maxWidth: '300px', transform: 'rotate(-18deg)' }} viewBox="0 0 400 400">
                  {/* Concentric Grid lines */}
                  <circle cx="200" cy="200" r="160" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  <circle cx="200" cy="200" r="120" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  <circle cx="200" cy="200" r="80" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  <circle cx="200" cy="200" r="40" fill="none" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  
                  {/* Axis lines */}
                  <line x1="200" y1="40" x2="200" y2="360" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  <line x1="40" y1="200" x2="360" y2="200" stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  
                  {/* Production overlay */}
                  <polygon 
                    points="200,60 340,160 300,320 100,320 60,160" 
                    fill="rgba(59, 130, 246, 0.08)" 
                    stroke="var(--primary-color)" 
                    strokeWidth="2" 
                  />
                  
                  {/* Staging overlay */}
                  <polygon 
                    points="200,100 280,180 250,280 150,280 120,180" 
                    fill="rgba(139, 92, 246, 0.08)" 
                    stroke="var(--secondary-color)" 
                    strokeWidth="1.5" 
                    strokeDasharray="4"
                  />
                </svg>
                
                {/* Labels absolutely positioned around container */}
                <div style={{ position: 'absolute', top: '15px', left: '50%', transform: 'translateX(-50%)', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>VALIDITY</div>
                <div style={{ position: 'absolute', right: '15px', top: '50%', transform: 'translateY(-50%)', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>LATENCY</div>
                <div style={{ position: 'absolute', bottom: '15px', right: '22%', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>VOLUME</div>
                <div style={{ position: 'absolute', bottom: '15px', left: '22%', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>DENSITY</div>
                <div style={{ position: 'absolute', left: '15px', top: '50%', transform: 'translateY(-50%)', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>SECURITY</div>
              </div>
              
              {/* Legend row */}
              <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--primary-color)', display: 'inline-block' }} />
                  Production Clusters
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--secondary-color)', display: 'inline-block' }} />
                  Staging Environment
                </div>
              </div>
            </GlassCard>

          </div>

          {/* RIGHT 4-COLUMN COLUMN */}
          <div style={{ gridColumn: 'span 4', display: 'flex', flexDirection: 'column', gap: '24px' }} className="col-span-4-responsive">
            
            {/* Card 1: 3D Telemetry Globe Container (Highly Visible) */}
            <GlassCard style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <h3 style={{ margin: '0 0 2px 0', fontSize: '1rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>Global Operations Globe</h3>
                <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)' }}>Interactive 3D network nodes telemetry.</p>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '220px', backgroundColor: 'rgba(0,0,0,0.1)', borderRadius: '6px', overflow: 'hidden' }}>
                <EnterpriseGlobe3D height={220} />
              </div>
            </GlassCard>

            {/* Card 2: Recent Analyses list feed */}
            <GlassCard style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>Recent Analyses</h3>
                <button 
                  onClick={onNavigateToUpload}
                  style={{ background: 'none', border: 'none', color: 'var(--primary-color)', fontSize: '0.75rem', fontWeight: 600, padding: 0, cursor: 'pointer' }}
                >
                  View All
                </button>
              </div>
              <p style={{ margin: '0 0 16px 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Historical task log stream.</p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {[
                  { title: 'Q4 Market Sentiment', meta: '14 mins ago • NLP Model', icon: 'pie_chart', status: 'SYNCED' },
                  { title: 'Churn Prediction V3', meta: '2 hours ago • Logistic Reg', icon: 'hub', status: 'SYNCED' },
                  { title: 'Supply Chain Optima', meta: '5 hours ago • Linear Pro', icon: 'inventory_2', status: 'SYNCING' },
                  { title: 'Regional Yield Delta', meta: 'Yesterday • Geo-Analytic', icon: 'query_stats', status: 'SYNCED' },
                  { title: 'Fraud Vector Test', meta: 'Yesterday • Neural Net', icon: 'shield', status: 'FAILED' }
                ].map((item, idx) => (
                  <div 
                    key={idx} 
                    style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between', 
                      padding: '10px 12px', 
                      backgroundColor: 'rgba(255, 255, 255, 0.02)', 
                      borderRadius: '6px', 
                      border: '1px solid var(--border-color)' 
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{ color: 'var(--text-muted)', display: 'flex' }}>
                        <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>{item.icon}</span>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>{item.title}</span>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{item.meta}</span>
                      </div>
                    </div>
                    
                    {/* Status dot */}
                    <span 
                      style={{ 
                        width: '6px', 
                        height: '6px', 
                        borderRadius: '50%', 
                        backgroundColor: item.status === 'SYNCED' ? '#10b981' : item.status === 'SYNCING' ? '#f59e0b' : '#ef4444' 
                      }} 
                    />
                  </div>
                ))}
              </div>
            </GlassCard>

            {/* Card 3: Secondary Intelligence Insight card */}
            <GlassCard 
              style={{ 
                padding: '20px', 
                background: 'linear-gradient(to bottom right, var(--bg-card), rgba(139, 92, 246, 0.05))',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div>
                <span 
                  style={{ 
                    fontSize: '9px', 
                    fontWeight: 600, 
                    color: 'var(--secondary-color)', 
                    backgroundColor: 'rgba(139, 92, 246, 0.1)', 
                    padding: '3px 8px', 
                    borderRadius: '4px',
                    border: '1px solid rgba(139, 92, 246, 0.2)',
                    display: 'inline-block'
                  }}
                >
                  PRO FEATURE
                </span>
                <h4 style={{ margin: '8px 0 6px 0', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-main)', lineHeight: 1.3, fontFamily: 'Geist' }}>
                  Anomaly Detected in Ad-Yield Pipeline
                </h4>
                <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                  A sudden 15% drop in conversion from mobile devices was identified. The AI suggests a potential API timeout issue in checkout.
                </p>
              </div>
              <button 
                className="primary-btn" 
                onClick={() => runAnalysis('what')}
                style={{ width: '100%', fontSize: '0.8rem', padding: '8px 0', borderRadius: '4px', backgroundColor: 'var(--primary-color)', color: 'white', border: 'none' }}
              >
                Launch Diagnostic
              </button>
              
              {/* Back lighting glow effect */}
              <div 
                style={{ 
                  position: 'absolute', 
                  right: '-20px', 
                  bottom: '-20px', 
                  width: '80px', 
                  height: '80px', 
                  borderRadius: '50%', 
                  backgroundColor: 'rgba(139, 92, 246, 0.08)', 
                  filter: 'blur(24px)' 
                }} 
              />
            </GlassCard>

          </div>

        </div>

        {/* Analytics Command Center panel (Full columns at bottom) */}
        <GlassCard style={{ padding: '24px' }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '1.125rem', color: 'var(--text-main)', fontFamily: 'Geist', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span className="material-symbols-outlined">settings_suggest</span>
            Analytics Command Center
          </h3>
          <p style={{ margin: '0 0 18px 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Select an analysis module to execute on active datasets.
          </p>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button 
              className="primary-btn" 
              onClick={() => runAnalysis('what')}
              disabled={!hasData}
              style={{ opacity: hasData ? 1 : 0.5, cursor: hasData ? 'pointer' : 'not-allowed' }}
            >
              🔍 Run Automated Driver Scan
            </button>
            <button 
              className="secondary-btn" 
              onClick={loadFailures} 
              disabled={!hasData}
              style={{ borderColor: 'var(--danger-color)', color: 'var(--danger-color)', opacity: hasData ? 1 : 0.5, cursor: hasData ? 'pointer' : 'not-allowed' }}
            >
              📋 Highlighted Target Records
            </button>
            <button 
              className="primary-btn" 
              onClick={() => runAnalysis('why')}
              disabled={!hasData}
              style={{ backgroundColor: 'var(--secondary-color)', opacity: hasData ? 1 : 0.5, cursor: hasData ? 'pointer' : 'not-allowed' }}
            >
              🧠 Generate Executive Insights Report
            </button>
          </div>
        </GlassCard>

        {/* Infrastructure Status Banner Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginTop: '8px' }}>
          <div style={{ display: 'flex', gap: '12px', padding: '16px', borderLeft: '3px solid var(--accent-color)', backgroundColor: 'var(--bg-card)', borderRadius: '4px' }}>
            <span className="material-symbols-outlined" style={{ color: 'var(--accent-color)' }}>cloud_done</span>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>System Core</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>Fully Operational</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', padding: '16px', borderLeft: '3px solid var(--primary-color)', backgroundColor: 'var(--bg-card)', borderRadius: '4px' }}>
            <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)' }}>speed</span>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Avg. Query Time</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>128ms</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', padding: '16px', borderLeft: '3px solid var(--secondary-color)', backgroundColor: 'var(--bg-card)', borderRadius: '4px' }}>
            <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)' }}>storage</span>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Storage Utilization</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>62.8% Capacity</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', padding: '16px', borderLeft: '3px solid var(--text-muted)', backgroundColor: 'var(--bg-card)', borderRadius: '4px' }}>
            <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)' }}>lock</span>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Encryption</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>AES-256 Enabled</div>
            </div>
          </div>
        </div>

      </div>
    </FadeIn>
  );
}
