import React from 'react';
import GlassCard from '../components/GlassCard';
import ChromaticBorder from '../components/ChromaticBorder';
import FadeIn from '../components/animation/FadeIn';
import AmbientAIOrb3D from '../three/AmbientAIOrb3D';
import Upload from '../Upload';

export default function WelcomeDashboard({ 
  user, 
  onNavigateToTab, 
  handleUploadStart, 
  handleUploadSuccess 
}) {
  const username = user?.name || "Enterprise Analyst";
  const userRole = user?.role || "Lead Analyst";

  return (
    <FadeIn>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* Welcome Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)', margin: '0 0 6px 0', fontFamily: 'Geist', letterSpacing: '-0.02em' }}>
              Welcome back, {username}
            </h1>
            <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              Logged in as <span style={{ color: 'var(--primary-color)', fontWeight: 500 }}>{userRole}</span> • Secure session active
            </p>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            System ready for data ingress
          </div>
        </div>

        {/* 12-Column Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '24px' }} className="grid-cols-12-responsive">
          
          {/* LEFT 8-COLUMN: START YOUR FIRST ANALYSIS TIMELINE & UPLOAD */}
          <div style={{ gridColumn: 'span 8', display: 'flex', flexDirection: 'column', gap: '24px' }} className="col-span-8-responsive">
            
            {/* Start Your First Analysis Workflow */}
            <ChromaticBorder>
              <div style={{ padding: '28px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
                  <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)' }}>start</span>
                  <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                    Start Your First Analysis
                  </h3>
                </div>

                {/* Vertical Process Timeline */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative' }}>
                  
                  {/* Step 1 */}
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      backgroundColor: 'rgba(59, 130, 246, 0.1)',
                      border: '1.5px solid var(--primary-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--primary-color)',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      flexShrink: 0
                    }}>
                      1
                    </div>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', margin: '0 0 4px 0', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                        Upload CSV Dataset
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.45 }}>
                        Drag and drop or select your structured CSV file to initiate secure local profiling.
                      </p>
                    </div>
                  </div>

                  {/* Step 2 */}
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      backgroundColor: 'var(--bg-input)',
                      border: '1.5px solid var(--border-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--text-muted)',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      flexShrink: 0
                    }}>
                      2
                    </div>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', margin: '0 0 4px 0', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                        Align Variables & Targets
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.45 }}>
                        Review automated target suggestions and define acronym values to calibrate the intelligence engine.
                      </p>
                    </div>
                  </div>

                  {/* Step 3 */}
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      backgroundColor: 'var(--bg-input)',
                      border: '1.5px solid var(--border-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--text-muted)',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      flexShrink: 0
                    }}>
                      3
                    </div>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', margin: '0 0 4px 0', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                        Collaborate with AI Copilot
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.45 }}>
                        Interrogate your data structures and domain definitions dynamically in the chat workspace.
                      </p>
                    </div>
                  </div>

                  {/* Step 4 */}
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      backgroundColor: 'var(--bg-input)',
                      border: '1.5px solid var(--border-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--text-muted)',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      flexShrink: 0
                    }}>
                      4
                    </div>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', margin: '0 0 4px 0', color: 'var(--text-main)', fontFamily: 'Geist' }}>
                        Export Executive Briefing
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.45 }}>
                        Download generated linear model summaries and driver scans as PDF dossiers for leadership.
                      </p>
                    </div>
                  </div>

                </div>
              </div>
            </ChromaticBorder>

            {/* Ingestion Box Card */}
            <GlassCard style={{ padding: '28px' }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.1rem', fontFamily: 'Geist' }}>
                Secure Data Ingestion
              </h3>
              <p style={{ margin: '0 0 20px 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Your data is parsed locally on corporate servers. Files are processed with strict isolation protocols.
              </p>
              <Upload onUploadSuccess={handleUploadSuccess} onUploadStart={handleUploadStart} />
            </GlassCard>

            {/* Recent Datasets empty list */}
            <GlassCard style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
                <h3 style={{ margin: 0, fontSize: '1rem', fontFamily: 'Geist' }}>📁 Active Workspaces</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>0 connected</span>
              </div>
              <div style={{ textAlign: 'center', padding: '30px 0' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '36px', color: 'var(--text-muted)', marginBottom: '10px' }}>folder_open</span>
                <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  No active datasets. Upload a CSV above to display files.
                </p>
              </div>
            </GlassCard>

          </div>

          {/* RIGHT 4-COLUMN: AI ORB STATE & QUICK ACTIONS */}
          <div style={{ gridColumn: 'span 4', display: 'flex', flexDirection: 'column', gap: '24px' }} className="col-span-4-responsive">
            
            {/* System Status Orb */}
            <GlassCard style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center', textAlign: 'center' }}>
              <div>
                <h3 style={{ margin: '0 0 2px 0', fontSize: '1rem', color: 'var(--text-main)', fontFamily: 'Geist' }}>Cognitive Core</h3>
                <span style={{ fontSize: '11px', color: 'var(--accent-color)', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--accent-color)' }}></span>
                  READY FOR INGEST
                </span>
              </div>
              
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', margin: '10px 0' }}>
                <AmbientAIOrb3D height={180} isThinking={false} />
              </div>

              <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                Connect a structured dataset pipeline to activate full real-time telemetry overlays.
              </p>
            </GlassCard>

            {/* Quick Actions Panel */}
            <GlassCard style={{ padding: '20px' }}>
              <h3 style={{ margin: '0 0 14px 0', fontSize: '1rem', fontFamily: 'Geist' }}>Quick Actions</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <button
                  onClick={() => onNavigateToTab('analysis')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)',
                    borderRadius: '6px',
                    textAlign: 'left',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    width: '100%',
                    fontWeight: 500
                  }}
                >
                  <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)', fontSize: '18px' }}>upload_file</span>
                  Ingest Dataset File
                </button>

                <button
                  onClick={() => onNavigateToTab('manuals')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)',
                    borderRadius: '6px',
                    textAlign: 'left',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    width: '100%',
                    fontWeight: 500
                  }}
                >
                  <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)', fontSize: '18px' }}>menu_book</span>
                  Search Reference Manuals
                </button>

                <button
                  onClick={() => onNavigateToTab('settings')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)',
                    borderRadius: '6px',
                    textAlign: 'left',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    width: '100%',
                    fontWeight: 500
                  }}
                >
                  <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: '18px' }}>settings</span>
                  System Settings
                </button>
              </div>
            </GlassCard>

            {/* Recent Reports empty feed */}
            <GlassCard style={{ padding: '20px' }}>
              <h3 style={{ margin: '0 0 12px 0', fontSize: '1rem', fontFamily: 'Geist' }}>📝 Recent Reports</h3>
              <div style={{ textAlign: 'center', padding: '16px 0', border: '1px dashed var(--border-color)', borderRadius: '6px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>No reports compiled yet.</span>
              </div>
            </GlassCard>

          </div>

        </div>

      </div>
    </FadeIn>
  );
}
