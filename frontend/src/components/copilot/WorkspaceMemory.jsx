import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';

export default function WorkspaceMemory({ 
  activeDataset, 
  domainProfile, 
  conversationHistory, 
  recentQuestions, 
  pinnedInsights, 
  onSelectQuestion, 
  onUnpinInsight 
}) {
  const [reports, setReports] = useState([]);
  const [loadingReports, setLoadingReports] = useState(false);

  // Load generated reports from backend
  const fetchReports = async () => {
    setLoadingReports(true);
    try {
      const res = await api.getReportsList();
      if (Array.isArray(res.data)) {
        setReports(res.data);
      }
    } catch (e) {
      console.error("Failed to load reports in memory panel", e);
    } finally {
      setLoadingReports(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [conversationHistory]); // Refresh when conversation changes

  // Compute statistics
  const userTurns = conversationHistory.filter(m => m.role === 'user').length;
  const assistantMsgs = conversationHistory.filter(m => m.role === 'ai' || m.role === 'assistant');
  const avgConfidence = assistantMsgs.length > 0 
    ? Math.round(assistantMsgs.reduce((sum, m) => sum + (m.confidence || 90), 0) / assistantMsgs.length)
    : 0;

  // Resolve last updated time
  const lastMsg = conversationHistory[conversationHistory.length - 1];
  const lastUpdated = lastMsg?.timestamp || 'N/A';

  return (
    <aside 
      className="glass-card" 
      style={{ 
        width: '300px', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '24px', 
        padding: '20px',
        backgroundColor: 'var(--bg-sidebar)',
        borderLeft: '1px solid var(--border-color)',
        borderRight: 'none',
        borderRadius: 0,
        height: '100%',
        boxSizing: 'border-box',
        overflowY: 'auto'
      }}
    >
      {/* Section 1: Active Context */}
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
          Active Focus
        </h3>
        <div 
          style={{ 
            background: 'var(--bg-input)', 
            border: '1px solid var(--border-color)', 
            borderRadius: '8px', 
            padding: '12px',
            fontSize: '0.85rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--primary-color)' }}>
              dataset
            </span>
            <span style={{ color: 'var(--text-main)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '220px' }} title={activeDataset?.filename}>
              {activeDataset?.filename || 'No Dataset'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--accent-color)' }}>
              analytics
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.785rem' }}>
              Type: <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>{domainProfile?.analysis_type || 'General'}</span>
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--warning-color)' }}>
              update
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.785rem' }}>
              Last Chat: <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>{lastUpdated}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Section 2: Session Stats */}
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
          Session Stats
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div style={{ background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary-color)' }}>{userTurns}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '2px' }}>Questions</div>
          </div>
          <div style={{ background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-color)' }}>{avgConfidence}%</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '2px' }}>Avg Cert</div>
          </div>
        </div>
      </div>

      {/* Section 3: Recent Questions */}
      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '10px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Recent Questions
        </h3>
        {recentQuestions.length === 0 ? (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No questions asked yet.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {recentQuestions.slice(0, 5).map((q, idx) => (
              <div 
                key={idx}
                onClick={() => onSelectQuestion(q)}
                style={{
                  padding: '8px 10px',
                  borderRadius: '6px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-main)',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--primary-color)';
                  e.currentTarget.style.transform = 'translateX(2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-color)';
                  e.currentTarget.style.transform = 'none';
                }}
              >
                {q}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 4: Pinned Insights */}
      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '10px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Pinned Insights
        </h3>
        {pinnedInsights.length === 0 ? (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Pin cards to save key metrics.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {pinnedInsights.map((insight, idx) => (
              <div 
                key={idx}
                style={{
                  padding: '10px',
                  borderRadius: '8px',
                  background: 'rgba(59, 130, 246, 0.03)',
                  border: '1px solid rgba(59, 130, 246, 0.15)',
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                  <div 
                    style={{ 
                      fontSize: '0.8rem', 
                      fontWeight: 600, 
                      color: 'var(--primary-color)',
                      maxWidth: '180px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap' 
                    }}
                  >
                    {insight.question}
                  </div>
                  <button
                    onClick={() => onUnpinInsight(insight.id)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      padding: 0,
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.color = 'var(--danger-color)'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>
                      close
                    </span>
                  </button>
                </div>
                <p 
                  style={{ 
                    fontSize: '0.75rem', 
                    color: 'var(--text-main)', 
                    margin: 0, 
                    lineHeight: '1.4',
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden'
                  }}
                >
                  {insight.analysis}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 5: Generated Reports */}
      <div>
        <h3 
          style={{ 
            fontSize: '0.8rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.08em', 
            color: 'var(--text-muted)', 
            marginBottom: '10px',
            fontFamily: 'Geist',
            fontWeight: 700
          }}
        >
          Reports Created
        </h3>
        {loadingReports ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <div className="spinner-small" />
            <span>Loading...</span>
          </div>
        ) : reports.length === 0 ? (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No reports created yet.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {reports.map((report) => {
              const createdDate = report.timestamp ? new Date(report.timestamp) : new Date();
              const formattedTime = createdDate.toLocaleString([], {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              });

              return (
                <div 
                  key={report.id}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    fontSize: '0.8rem',
                    color: 'var(--text-main)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '6px', marginBottom: '2px' }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '16px', color: 'var(--primary-color)' }}>
                      dataset
                    </span>
                    <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={report.machine_name}>
                      {report.machine_name || activeDataset?.filename || 'Unknown Dataset'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>
                      {report.analysis_type || 'Custom Report'}
                    </span>
                    <span 
                      style={{ 
                        background: 'rgba(16, 185, 129, 0.08)', 
                        border: '1px solid rgba(16, 185, 129, 0.2)', 
                        color: 'var(--accent-color)', 
                        padding: '1px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}
                    >
                      Completed
                    </span>
                  </div>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>
                      schedule
                    </span>
                    <span>{formattedTime}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
