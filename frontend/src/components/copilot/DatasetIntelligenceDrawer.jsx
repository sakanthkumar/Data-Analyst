import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';

function formatBytes(bytes, decimals = 2) {
  if (!bytes) return 'N/A';
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export default function DatasetIntelligenceDrawer({
  activeDataset,
  domainProfile,
  conversationHistory,
  recentQuestions,
  pinnedInsights,
  onSelectQuestion,
  onUnpinInsight,
  onClose
}) {
  const [activeTab, setActiveTab] = useState('telemetry');
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
      console.error("Failed to load reports in intelligence drawer", e);
    } finally {
      setLoadingReports(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'reports') {
      fetchReports();
    }
  }, [activeTab, conversationHistory]);

  if (!activeDataset) return null;

  const rows = activeDataset.shape?.[0] || 0;
  const cols = activeDataset.shape?.[1] || 0;
  const fileSize = formatBytes(activeDataset.file_size_bytes);
  const targetCol = activeDataset.target_column || domainProfile?.target_column || 'Not Defined';

  // Quality analysis calculations
  const missingCount = Object.values(activeDataset.missing_values || {}).reduce((a, b) => a + b, 0);
  const duplicateCount = activeDataset.duplicate_rows || 0;
  const outlierCount = Object.values(activeDataset.outliers || {}).reduce((a, b) => a + b, 0);
  const totalCells = rows * cols;
  const qualityScore = totalCells > 0
    ? Math.max(0, Math.min(100, Math.round((1 - (missingCount + duplicateCount) / totalCells) * 100)))
    : 100;

  const getQualityText = (score) => {
    if (score >= 90) return 'Excellent';
    if (score >= 70) return 'Fair';
    return 'Action Required';
  };

  const getQualityColor = (score) => {
    if (score >= 90) return '#10b981'; // Success Emerald
    if (score >= 70) return '#f59e0b'; // Warning Amber
    return '#f43f5e'; // Error Rose
  };

  return (
    <aside
      className="glass-card custom-scrollbar"
      style={{
        width: '320px',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-sidebar)',
        borderLeft: '1px solid var(--border-color)',
        borderRadius: 0,
        height: '100%',
        boxSizing: 'border-box',
        overflow: 'hidden',
        zIndex: 20,
        position: 'relative'
      }}
    >
      {/* Drawer Header */}
      <div 
        style={{ 
          padding: '16px 20px', 
          borderBottom: '1px solid var(--border-color)',
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between' 
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <h3 
            style={{ 
              fontSize: '0.85rem', 
              textTransform: 'uppercase', 
              letterSpacing: '0.08em', 
              color: 'var(--primary-color)', 
              margin: 0,
              fontFamily: 'Geist',
              fontWeight: 700
            }}
          >
            Dataset Intelligence
          </h3>
          <span 
            style={{ 
              fontSize: '0.75rem', 
              color: 'var(--text-muted)',
              maxWidth: '220px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
            title={activeDataset.filename}
          >
            {activeDataset.filename}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '4px',
            display: 'flex',
            alignItems: 'center',
            borderRadius: '4px'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-main)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          title="Collapse Drawer"
        >
          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
            close_fullscreen
          </span>
        </button>
      </div>

      {/* Tabs Selector Navigation */}
      <div 
        style={{ 
          display: 'flex', 
          borderBottom: '1px solid var(--border-color)',
          backgroundColor: 'rgba(255,255,255,0.01)'
        }}
      >
        {['telemetry', 'insights', 'reports'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              flex: 1,
              padding: '12px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid var(--primary-color)' : '2px solid transparent',
              color: activeTab === tab ? 'var(--text-main)' : 'var(--text-muted)',
              fontSize: '0.785rem',
              fontWeight: activeTab === tab ? 600 : 500,
              cursor: 'pointer',
              textTransform: 'capitalize',
              borderRadius: 0,
              transition: 'all 0.2s'
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content Container */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', boxSizing: 'border-box' }} className="custom-scrollbar">
        
        {/* TAB 1: TELEMETRY */}
        {activeTab === 'telemetry' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Health Score KPI Card */}
            <div 
              style={{ 
                background: 'rgba(255, 255, 255, 0.02)', 
                border: '1px solid var(--border-color)', 
                borderRadius: '8px', 
                padding: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Dataset Health</span>
                <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  {getQualityText(qualityScore)}
                </span>
              </div>
              <div 
                style={{ 
                  backgroundColor: `${getQualityColor(qualityScore)}15`, 
                  border: `1px solid ${getQualityColor(qualityScore)}30`, 
                  color: getQualityColor(qualityScore), 
                  padding: '4px 10px', 
                  borderRadius: '16px', 
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: getQualityColor(qualityScore) }} />
                {qualityScore}%
              </div>
            </div>

            {/* Quality Metrics Grid */}
            <div>
              <h4 style={{ fontSize: '0.785rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px', letterSpacing: '0.04em' }}>
                Data Diagnostics
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div style={{ background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Missing Values</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: missingCount > 0 ? 'var(--warning-color)' : 'var(--text-main)', marginTop: '4px' }}>
                    {missingCount.toLocaleString()}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Outliers Found</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: outlierCount > 0 ? 'var(--warning-color)' : 'var(--text-main)', marginTop: '4px' }}>
                    {outlierCount.toLocaleString()}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px', gridColumn: 'span 2' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Duplicate Rows</span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 700, color: duplicateCount > 0 ? 'var(--warning-color)' : 'var(--text-main)' }}>
                      {duplicateCount.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Profile Metrics */}
            <div>
              <h4 style={{ fontSize: '0.785rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px', letterSpacing: '0.04em' }}>
                Structural Context
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.785rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Rows × Columns</span>
                  <span style={{ color: 'var(--text-main)', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>
                    {rows.toLocaleString()} × {cols}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.785rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>File Size</span>
                  <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>{fileSize}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.785rem', alignItems: 'center' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Target Column</span>
                  <span 
                    style={{ 
                      color: 'var(--primary-color)', 
                      background: 'rgba(59, 130, 246, 0.08)', 
                      padding: '2px 8px', 
                      borderRadius: '4px',
                      border: '1px solid rgba(59, 130, 246, 0.15)',
                      fontWeight: 600,
                      maxWidth: '120px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontSize: '0.72rem'
                    }}
                    title={targetCol}
                  >
                    {targetCol}
                  </span>
                </div>
                {domainProfile?.domain && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.785rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Domain Profile</span>
                    <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>{domainProfile.domain}</span>
                  </div>
                )}
                {domainProfile?.analysis_type && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.785rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Analysis Class</span>
                    <span style={{ color: 'var(--text-main)', fontWeight: 500, textTransform: 'capitalize' }}>
                      {domainProfile.analysis_type}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: INSIGHTS */}
        {activeTab === 'insights' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            {/* Pinned Insights */}
            <div>
              <h4 style={{ fontSize: '0.785rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>push_pin</span>
                Pinned Insights
              </h4>
              {pinnedInsights.length === 0 ? (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '12px 4px' }}>
                  Pin response cards to save key summaries here.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {pinnedInsights.map((insight, idx) => (
                    <div 
                      key={insight.id || idx}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        background: 'rgba(255, 255, 255, 0.02)',
                        border: '1px solid var(--border-color)',
                        borderLeft: '2px solid var(--primary-color)',
                        position: 'relative'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                        <span 
                          style={{ 
                            fontSize: '0.785rem', 
                            fontWeight: 600, 
                            color: 'var(--primary-color)',
                            maxWidth: '200px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          {insight.question}
                        </span>
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
                          title="Unpin Insight"
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
                          lineHeight: '1.45',
                          display: '-webkit-box',
                          WebkitLineClamp: 4,
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

            {/* Recent Questions / History */}
            <div>
              <h4 style={{ fontSize: '0.785rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '10px', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>history</span>
                Recent Questions
              </h4>
              {recentQuestions.length === 0 ? (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '12px 4px' }}>
                  No recent questions.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {recentQuestions.slice(0, 5).map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSelectQuestion(q)}
                      style={{
                        width: '100%',
                        padding: '8px 10px',
                        borderRadius: '6px',
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border-color)',
                        color: 'var(--text-main)',
                        fontSize: '0.785rem',
                        textAlign: 'left',
                        cursor: 'pointer',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        transition: 'all 0.2s ease',
                        fontFamily: 'Inter, sans-serif'
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
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: REPORTS */}
        {activeTab === 'reports' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <h4 style={{ fontSize: '0.785rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>summarize</span>
              Reports Created
            </h4>
            {loadingReports ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)', padding: '12px 4px' }}>
                <span className="spinner-small" />
                <span>Loading...</span>
              </div>
            ) : reports.length === 0 ? (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '12px 4px' }}>
                No reports generated yet. Run target analysis to compile executive reports.
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
                        background: 'rgba(255,255,255,0.01)',
                        border: '1px solid var(--border-color)',
                        fontSize: '0.785rem',
                        color: 'var(--text-main)'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: '6px', marginBottom: '2px' }}>
                        <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--primary-color)' }}>
                          dataset
                        </span>
                        <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={report.machine_name}>
                          {report.machine_name || activeDataset?.filename}
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

                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
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
        )}
      </div>
    </aside>
  );
}
