import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import ChromaticBorder from '../components/ChromaticBorder';
import StatusBadge from '../components/StatusBadge';
import FadeIn from '../components/animation/FadeIn';
import { api } from '../services/api';

// ---------------------------------------------------------------------------
// NO MOCK DATA POLICY
// All data originates from activeDataset (EDA response), domainProfile,
// plots (base64), reports (backend text), and paginated /data API.
// ---------------------------------------------------------------------------

const WORKSPACE_TABS = [
  { id: 'overview',       label: 'Overview',       icon: 'dashboard'       },
  { id: 'preview',        label: 'Data Preview',   icon: 'table'           },
  { id: 'schema',         label: 'Schema',         icon: 'schema'          },
  { id: 'quality',        label: 'Data Quality',   icon: 'health_metrics'  },
  { id: 'correlations',   label: 'Correlations',   icon: 'bubble_chart'    },
  { id: 'distributions',  label: 'Distributions',  icon: 'bar_chart_4_bars'},
  { id: 'insights',       label: 'AI Insights',    icon: 'auto_awesome'    },
];

// ── Helpers ─────────────────────────────────────────────────────────────────

function formatFileSize(bytes) {
  if (!bytes || bytes <= 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getDtypeBadgeColor(dtype) {
  if (!dtype) return 'var(--text-muted)';
  const d = dtype.toLowerCase();
  if (d.includes('int') || d.includes('float')) return 'var(--primary-color)';
  if (d.includes('object') || d.includes('str') || d.includes('categ')) return 'var(--secondary-color)';
  if (d.includes('bool')) return 'var(--accent-color)';
  if (d.includes('date') || d.includes('time')) return 'var(--warning-color)';
  return 'var(--text-muted)';
}

function getCorrelationColor(val) {
  if (val === null || val === undefined) return 'var(--bg-input)';
  const abs = Math.abs(val);
  const alpha = Math.min(0.9, abs * 0.9 + 0.1);
  if (val > 0) return `rgba(59, 130, 246, ${alpha})`;
  return `rgba(239, 68, 68, ${alpha})`;
}

function getMissingBadgeStyle(pct) {
  if (pct === 0) return { color: 'var(--accent-color)', bg: 'rgba(16,185,129,0.1)' };
  if (pct < 5) return { color: 'var(--warning-color)', bg: 'rgba(245,158,11,0.1)' };
  return { color: 'var(--danger-color)', bg: 'rgba(239,68,68,0.1)' };
}

function SectionHeader({ icon, title, subtitle, children }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <span className="material-symbols-outlined" style={{ fontSize: '20px', color: 'var(--primary-color)', fontVariationSettings: "'FILL' 1" }}>{icon}</span>
          <h2 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Geist', color: 'var(--text-main)' }}>{title}</h2>
        </div>
        {subtitle && <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-muted)' }}>{subtitle}</p>}
      </div>
      {children && <div>{children}</div>}
    </div>
  );
}

function MetaChip({ icon, label, value, accent }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '8px',
      padding: '10px 14px',
      backgroundColor: 'var(--bg-input)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      minWidth: '120px',
      flex: '1 1 auto',
    }}>
      <span className="material-symbols-outlined" style={{ fontSize: '18px', color: accent || 'var(--primary-color)', fontVariationSettings: "'FILL' 1" }}>{icon}</span>
      <div>
        <div style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist', marginTop: '1px' }}>{value ?? '—'}</div>
      </div>
    </div>
  );
}

// ── Tab Sections ─────────────────────────────────────────────────────────────

// SECTION 1 — OVERVIEW
function OverviewSection({ data, domainProfile, reports, reportLoading, runAnalysis, loadFailures, downloadPDF, onNavigateToTab }) {
  const rows = data?.shape?.[0] ?? 0;
  const cols = data?.shape?.[1] ?? 0;
  const missingTotal = data?.missing_values ? Object.values(data.missing_values).reduce((a, b) => a + b, 0) : 0;
  const outliersTotal = data?.outliers ? Object.values(data.outliers).reduce((a, b) => a + b, 0) : 0;
  const duplicates = data?.duplicate_rows ?? 0;
  const completeness = rows > 0 && cols > 0
    ? (((rows * cols - missingTotal) / (rows * cols)) * 100).toFixed(1)
    : '—';
  const hasReports = Object.keys(reports || {}).length > 0;

  // Top correlations with target
  const targetCol = data?.target_column || domainProfile?.target_column;
  const topCorrelations = useMemo(() => {
    const corrs = data?.correlations;
    const tCol = data?.target_column || domainProfile?.target_column;
    if (!corrs || !tCol || !corrs[tCol]) return [];
    return Object.entries(corrs[tCol])
      .filter(([col]) => col !== tCol)
      .map(([col, val]) => ({ col, val }))
      .filter(x => x.val !== null && x.val !== undefined)
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 5);
  }, [data, domainProfile]);

  const ReportView = ({ text }) => {
    if (!text) return null;
    return (
      <div style={{ fontSize: '0.875rem', lineHeight: 1.65, color: 'var(--text-main)' }}>
        {text.split('\n').map((line, i) => {
          if (line.startsWith('###')) return <h4 key={i} style={{ color: 'var(--primary-color)', fontSize: '0.9rem', margin: '12px 0 4px', fontFamily: 'Geist' }}>{line.replace(/^###\s*/, '')}</h4>;
          if (line.startsWith('**') && line.endsWith('**')) return <strong key={i} style={{ display: 'block', marginTop: '8px', color: 'var(--text-main)' }}>{line.replace(/\*\*/g, '')}</strong>;
          if (!line.trim()) return <br key={i} />;
          return <p key={i} style={{ margin: '2px 0', color: 'var(--text-muted)' }}>{line}</p>;
        })}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Analysis Summary Card */}
      <ChromaticBorder>
        <div style={{ padding: '24px' }}>
          <SectionHeader icon="summarize" title="Analysis Summary" subtitle={`Dataset: ${data?.filename ?? '—'} · ${domainProfile?.domain ?? 'Domain detecting...'}`} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
            <MetaChip icon="database" label="Total Rows" value={rows.toLocaleString()} />
            <MetaChip icon="view_column" label="Columns" value={cols} />
            <MetaChip icon="target" label="Target Column" value={data?.target_column || '—'} accent="var(--secondary-color)" />
            <MetaChip icon="analytics" label="Analysis Type" value={domainProfile?.analysis_type ? domainProfile.analysis_type.charAt(0).toUpperCase() + domainProfile.analysis_type.slice(1) : '—'} accent="var(--accent-color)" />
            <MetaChip icon="folder_zip" label="File Size" value={formatFileSize(data?.file_size_bytes)} />
            <MetaChip icon="percent" label="Completeness" value={completeness !== '—' ? `${completeness}%` : '—'} accent={parseFloat(completeness) >= 95 ? 'var(--accent-color)' : parseFloat(completeness) >= 80 ? 'var(--warning-color)' : 'var(--danger-color)'} />
          </div>
        </div>
      </ChromaticBorder>

      {/* Quality Snapshot Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <GlassCard style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span className="material-symbols-outlined" style={{ color: missingTotal > 0 ? 'var(--warning-color)' : 'var(--accent-color)', fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>data_alert</span>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Missing Values</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>{missingTotal.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>across {data?.missing_values ? Object.values(data.missing_values).filter(v => v > 0).length : 0} columns</div>
        </GlassCard>
        <GlassCard style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span className="material-symbols-outlined" style={{ color: outliersTotal > 0 ? 'var(--warning-color)' : 'var(--accent-color)', fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>scatter_plot</span>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Outliers Detected</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>{outliersTotal.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>IQR method · {data?.outliers ? Object.keys(data.outliers).length : 0} numeric cols</div>
        </GlassCard>
        <GlassCard style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span className="material-symbols-outlined" style={{ color: duplicates > 0 ? 'var(--warning-color)' : 'var(--accent-color)', fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>content_copy</span>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Duplicate Rows</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>{duplicates.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>{rows > 0 ? `${((duplicates / rows) * 100).toFixed(2)}% of dataset` : '—'}</div>
        </GlassCard>
        <GlassCard style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span className="material-symbols-outlined" style={{ color: data?.failure_rate > 30 ? 'var(--danger-color)' : 'var(--accent-color)', fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>flag</span>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Target Event Rate</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'Geist' }}>{data?.failure_rate != null ? `${data.failure_rate}%` : '—'}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>{data?.failure_count?.toLocaleString() ?? '—'} target instances</div>
        </GlassCard>
      </div>

      {/* Top Correlations + Workspace Actions row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }} className="col-span-8-responsive">
        {/* Top Correlations with Target */}
        <GlassCard style={{ padding: '22px' }}>
          <SectionHeader icon="hub" title="Top Feature Correlations" subtitle={targetCol ? `Pearson correlations with target: ${targetCol}` : 'No target column confirmed'} />
          {topCorrelations.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {topCorrelations.map(({ col, val }) => (
                <div key={col} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ width: '140px', fontSize: '0.8rem', color: 'var(--text-main)', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{col}</div>
                  <div style={{ flex: 1, height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.abs(val) * 100}%`,
                      height: '100%',
                      backgroundColor: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)',
                      borderRadius: '3px',
                      transition: 'width 0.6s ease'
                    }} />
                  </div>
                  <div style={{ width: '52px', textAlign: 'right', fontSize: '0.8rem', fontWeight: 700, fontFamily: 'Geist', color: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)' }}>
                    {val > 0 ? '+' : ''}{val.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              {targetCol ? 'No numeric correlations available.' : 'Confirm a target column to see correlations.'}
            </div>
          )}
        </GlassCard>

        {/* Workspace Actions Panel */}
        <GlassCard style={{ padding: '22px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <SectionHeader icon="rocket_launch" title="Workspace Actions" />
          <button onClick={() => runAnalysis('what')} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'11px 14px', backgroundColor:'var(--primary-color)', color:'white', border:'none', borderRadius:'8px', fontWeight:600, cursor:'pointer', fontSize:'0.875rem', width:'100%' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px' }}>manage_search</span>
            Run Driver Scan
          </button>
          <button onClick={() => runAnalysis('why')} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'11px 14px', backgroundColor:'var(--secondary-color)', color:'white', border:'none', borderRadius:'8px', fontWeight:600, cursor:'pointer', fontSize:'0.875rem', width:'100%' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px' }}>psychology</span>
            Generate AI Report
          </button>
          <button onClick={loadFailures} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'11px 14px', backgroundColor:'transparent', color:'var(--warning-color)', border:'1px solid var(--warning-color)', borderRadius:'8px', fontWeight:600, cursor:'pointer', fontSize:'0.875rem', width:'100%' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px' }}>flag_circle</span>
            View Target Records
          </button>
          <button onClick={downloadPDF} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'11px 14px', backgroundColor:'transparent', color:'var(--accent-color)', border:'1px solid var(--accent-color)', borderRadius:'8px', fontWeight:600, cursor:'pointer', fontSize:'0.875rem', width:'100%' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px' }}>download</span>
            Export PDF Report
          </button>
          <button onClick={() => onNavigateToTab('copilot')} style={{ display:'flex', alignItems:'center', gap:'10px', padding:'11px 14px', backgroundColor:'transparent', color:'var(--text-muted)', border:'1px solid var(--border-color)', borderRadius:'8px', fontWeight:500, cursor:'pointer', fontSize:'0.875rem', width:'100%' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px' }}>chat</span>
            Open AI Copilot
          </button>
        </GlassCard>
      </div>

      {/* Generated Reports viewer */}
      {hasReports && (
        <GlassCard style={{ padding: '22px', borderLeft: '3px solid var(--primary-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <SectionHeader icon="description" title="Generated Reports" subtitle="AI-authored executive analysis based on your dataset" />
            {reportLoading && <span className="spinner-small" />}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {Object.entries(reports).map(([title, content]) => (
              <div key={title} style={{ padding: '16px', backgroundColor: 'rgba(0,0,0,0.15)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{ margin: '0 0 10px', color: 'var(--primary-color)', fontFamily: 'Geist', fontSize: '0.95rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>{title}</h4>
                <ReportView text={content} />
              </div>
            ))}
          </div>
        </GlassCard>
      )}
    </div>
  );
}

// SECTION 2 — DATA PREVIEW
function DataPreviewSection({ data }) {
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState([]);
  const [totalRows, setTotalRows] = useState(0);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const limit = 50;

  const columns = data?.columns ?? [];

  const fetchPage = useCallback(async (p) => {
    setLoading(true);
    try {
      const res = await api.getLogs(p, limit);
      if (res.data?.data) {
        setRows(res.data.data);
        setTotalRows(res.data.total_rows);
      }
    } catch (e) {
      console.error('Preview fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchPage(page); }, [page, fetchPage]);

  const filteredRows = useMemo(() => {
    let r = rows;
    if (search.trim()) {
      const term = search.toLowerCase();
      r = r.filter(row => Object.values(row).some(v => String(v ?? '').toLowerCase().includes(term)));
    }
    if (sortCol) {
      r = [...r].sort((a, b) => {
        const av = a[sortCol] ?? '';
        const bv = b[sortCol] ?? '';
        if (av < bv) return sortDir === 'asc' ? -1 : 1;
        if (av > bv) return sortDir === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return r;
  }, [rows, search, sortCol, sortDir]);

  const toggleSort = (col) => {
    if (sortCol === col) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
  };

  const totalPages = Math.ceil(totalRows / limit);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <SectionHeader icon="table" title="Data Preview" subtitle={`${totalRows.toLocaleString()} rows · ${columns.length} columns · Page ${page} of ${totalPages || 1}`}>
        <div style={{ position: 'relative' }}>
          <span className="material-symbols-outlined" style={{ position:'absolute', left:'10px', top:'50%', transform:'translateY(-50%)', fontSize:'16px', color:'var(--text-muted)', pointerEvents:'none' }}>search</span>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search rows..."
            style={{ backgroundColor:'var(--bg-input)', border:'1px solid var(--border-color)', color:'var(--text-main)', borderRadius:'8px', padding:'8px 12px 8px 32px', fontSize:'0.82rem', outline:'none', width:'220px' }}
          />
        </div>
      </SectionHeader>

      <GlassCard style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '480px' }}>
          {loading ? (
            <div style={{ display:'flex', justifyContent:'center', alignItems:'center', height:'200px' }}>
              <div className="spinner" />
            </div>
          ) : (
            <table style={{ width:'100%', borderCollapse:'collapse', fontSize:'0.8rem' }}>
              <thead>
                <tr style={{ position:'sticky', top:0, backgroundColor:'var(--bg-sidebar)', zIndex:2 }}>
                  {columns.map(col => (
                    <th key={col}
                      onClick={() => toggleSort(col)}
                      style={{
                        padding: '10px 14px', textAlign:'left', fontWeight:600,
                        color: sortCol === col ? 'var(--primary-color)' : 'var(--text-muted)',
                        borderBottom: '1px solid var(--border-color)',
                        whiteSpace:'nowrap', cursor:'pointer', userSelect:'none',
                        fontSize:'0.75rem', textTransform:'uppercase', letterSpacing:'0.05em'
                      }}
                    >
                      <span style={{ display:'flex', alignItems:'center', gap:'4px' }}>
                        {col}
                        {sortCol === col && (
                          <span className="material-symbols-outlined" style={{ fontSize:'12px' }}>{sortDir === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                        )}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredRows.length === 0 ? (
                  <tr><td colSpan={columns.length} style={{ textAlign:'center', padding:'40px', color:'var(--text-muted)' }}>No rows match your search.</td></tr>
                ) : (
                  filteredRows.map((row, i) => (
                    <tr key={i} style={{ backgroundColor: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)', transition:'background 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(59,130,246,0.06)'}
                      onMouseLeave={e => e.currentTarget.style.backgroundColor = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)'}
                    >
                      {columns.map(col => (
                        <td key={col} style={{ padding:'9px 14px', color:'var(--text-main)', borderBottom:'1px solid rgba(255,255,255,0.04)', maxWidth:'200px', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                          {row[col] === null || row[col] === undefined ? (
                            <span style={{ color:'var(--text-muted)', fontStyle:'italic', fontSize:'0.75rem' }}>null</span>
                          ) : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'10px 16px', borderTop:'1px solid var(--border-color)', backgroundColor:'var(--bg-input)' }}>
          <span style={{ fontSize:'0.78rem', color:'var(--text-muted)' }}>
            Showing rows {((page - 1) * limit + 1).toLocaleString()}–{Math.min(page * limit, totalRows).toLocaleString()} of {totalRows.toLocaleString()}
          </span>
          <div style={{ display:'flex', gap:'8px', alignItems:'center' }}>
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} style={{ padding:'5px 12px', borderRadius:'6px', border:'1px solid var(--border-color)', backgroundColor:'var(--bg-card)', color: page === 1 ? 'var(--text-muted)' : 'var(--text-main)', cursor: page === 1 ? 'not-allowed' : 'pointer', fontSize:'0.8rem' }}>← Prev</button>
            <span style={{ fontSize:'0.8rem', color:'var(--text-muted)', minWidth:'80px', textAlign:'center' }}>Page {page} / {totalPages || 1}</span>
            <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages} style={{ padding:'5px 12px', borderRadius:'6px', border:'1px solid var(--border-color)', backgroundColor:'var(--bg-card)', color: page >= totalPages ? 'var(--text-muted)' : 'var(--text-main)', cursor: page >= totalPages ? 'not-allowed' : 'pointer', fontSize:'0.8rem' }}>Next →</button>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}

// SECTION 3 — SCHEMA EXPLORER
function SchemaSection({ data }) {
  const [sortKey, setSortKey] = useState('name');
  const [filterText, setFilterText] = useState('');

  const columnCount = data?.columns?.length ?? 0;
  const confirmedTarget = data?.target_column;

  const schemaRows = useMemo(() => {
    const cols = data?.columns ?? [];
    const dts = data?.dtypes ?? {};
    const miss = data?.missing_values ?? {};
    const sts = data?.statistics ?? {};
    const r = data?.shape?.[0] ?? 0;
    const tCol = data?.target_column;
    let items = cols.map(col => {
      const missingCount = miss[col] ?? 0;
      const missingPct = r > 0 ? (missingCount / r) * 100 : 0;
      const colStats = sts[col] ?? {};
      const uniqueCount = colStats['count'] ? Math.round(colStats['count']) : null;
      return { col, dtype: dts[col] ?? '—', missingCount, missingPct, uniqueCount, isTarget: col === tCol };
    });

    if (filterText) {
      const term = filterText.toLowerCase();
      items = items.filter(row => row.col.toLowerCase().includes(term) || row.dtype.toLowerCase().includes(term));
    }

    if (sortKey === 'missing') items.sort((a, b) => b.missingPct - a.missingPct);
    else if (sortKey === 'dtype') items.sort((a, b) => a.dtype.localeCompare(b.dtype));
    else items.sort((a, b) => a.col.localeCompare(b.col));

    return items;
  }, [data, sortKey, filterText]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <SectionHeader icon="schema" title="Schema Explorer" subtitle={`${columnCount} columns · Target: ${confirmedTarget ?? 'Not confirmed'}`}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <input type="text" value={filterText} onChange={e => setFilterText(e.target.value)} placeholder="Filter columns..." style={{ backgroundColor:'var(--bg-input)', border:'1px solid var(--border-color)', color:'var(--text-main)', borderRadius:'6px', padding:'7px 10px', fontSize:'0.8rem', outline:'none', width:'160px' }} />
          <select value={sortKey} onChange={e => setSortKey(e.target.value)} style={{ backgroundColor:'var(--bg-input)', border:'1px solid var(--border-color)', color:'var(--text-main)', borderRadius:'6px', padding:'7px 10px', fontSize:'0.8rem', outline:'none' }}>
            <option value="name">Sort: Name</option>
            <option value="missing">Sort: Missing %</option>
            <option value="dtype">Sort: Type</option>
          </select>
        </div>
      </SectionHeader>

      <GlassCard style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:'0.82rem' }}>
            <thead>
              <tr style={{ backgroundColor:'var(--bg-sidebar)' }}>
                {['Column Name','Data Type','Missing','Missing %','Notes'].map(h => (
                  <th key={h} style={{ padding:'10px 16px', textAlign:'left', color:'var(--text-muted)', fontWeight:600, fontSize:'0.72rem', textTransform:'uppercase', letterSpacing:'0.06em', borderBottom:'1px solid var(--border-color)', whiteSpace:'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {schemaRows.map(({ col, dtype, missingCount, missingPct, isTarget }, i) => {
                const badge = getMissingBadgeStyle(missingPct);
                return (
                  <tr key={col} style={{ borderBottom:'1px solid rgba(255,255,255,0.04)', transition:'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(59,130,246,0.05)'}
                    onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
                  >
                    <td style={{ padding:'10px 16px', fontWeight:600, color:'var(--text-main)', display:'flex', alignItems:'center', gap:'8px' }}>
                      {col}
                      {isTarget && <span style={{ fontSize:'10px', fontWeight:700, color:'var(--secondary-color)', backgroundColor:'rgba(139,92,246,0.1)', border:'1px solid rgba(139,92,246,0.3)', borderRadius:'4px', padding:'1px 6px' }}>TARGET</span>}
                    </td>
                    <td style={{ padding:'10px 16px' }}>
                      <span style={{ fontSize:'11px', fontWeight:600, color: getDtypeBadgeColor(dtype), backgroundColor:`${getDtypeBadgeColor(dtype)}18`, border:`1px solid ${getDtypeBadgeColor(dtype)}40`, borderRadius:'4px', padding:'2px 8px' }}>{dtype}</span>
                    </td>
                    <td style={{ padding:'10px 16px', color:'var(--text-muted)' }}>{missingCount.toLocaleString()}</td>
                    <td style={{ padding:'10px 16px' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
                        <div style={{ width:'80px', height:'5px', backgroundColor:'var(--bg-input)', borderRadius:'3px', overflow:'hidden' }}>
                          <div style={{ width:`${Math.min(missingPct, 100)}%`, height:'100%', backgroundColor: badge.color, borderRadius:'3px' }} />
                        </div>
                        <span style={{ fontSize:'0.75rem', fontWeight:600, color: badge.color }}>{missingPct.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td style={{ padding:'10px 16px', color:'var(--text-muted)', fontSize:'0.75rem' }}>
                      {missingPct === 0 ? '✓ Complete' : missingPct < 5 ? 'Minor gaps' : missingPct < 20 ? 'Moderate gaps' : '⚠ High missing'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
}

// SECTION 4 — DATA QUALITY
function DataQualitySection({ data }) {
  const rows = data?.shape?.[0] ?? 0;
  const cols = data?.shape?.[1] ?? 0;
  const missing = data?.missing_values ?? {};
  const outliers = data?.outliers ?? {};
  const duplicates = data?.duplicate_rows ?? 0;
  const missingTotal = Object.values(missing).reduce((a, b) => a + b, 0);
  const outliersTotal = Object.values(outliers).reduce((a, b) => a + b, 0);
  const completeness = rows > 0 && cols > 0 ? (((rows * cols - missingTotal) / (rows * cols)) * 100).toFixed(1) : '—';

  const missingCols = Object.entries(missing).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const maxMissing = missingCols.length > 0 ? missingCols[0][1] : 1;

  const outlierCols = Object.entries(outliers).sort((a, b) => b[1] - a[1]);
  const maxOutlier = outlierCols.length > 0 ? outlierCols[0][1] : 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <SectionHeader icon="health_metrics" title="Data Quality Center" subtitle="All metrics derived from your uploaded dataset via IQR outlier detection and null analysis" />

      {/* Quality KPI strip */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(180px, 1fr))', gap:'14px' }}>
        <GlassCard style={{ padding:'16px' }}>
          <div style={{ fontSize:'0.7rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Completeness</div>
          <div style={{ fontSize:'1.6rem', fontWeight:700, fontFamily:'Geist', color: parseFloat(completeness) >= 95 ? 'var(--accent-color)' : parseFloat(completeness) >= 80 ? 'var(--warning-color)' : 'var(--danger-color)' }}>{completeness}%</div>
          <div style={{ height:'4px', backgroundColor:'var(--bg-input)', borderRadius:'2px', marginTop:'8px', overflow:'hidden' }}>
            <div style={{ width:`${completeness}%`, height:'100%', backgroundColor: parseFloat(completeness) >= 95 ? 'var(--accent-color)' : parseFloat(completeness) >= 80 ? 'var(--warning-color)' : 'var(--danger-color)' }} />
          </div>
        </GlassCard>
        <GlassCard style={{ padding:'16px' }}>
          <div style={{ fontSize:'0.7rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Total Missing Cells</div>
          <div style={{ fontSize:'1.6rem', fontWeight:700, fontFamily:'Geist', color: missingTotal > 0 ? 'var(--warning-color)' : 'var(--accent-color)' }}>{missingTotal.toLocaleString()}</div>
          <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:'4px' }}>{missingCols.length} columns affected</div>
        </GlassCard>
        <GlassCard style={{ padding:'16px' }}>
          <div style={{ fontSize:'0.7rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Outlier Cells</div>
          <div style={{ fontSize:'1.6rem', fontWeight:700, fontFamily:'Geist', color: outliersTotal > 0 ? 'var(--warning-color)' : 'var(--accent-color)' }}>{outliersTotal.toLocaleString()}</div>
          <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:'4px' }}>{outlierCols.length} columns with outliers</div>
        </GlassCard>
        <GlassCard style={{ padding:'16px' }}>
          <div style={{ fontSize:'0.7rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Duplicate Rows</div>
          <div style={{ fontSize:'1.6rem', fontWeight:700, fontFamily:'Geist', color: duplicates > 0 ? 'var(--warning-color)' : 'var(--accent-color)' }}>{duplicates.toLocaleString()}</div>
          <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:'4px' }}>{rows > 0 ? `${((duplicates / rows) * 100).toFixed(2)}% of total` : '—'}</div>
        </GlassCard>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'20px' }} className="grid-cols-2-responsive">
        {/* Missing Values Per Column */}
        <GlassCard style={{ padding:'20px' }}>
          <h3 style={{ margin:'0 0 16px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)', display:'flex', alignItems:'center', gap:'6px' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px', color:'var(--warning-color)' }}>data_alert</span>
            Missing Values by Column
          </h3>
          {missingCols.length === 0 ? (
            <div style={{ textAlign:'center', padding:'30px 0', color:'var(--accent-color)' }}>
              <span className="material-symbols-outlined" style={{ fontSize:'32px', marginBottom:'8px', display:'block' }}>check_circle</span>
              <p style={{ margin:0, fontSize:'0.85rem' }}>No missing values detected</p>
            </div>
          ) : (
            <div style={{ display:'flex', flexDirection:'column', gap:'10px', maxHeight:'300px', overflowY:'auto' }}>
              {missingCols.map(([col, count]) => {
                const pct = rows > 0 ? (count / rows) * 100 : 0;
                const badge = getMissingBadgeStyle(pct);
                return (
                  <div key={col}>
                    <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'4px' }}>
                      <span style={{ fontSize:'0.78rem', color:'var(--text-main)', fontWeight:500 }}>{col}</span>
                      <span style={{ fontSize:'0.75rem', fontWeight:700, color:badge.color }}>{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
                    </div>
                    <div style={{ height:'5px', backgroundColor:'var(--bg-input)', borderRadius:'3px', overflow:'hidden' }}>
                      <div style={{ width:`${(count / maxMissing) * 100}%`, height:'100%', backgroundColor:badge.color, borderRadius:'3px', transition:'width 0.5s ease' }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </GlassCard>

        {/* Outliers Per Column */}
        <GlassCard style={{ padding:'20px' }}>
          <h3 style={{ margin:'0 0 16px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)', display:'flex', alignItems:'center', gap:'6px' }}>
            <span className="material-symbols-outlined" style={{ fontSize:'18px', color:'var(--warning-color)' }}>scatter_plot</span>
            Outliers by Column (IQR Method)
          </h3>
          {outlierCols.length === 0 ? (
            <div style={{ textAlign:'center', padding:'30px 0', color:'var(--accent-color)' }}>
              <span className="material-symbols-outlined" style={{ fontSize:'32px', marginBottom:'8px', display:'block' }}>verified</span>
              <p style={{ margin:0, fontSize:'0.85rem' }}>No outliers detected</p>
            </div>
          ) : (
            <div style={{ display:'flex', flexDirection:'column', gap:'10px', maxHeight:'300px', overflowY:'auto' }}>
              {outlierCols.map(([col, count]) => {
                const pct = rows > 0 ? (count / rows) * 100 : 0;
                return (
                  <div key={col}>
                    <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'4px' }}>
                      <span style={{ fontSize:'0.78rem', color:'var(--text-main)', fontWeight:500 }}>{col}</span>
                      <span style={{ fontSize:'0.75rem', fontWeight:700, color:'var(--warning-color)' }}>{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
                    </div>
                    <div style={{ height:'5px', backgroundColor:'var(--bg-input)', borderRadius:'3px', overflow:'hidden' }}>
                      <div style={{ width:`${(count / maxOutlier) * 100}%`, height:'100%', backgroundColor:'var(--warning-color)', borderRadius:'3px', transition:'width 0.5s ease' }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}

// SECTION 5 — CORRELATIONS
function CorrelationsSection({ data, plots }) {
  const numeric = data?.numeric_cols ?? [];
  const correlations = data?.correlations ?? {};
  const targetCol = data?.target_column;
  const heatmapBase64 = plots?.heatmap;

  const topPairs = useMemo(() => {
    const num = data?.numeric_cols ?? [];
    const corrs = data?.correlations ?? {};
    const pairs = [];
    num.forEach(colA => {
      num.forEach(colB => {
        if (colA >= colB) return;
        const val = corrs[colA]?.[colB];
        if (val !== null && val !== undefined && !isNaN(val)) {
          pairs.push({ colA, colB, val });
        }
      });
    });
    return pairs.sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, 8);
  }, [data]);

  const limitedCols = numeric.slice(0, 10); // Cap matrix at 10×10 for readability

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'20px' }}>
      <SectionHeader icon="bubble_chart" title="Correlation Explorer" subtitle={`Pearson correlations across ${numeric.length} numeric columns`} />

      {numeric.length < 2 ? (
        <GlassCard style={{ padding:'40px', textAlign:'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize:'40px', color:'var(--text-muted)', marginBottom:'12px', display:'block' }}>info</span>
          <p style={{ color:'var(--text-muted)', margin:0 }}>At least 2 numeric columns are required to compute correlations.</p>
        </GlassCard>
      ) : (
        <div style={{ display:'flex', flexDirection:'column', gap:'20px' }}>
          {/* Backend heatmap (if available) */}
          {heatmapBase64 && (
            <GlassCard style={{ padding:'20px' }}>
              <h3 style={{ margin:'0 0 14px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)', display:'flex', alignItems:'center', gap:'6px' }}>
                <span className="material-symbols-outlined" style={{ fontSize:'18px', color:'var(--primary-color)' }}>grid_on</span>
                Correlation Heatmap
                <span style={{ fontSize:'10px', color:'var(--text-muted)', marginLeft:'auto', fontWeight:400 }}>Generated by backend · Seaborn</span>
              </h3>
              <div style={{ display:'flex', justifyContent:'center', borderRadius:'8px', overflow:'hidden' }}>
                <img src={`data:image/png;base64,${heatmapBase64}`} alt="Correlation Heatmap" style={{ maxWidth:'100%', borderRadius:'6px' }} />
              </div>
            </GlassCard>
          )}

          {/* CSS Correlation Matrix */}
          <GlassCard style={{ padding:'20px', overflowX:'auto' }}>
            <h3 style={{ margin:'0 0 14px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)', display:'flex', alignItems:'center', gap:'6px' }}>
              <span className="material-symbols-outlined" style={{ fontSize:'18px', color:'var(--secondary-color)' }}>table_chart</span>
              Correlation Matrix
              {numeric.length > 10 && <span style={{ fontSize:'11px', color:'var(--text-muted)', marginLeft:'8px' }}>Showing top 10 columns</span>}
            </h3>
            <div style={{ overflowX:'auto' }}>
              <table style={{ borderCollapse:'collapse', fontSize:'0.72rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding:'6px 10px', color:'var(--text-muted)', fontWeight:600, textAlign:'left', minWidth:'100px' }}></th>
                    {limitedCols.map(col => (
                      <th key={col} style={{ padding:'6px 8px', color: col === targetCol ? 'var(--secondary-color)' : 'var(--text-muted)', fontWeight:600, whiteSpace:'nowrap', maxWidth:'80px', overflow:'hidden', textOverflow:'ellipsis', fontSize:'0.68rem' }}>
                        {col.length > 10 ? col.slice(0, 9) + '…' : col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {limitedCols.map(rowCol => (
                    <tr key={rowCol}>
                      <td style={{ padding:'5px 10px', color: rowCol === targetCol ? 'var(--secondary-color)' : 'var(--text-muted)', fontWeight:600, whiteSpace:'nowrap', fontSize:'0.72rem' }}>
                        {rowCol.length > 12 ? rowCol.slice(0, 11) + '…' : rowCol}
                      </td>
                      {limitedCols.map(colCol => {
                        const val = correlations[rowCol]?.[colCol];
                        const isDiag = rowCol === colCol;
                        return (
                          <td key={colCol} title={val != null ? `${rowCol} × ${colCol}: ${(val).toFixed(3)}` : 'N/A'} style={{
                            padding:'5px 8px',
                            backgroundColor: isDiag ? 'rgba(59,130,246,0.15)' : getCorrelationColor(val),
                            color:'white',
                            textAlign:'center',
                            fontWeight: isDiag ? 700 : 500,
                            fontSize:'0.7rem',
                            minWidth:'56px',
                            borderRadius:'2px',
                            cursor:'default'
                          }}>
                            {isDiag ? '1.00' : (val != null ? val.toFixed(2) : '—')}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ display:'flex', gap:'16px', marginTop:'12px', fontSize:'0.72rem', color:'var(--text-muted)' }}>
              <span style={{ display:'flex', alignItems:'center', gap:'4px' }}><span style={{ display:'inline-block', width:'12px', height:'12px', backgroundColor:'rgba(59,130,246,0.7)', borderRadius:'2px' }} />Strong positive</span>
              <span style={{ display:'flex', alignItems:'center', gap:'4px' }}><span style={{ display:'inline-block', width:'12px', height:'12px', backgroundColor:'rgba(239,68,68,0.7)', borderRadius:'2px' }} />Strong negative</span>
              <span style={{ display:'flex', alignItems:'center', gap:'4px' }}><span style={{ display:'inline-block', width:'12px', height:'12px', backgroundColor:'var(--bg-input)', borderRadius:'2px', border:'1px solid var(--border-color)' }} />Near zero</span>
            </div>
          </GlassCard>

          {/* Top Correlated Pairs */}
          <GlassCard style={{ padding:'20px' }}>
            <h3 style={{ margin:'0 0 14px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)' }}>Top Correlated Feature Pairs</h3>
            <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
              {topPairs.map(({ colA, colB, val }) => (
                <div key={`${colA}-${colB}`} style={{ display:'flex', alignItems:'center', gap:'12px' }}>
                  <div style={{ flex:1, fontSize:'0.8rem', color:'var(--text-main)' }}>
                    <span style={{ color:'var(--primary-color)', fontWeight:600 }}>{colA}</span>
                    <span style={{ color:'var(--text-muted)', margin:'0 6px' }}>↔</span>
                    <span style={{ color:'var(--primary-color)', fontWeight:600 }}>{colB}</span>
                  </div>
                  <div style={{ width:'120px', height:'5px', backgroundColor:'var(--bg-input)', borderRadius:'3px', overflow:'hidden' }}>
                    <div style={{ width:`${Math.abs(val) * 100}%`, height:'100%', backgroundColor: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)', borderRadius:'3px' }} />
                  </div>
                  <div style={{ width:'50px', textAlign:'right', fontSize:'0.82rem', fontWeight:700, fontFamily:'Geist', color: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)' }}>
                    {val > 0 ? '+' : ''}{val.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  );
}

// SECTION 6 — DISTRIBUTIONS
function DistributionsSection({ data, plots }) {
  const allCols = data?.columns ?? [];
  const numericCols = data?.numeric_cols ?? [];
  const stats = data?.statistics ?? {};
  const distributions = data?.distributions ?? {};

  const [selectedCol, setSelectedCol] = useState(allCols[0] ?? null);

  useEffect(() => {
    const cols = data?.columns ?? [];
    if (!selectedCol && cols.length > 0) setSelectedCol(cols[0]);
  }, [data, selectedCol]);

  const isNumeric = numericCols.includes(selectedCol);
  const colStats = stats[selectedCol] ?? {};
  const colDist = distributions[selectedCol] ?? {};
  const base64Key = `dist_${selectedCol}`;
  const hasBase64 = plots && plots[base64Key];

  // SVG histogram approximation from quartile stats
  const svgHistogram = useMemo(() => {
    const numCols = data?.numeric_cols ?? [];
    const sts = data?.statistics ?? {};
    if (!selectedCol || !numCols.includes(selectedCol)) return null;
    const cs = sts[selectedCol] ?? {};
    const min = cs['min'];
    const max = cs['max'];
    const q1 = cs['25%'];
    const q2 = cs['50%'];
    const q3 = cs['75%'];
    const mean = cs['mean'];
    if (min == null || max == null) return null;
    const buckets = [
      { label: `${min?.toFixed(1)}`, approxHeight: 30 },
      { label: `${q1?.toFixed(1)}`, approxHeight: 70 },
      { label: `${q2?.toFixed(1)} (median)`, approxHeight: 100 },
      { label: `${q3?.toFixed(1)}`, approxHeight: 70 },
      { label: `${max?.toFixed(1)}`, approxHeight: 30 },
    ];
    return { buckets, min, max, q1, q2, q3, mean };
  }, [selectedCol, data]);

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'20px' }}>
      <SectionHeader icon="bar_chart_4_bars" title="Distribution Analysis" subtitle="Select a column to explore its statistical distribution">
        <select value={selectedCol || ''} onChange={e => setSelectedCol(e.target.value)} style={{ backgroundColor:'var(--bg-input)', border:'1px solid var(--border-color)', color:'var(--text-main)', borderRadius:'6px', padding:'7px 12px', fontSize:'0.85rem', outline:'none', minWidth:'180px' }}>
          {allCols.map(col => <option key={col} value={col}>{col} {numericCols.includes(col) ? '(numeric)' : '(categorical)'}</option>)}
        </select>
      </SectionHeader>

      {selectedCol && (
        <div style={{ display:'flex', flexDirection:'column', gap:'16px' }}>
          {/* Distribution Visualization */}
          <GlassCard style={{ padding:'20px' }}>
            <h3 style={{ margin:'0 0 16px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)' }}>
              Distribution of: <span style={{ color:'var(--primary-color)' }}>{selectedCol}</span>
              <span style={{ marginLeft:'10px', fontSize:'11px', fontWeight:400, color:'var(--text-muted)', backgroundColor:'var(--bg-input)', padding:'2px 8px', borderRadius:'4px' }}>{isNumeric ? 'Numeric' : 'Categorical'}</span>
            </h3>

            {hasBase64 ? (
              <div style={{ display:'flex', justifyContent:'center' }}>
                <img src={`data:image/png;base64,${plots[base64Key]}`} alt={`Distribution of ${selectedCol}`} style={{ maxWidth:'100%', borderRadius:'6px' }} />
              </div>
            ) : isNumeric && svgHistogram ? (
              <div>
                <div style={{ display:'flex', alignItems:'flex-end', gap:'4px', height:'120px', padding:'0 8px' }}>
                  {svgHistogram.buckets.map((b, i) => (
                    <div key={i} style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', gap:'4px' }}>
                      <div style={{ width:'100%', height:`${b.approxHeight}%`, backgroundColor: i === 2 ? 'var(--primary-color)' : 'rgba(59,130,246,0.35)', borderRadius:'3px 3px 0 0', transition:'height 0.4s ease' }} />
                    </div>
                  ))}
                </div>
                <div style={{ display:'flex', justifyContent:'space-between', padding:'4px 8px', fontSize:'0.68rem', color:'var(--text-muted)' }}>
                  <span>{svgHistogram.min?.toFixed(2)}</span>
                  <span>Q1: {svgHistogram.q1?.toFixed(2)}</span>
                  <span>Median: {svgHistogram.q2?.toFixed(2)}</span>
                  <span>Q3: {svgHistogram.q3?.toFixed(2)}</span>
                  <span>{svgHistogram.max?.toFixed(2)}</span>
                </div>
                <p style={{ textAlign:'center', fontSize:'0.72rem', color:'var(--text-muted)', margin:'8px 0 0', fontStyle:'italic' }}>Quartile approximation — upload a dataset to trigger full histogram generation for top columns</p>
              </div>
            ) : !isNumeric ? (
              <div style={{ display:'flex', flexDirection:'column', gap:'8px', maxHeight:'280px', overflowY:'auto' }}>
                {Object.entries(colDist).length === 0 ? (
                  <p style={{ color:'var(--text-muted)', textAlign:'center', padding:'24px' }}>No distribution data available.</p>
                ) : (
                  (() => {
                    const max = Math.max(...Object.values(colDist));
                    return Object.entries(colDist).map(([val, count]) => (
                      <div key={val} style={{ display:'flex', alignItems:'center', gap:'10px' }}>
                        <div style={{ width:'130px', fontSize:'0.8rem', color:'var(--text-main)', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{String(val)}</div>
                        <div style={{ flex:1, height:'18px', backgroundColor:'var(--bg-input)', borderRadius:'3px', overflow:'hidden' }}>
                          <div style={{ width:`${(count / max) * 100}%`, height:'100%', backgroundColor:'var(--secondary-color)', borderRadius:'3px', opacity:0.8 }} />
                        </div>
                        <div style={{ width:'60px', textAlign:'right', fontSize:'0.78rem', color:'var(--text-muted)' }}>{count.toLocaleString()}</div>
                      </div>
                    ));
                  })()
                )}
              </div>
            ) : (
              <div style={{ textAlign:'center', padding:'40px', color:'var(--text-muted)' }}>No distribution data available for this column.</div>
            )}
          </GlassCard>

          {/* Summary Statistics */}
          {isNumeric && Object.keys(colStats).length > 0 && (
            <GlassCard style={{ padding:'20px' }}>
              <h3 style={{ margin:'0 0 14px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)' }}>Summary Statistics</h3>
              <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(130px, 1fr))', gap:'12px' }}>
                {[
                  { label:'Count', key:'count' },
                  { label:'Mean', key:'mean' },
                  { label:'Std Dev', key:'std' },
                  { label:'Min', key:'min' },
                  { label:'25th %ile', key:'25%' },
                  { label:'Median', key:'50%' },
                  { label:'75th %ile', key:'75%' },
                  { label:'Max', key:'max' },
                ].map(({ label, key }) => {
                  const val = colStats[key];
                  return val != null ? (
                    <div key={key} style={{ padding:'12px', backgroundColor:'var(--bg-input)', borderRadius:'8px', border:'1px solid var(--border-color)' }}>
                      <div style={{ fontSize:'0.68rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'4px' }}>{label}</div>
                      <div style={{ fontSize:'1rem', fontWeight:700, color:'var(--text-main)', fontFamily:'Geist' }}>
                        {typeof val === 'number' ? (Number.isInteger(val) ? val.toLocaleString() : val.toFixed(3)) : '—'}
                      </div>
                    </div>
                  ) : null;
                })}
              </div>
            </GlassCard>
          )}
        </div>
      )}
    </div>
  );
}

// SECTION 7 — AI INSIGHTS
function AIInsightsSection({ data, domainProfile, reports, reportLoading, runAnalysis }) {
  const targetCol = data?.target_column || domainProfile?.target_column;
  const hasReports = Object.keys(reports || {}).length > 0;

  const targetCorrelations = useMemo(() => {
    const tCol = data?.target_column || domainProfile?.target_column;
    const corrs = data?.correlations ?? {};
    if (!tCol || !corrs[tCol]) return [];
    return Object.entries(corrs[tCol])
      .filter(([col]) => col !== tCol)
      .map(([col, val]) => ({ col, val }))
      .filter(x => x.val !== null && !isNaN(x.val))
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 10);
  }, [data, domainProfile]);

  const ReportView = ({ text }) => {
    if (!text || text === 'Analyzing...') {
      return (
        <div style={{ display:'flex', alignItems:'center', gap:'10px', color:'var(--text-muted)', fontSize:'0.85rem', padding:'12px 0' }}>
          <div className="spinner-small" />
          <span>Analysis running in background...</span>
        </div>
      );
    }
    return (
      <div style={{ fontSize:'0.875rem', lineHeight:1.65, color:'var(--text-main)' }}>
        {text.split('\n').map((line, i) => {
          if (line.startsWith('###')) return <h4 key={i} style={{ color:'var(--primary-color)', fontSize:'0.9rem', margin:'14px 0 4px', fontFamily:'Geist' }}>{line.replace(/^###\s*/,'')}</h4>;
          if (line.startsWith('**') && line.endsWith('**')) return <strong key={i} style={{ display:'block', marginTop:'8px', color:'var(--text-main)' }}>{line.replace(/\*\*/g,'')}</strong>;
          if (!line.trim()) return <br key={i} />;
          return <p key={i} style={{ margin:'2px 0', color:'var(--text-muted)' }}>{line}</p>;
        })}
      </div>
    );
  };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'20px' }}>
      <SectionHeader icon="auto_awesome" title="AI Insights" subtitle={`Domain: ${domainProfile?.domain ?? 'Detecting...'} · Analysis: ${domainProfile?.analysis_type ?? '—'}`}>
        <div style={{ display:'flex', gap:'8px' }}>
          <button onClick={() => runAnalysis('what')} disabled={reportLoading} style={{ padding:'8px 14px', backgroundColor:'var(--primary-color)', color:'white', border:'none', borderRadius:'6px', fontWeight:600, fontSize:'0.8rem', cursor:'pointer', opacity: reportLoading ? 0.6 : 1 }}>
            {reportLoading ? 'Running...' : '⚡ Driver Scan'}
          </button>
          <button onClick={() => runAnalysis('why')} disabled={reportLoading} style={{ padding:'8px 14px', backgroundColor:'var(--secondary-color)', color:'white', border:'none', borderRadius:'6px', fontWeight:600, fontSize:'0.8rem', cursor:'pointer', opacity: reportLoading ? 0.6 : 1 }}>
            🧠 AI Report
          </button>
        </div>
      </SectionHeader>

      {/* Domain + Status Card */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'16px' }} className="grid-cols-2-responsive">
        <GlassCard style={{ padding:'18px' }}>
          <div style={{ fontSize:'0.72rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Detected Domain</div>
          <div style={{ fontSize:'1.1rem', fontWeight:700, fontFamily:'Geist', color:'var(--text-main)' }}>{domainProfile?.domain ?? 'Analyzing dataset...'}</div>
          {domainProfile?.confidence && <div style={{ fontSize:'0.75rem', color:'var(--accent-color)', marginTop:'4px' }}>Confidence: {(domainProfile.confidence * 100).toFixed(0)}%</div>}
        </GlassCard>
        <GlassCard style={{ padding:'18px' }}>
          <div style={{ fontSize:'0.72rem', textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:'8px' }}>Target Variable</div>
          <div style={{ fontSize:'1.1rem', fontWeight:700, fontFamily:'Geist', color:'var(--secondary-color)' }}>{targetCol ?? 'Not confirmed'}</div>
          <div style={{ fontSize:'0.75rem', color:'var(--text-muted)', marginTop:'4px' }}>{domainProfile?.analysis_type ? `${domainProfile.analysis_type.charAt(0).toUpperCase() + domainProfile.analysis_type.slice(1)} problem` : '—'}</div>
        </GlassCard>
      </div>

      {/* Target Correlations Ranked */}
      {targetCorrelations.length > 0 && (
        <GlassCard style={{ padding:'20px' }}>
          <h3 style={{ margin:'0 0 16px', fontSize:'0.95rem', fontFamily:'Geist', color:'var(--text-main)' }}>
            Top Predictors of <span style={{ color:'var(--secondary-color)' }}>{targetCol}</span>
          </h3>
          <div style={{ display:'flex', flexDirection:'column', gap:'12px' }}>
            {targetCorrelations.map(({ col, val }, idx) => {
              const strength = Math.abs(val) > 0.7 ? 'Strong' : Math.abs(val) > 0.4 ? 'Moderate' : 'Weak';
              const dir = val > 0 ? 'positive' : 'negative';
              return (
                <div key={col} style={{ display:'flex', alignItems:'center', gap:'12px', padding:'10px 14px', backgroundColor:'var(--bg-input)', borderRadius:'8px', border:'1px solid var(--border-color)' }}>
                  <div style={{ width:'22px', height:'22px', borderRadius:'50%', backgroundColor:'rgba(59,130,246,0.15)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.7rem', fontWeight:700, color:'var(--primary-color)', flexShrink:0 }}>{idx + 1}</div>
                  <div style={{ flex:1 }}>
                    <div style={{ fontSize:'0.85rem', fontWeight:600, color:'var(--text-main)' }}>{col}</div>
                    <div style={{ fontSize:'0.72rem', color:'var(--text-muted)' }}>{strength} {dir} correlation</div>
                  </div>
                  <div style={{ width:'80px', height:'5px', backgroundColor:'var(--bg-sidebar)', borderRadius:'3px', overflow:'hidden' }}>
                    <div style={{ width:`${Math.abs(val) * 100}%`, height:'100%', backgroundColor: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)', borderRadius:'3px' }} />
                  </div>
                  <div style={{ width:'52px', textAlign:'right', fontSize:'0.82rem', fontWeight:700, color: val > 0 ? 'var(--primary-color)' : 'var(--danger-color)' }}>
                    {val > 0 ? '+' : ''}{val.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>
        </GlassCard>
      )}

      {/* Generated Reports */}
      {hasReports ? (
        <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
          {Object.entries(reports).map(([title, content]) => (
            <GlassCard key={title} style={{ padding:'20px', borderLeft:'3px solid var(--primary-color)' }}>
              <h3 style={{ margin:'0 0 12px', fontSize:'0.95rem', color:'var(--primary-color)', fontFamily:'Geist', borderBottom:'1px solid var(--border-color)', paddingBottom:'10px' }}>{title}</h3>
              <ReportView text={content} />
            </GlassCard>
          ))}
        </div>
      ) : (
        <GlassCard style={{ padding:'32px', textAlign:'center', border:'1px dashed var(--border-color)' }}>
          <span className="material-symbols-outlined" style={{ fontSize:'36px', color:'var(--text-muted)', marginBottom:'12px', display:'block' }}>auto_awesome</span>
          <p style={{ color:'var(--text-muted)', margin:'0 0 16px', fontSize:'0.9rem' }}>No AI reports generated yet. Run a Driver Scan or AI Report to get started.</p>
          <div style={{ display:'flex', gap:'10px', justifyContent:'center' }}>
            <button onClick={() => runAnalysis('what')} style={{ padding:'9px 18px', backgroundColor:'var(--primary-color)', color:'white', border:'none', borderRadius:'6px', fontWeight:600, cursor:'pointer', fontSize:'0.85rem' }}>⚡ Run Driver Scan</button>
            <button onClick={() => runAnalysis('why')} style={{ padding:'9px 18px', backgroundColor:'var(--secondary-color)', color:'white', border:'none', borderRadius:'6px', fontWeight:600, cursor:'pointer', fontSize:'0.85rem' }}>🧠 Generate AI Report</button>
          </div>
        </GlassCard>
      )}
    </div>
  );
}

// ── Main Export ───────────────────────────────────────────────────────────────

export default function AnalysisWorkspace({
  data,
  domainProfile,
  plots,
  runAnalysis,
  loadFailures,
  reports,
  reportLoading,
  downloadPDF,
  onNavigateToTab,
}) {
  const [activeTab, setActiveTab] = useState('overview');

  const profileStatus = domainProfile?.status ?? 'idle';

  const tabVariants = {
    enter: { opacity: 0, y: 8 },
    center: { opacity: 1, y: 0, transition: { duration: 0.22, ease: 'easeOut' } },
    exit: { opacity: 0, y: -8, transition: { duration: 0.15 } },
  };

  return (
    <FadeIn>
      <div style={{ display:'flex', flexDirection:'column', height:'100%', gap: 0 }}>

        {/* ── Dataset Context Identity Bar ─────────────────────────────── */}
        <div style={{
          padding: '14px 0 0',
          marginBottom: '0',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}>
          {/* Identity Row */}
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:'12px' }}>
            <div style={{ display:'flex', alignItems:'center', gap:'12px', flexWrap:'wrap' }}>
              <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
                <div style={{ width:'36px', height:'36px', borderRadius:'8px', backgroundColor:'rgba(59,130,246,0.12)', border:'1px solid rgba(59,130,246,0.3)', display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <span className="material-symbols-outlined" style={{ fontSize:'20px', color:'var(--primary-color)', fontVariationSettings:"'FILL' 1" }}>table_chart</span>
                </div>
                <div>
                  <div style={{ fontSize:'1.1rem', fontWeight:700, color:'var(--text-main)', fontFamily:'Geist' }}>{data?.filename ?? 'Unknown Dataset'}</div>
                  <div style={{ fontSize:'0.75rem', color:'var(--text-muted)' }}>
                    {(data?.shape?.[0] ?? 0).toLocaleString()} rows · {data?.shape?.[1] ?? 0} columns · {formatFileSize(data?.file_size_bytes)}
                  </div>
                </div>
              </div>
              {data?.target_column && (
                <div style={{ display:'flex', alignItems:'center', gap:'6px', backgroundColor:'rgba(139,92,246,0.1)', border:'1px solid rgba(139,92,246,0.3)', borderRadius:'20px', padding:'4px 12px', fontSize:'0.78rem' }}>
                  <span className="material-symbols-outlined" style={{ fontSize:'14px', color:'var(--secondary-color)' }}>target</span>
                  <span style={{ color:'var(--secondary-color)', fontWeight:600 }}>Target: {data.target_column}</span>
                </div>
              )}
              {domainProfile?.domain && (
                <div style={{ display:'flex', alignItems:'center', gap:'6px', backgroundColor:'rgba(16,185,129,0.08)', border:'1px solid rgba(16,185,129,0.25)', borderRadius:'20px', padding:'4px 12px', fontSize:'0.78rem' }}>
                  <span className="material-symbols-outlined" style={{ fontSize:'14px', color:'var(--accent-color)' }}>language</span>
                  <span style={{ color:'var(--accent-color)', fontWeight:500 }}>{domainProfile.domain}</span>
                </div>
              )}
            </div>
            <StatusBadge status={profileStatus === 'completed' ? 'COMPLETED' : profileStatus === 'running' ? 'ACTIVE' : 'IDLE'} />
          </div>

          {/* Tab Navigation */}
          <div style={{ display:'flex', gap:'2px', borderBottom:'1px solid var(--border-color)', paddingBottom:'0' }}>
            {WORKSPACE_TABS.map(tab => {
              const isActive = activeTab === tab.id;
              return (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                  display:'flex', alignItems:'center', gap:'6px',
                  padding:'9px 14px',
                  background:'transparent', border:'none',
                  color: isActive ? 'var(--primary-color)' : 'var(--text-muted)',
                  fontWeight: isActive ? 600 : 400,
                  fontSize:'0.82rem',
                  cursor:'pointer',
                  borderBottom: isActive ? '2px solid var(--primary-color)' : '2px solid transparent',
                  marginBottom:'-1px',
                  transition:'all 0.15s ease',
                  whiteSpace:'nowrap',
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize:'16px', fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}>{tab.icon}</span>
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Tab Content ────────────────────────────────────────────────── */}
        <div style={{ flex:1, overflowY:'auto', paddingTop:'20px', paddingRight:'2px' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              variants={tabVariants}
              initial="enter"
              animate="center"
              exit="exit"
            >
              {activeTab === 'overview' && (
                <OverviewSection
                  data={data} domainProfile={domainProfile}
                  reports={reports} reportLoading={reportLoading}
                  runAnalysis={runAnalysis} loadFailures={loadFailures}
                  downloadPDF={downloadPDF} onNavigateToTab={onNavigateToTab}
                />
              )}
              {activeTab === 'preview' && <DataPreviewSection data={data} />}
              {activeTab === 'schema' && <SchemaSection data={data} />}
              {activeTab === 'quality' && <DataQualitySection data={data} />}
              {activeTab === 'correlations' && <CorrelationsSection data={data} plots={plots} />}
              {activeTab === 'distributions' && <DistributionsSection data={data} plots={plots} />}
              {activeTab === 'insights' && (
                <AIInsightsSection
                  data={data} domainProfile={domainProfile}
                  reports={reports} reportLoading={reportLoading}
                  runAnalysis={runAnalysis}
                />
              )}
            </motion.div>
          </AnimatePresence>
        </div>

      </div>
    </FadeIn>
  );
}
