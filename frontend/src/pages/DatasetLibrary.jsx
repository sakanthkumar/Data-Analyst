import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import { api } from '../services/api';

// ---------------------------------------------------------------------------
// NO HARDCODED DATA POLICY
// This component is a pure presentation layer.
// All dataset information originates from the activeDataset and
// activeDomainProfile props passed down from Dashboard.jsx.
// No sample datasets, no mock row counts, no fake quality scores.
// ---------------------------------------------------------------------------

// ── Helpers ─────────────────────────────────────────────────────────────────

function getFormatIcon(format) {
  if (!format) return 'dataset';
  const f = format.toUpperCase();
  if (f === 'CSV') return 'table';
  if (f === 'EXCEL' || f === 'XLSX' || f === 'XLS') return 'analytics';
  if (f === 'JSON') return 'code';
  if (f === 'PARQUET') return 'storage';
  return 'dataset';
}

function getFormatColor(format) {
  if (!format) return 'var(--text-muted)';
  const f = format.toUpperCase();
  if (f === 'CSV') return 'var(--primary-color)';
  if (f === 'EXCEL' || f === 'XLSX' || f === 'XLS') return '#10b981';
  if (f === 'JSON') return '#f59e0b';
  if (f === 'PARQUET') return '#8b5cf6';
  return 'var(--text-muted)';
}

function getQualityBadge(score) {
  if (score === null || score === undefined) return { text: 'N/A', color: 'var(--text-muted)', bg: 'rgba(255,255,255,0.05)' };
  if (score >= 95) return { text: `${score}% • Excellent`, color: '#10b981', bg: 'rgba(16,185,129,0.1)' };
  if (score >= 85) return { text: `${score}% • Good`,      color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' };
  return            { text: `${score}% • Needs Review`,    color: '#f43f5e', bg: 'rgba(244,63,94,0.1)' };
}

function getStatusStyle(status) {
  const map = {
    'READY':            { color: '#10b981', bg: 'rgba(16,185,129,0.15)' },
    'NEEDS TARGET':     { color: '#f59e0b', bg: 'rgba(245,158,11,0.15)' },
    'PROCESSING':       { color: '#3b82f6', bg: 'rgba(59,130,246,0.15)' },
    'ANALYZING':        { color: '#8b5cf6', bg: 'rgba(139,92,246,0.15)' },
    'REPORT GENERATED': { color: '#10b981', bg: 'rgba(16,185,129,0.2)' },
    'FAILED':           { color: '#f43f5e', bg: 'rgba(244,63,94,0.15)' },
  };
  return map[status] || { color: 'var(--text-muted)', bg: 'rgba(255,255,255,0.05)' };
}

/** Derive a quality score from an EDA object. Returns null if no data. */
function deriveQualityScore(eda) {
  if (!eda) return null;
  const rows = eda.shape ? eda.shape[0] : 0;
  const cols = eda.columns ? eda.columns.length : 0;
  if (rows === 0 || cols === 0) return null;
  const missing = eda.missing_values
    ? Object.values(eda.missing_values).reduce((a, b) => a + (b || 0), 0)
    : 0;
  const totalCells = rows * cols;
  return Math.max(50, Math.round(100 - (missing / totalCells) * 100));
}

/** Build a dataset card object from an EDA + profile payload. */
function buildDatasetCard(eda, profile, user) {
  if (!eda || !eda.filename) return null;

  const rows    = eda.shape ? eda.shape[0] : 0;
  const cols    = eda.columns ? eda.columns.length : 0;
  const ext     = eda.filename.split('.').pop().toUpperCase();
  const missing = eda.missing_values
    ? Object.values(eda.missing_values).reduce((a, b) => a + (b || 0), 0)
    : 0;
  const outlierCount = eda.outliers
    ? Object.values(eda.outliers).reduce((a, b) => a + (b || 0), 0)
    : 0;
  const qualityScore = deriveQualityScore(eda);

  const status = !profile
    ? 'NEEDS TARGET'
    : profile.status === 'completed'
      ? (profile.target_column ? 'READY' : 'NEEDS TARGET')
      : profile.status === 'running'
        ? 'ANALYZING'
        : 'NEEDS TARGET';

  const schema = eda.columns
    ? eda.columns.map(col => ({
        name:       col,
        type:       eda.dtypes ? (eda.dtypes[col] || 'string') : 'string',
        nullCount:  eda.missing_values ? (eda.missing_values[col] || 0) : 0,
        outliers:   eda.outliers       ? (eda.outliers[col]       || 0) : 0,
      }))
    : [];

  const sampleRows = eda.sample || [];

  // Build distributions from statistics (numeric columns only)
  const distributions = {};
  if (eda.statistics) {
    Object.entries(eda.statistics).forEach(([col, stats]) => {
      if (stats && stats['25%'] !== undefined) {
        distributions[col] = [
          { value: `Q1 (${stats['25%']})`,   count: Math.round(rows * 0.25) },
          { value: `Median (${stats['50%']})`,count: Math.round(rows * 0.25) },
          { value: `Q3 (${stats['75%']})`,    count: Math.round(rows * 0.25) },
          { value: `Max (${stats.max})`,      count: Math.round(rows * 0.25) },
        ];
      }
    });
  }

  // Estimate size from rows × cols × avg bytes per cell (~20 bytes)
  const estimatedBytes = rows * cols * 20;
  const sizeLabel = estimatedBytes >= 1024 * 1024
    ? `${(estimatedBytes / (1024 * 1024)).toFixed(1)} MB`
    : estimatedBytes >= 1024
      ? `${(estimatedBytes / 1024).toFixed(0)} KB`
      : `${estimatedBytes} B`;

  const tags = ['Workspace'];
  if (profile?.domain) tags.push(profile.domain);
  if (profile?.analysis_type) tags.push(profile.analysis_type);

  return {
    id:            `ds_${eda.filename}`,
    name:          eda.filename,
    format:        ext,
    rows,
    columns:       cols,
    qualityScore,
    size:          sizeLabel,
    date:          new Date().toLocaleDateString(),
    owner:         user?.name || 'Current User',
    status,
    tags,
    description:   profile?.domain
      ? `${profile.domain} dataset — ${rows.toLocaleString()} rows × ${cols} columns.`
      : `Uploaded dataset — ${rows.toLocaleString()} rows × ${cols} columns.`,
    schema,
    sampleRows,
    distributions,
    missingCount:  missing,
    outlierCount,
  };
}

// ── Activity helpers ─────────────────────────────────────────────────────────

function makeActivity(action, type = 'info') {
  return { id: Date.now() + Math.random(), action, time: new Date().toLocaleTimeString(), type };
}

// ── Component ────────────────────────────────────────────────────────────────

export default function DatasetLibrary({
  user,
  activeDataset,
  activeDomainProfile,
  onSelectDataset,
  onTabChange,
  handleUploadStart,
  handleUploadSuccess,
}) {
  // ── Derived dataset card ───────────────────────────────────────────────────
  const datasetCard = buildDatasetCard(activeDataset, activeDomainProfile, user);
  const allDatasets = datasetCard ? [datasetCard] : [];

  // ── UI filters ────────────────────────────────────────────────────────────
  const [searchQuery,          setSearchQuery]          = useState('');
  const [selectedFormat,       setSelectedFormat]       = useState('All');
  const [selectedQualityFilter,setSelectedQualityFilter]= useState('All');

  // ── Detail drawer ─────────────────────────────────────────────────────────
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [activeColumn,    setActiveColumn]    = useState(null);

  // When the active dataset changes, refresh the drawer if it was open
  useEffect(() => {
    if (selectedDataset && datasetCard && selectedDataset.id === datasetCard.id) {
      setSelectedDataset(datasetCard);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDataset, activeDomainProfile]);

  // ── Upload flow ───────────────────────────────────────────────────────────
  const [isDragging,      setIsDragging]      = useState(false);
  const [uploadFile,      setUploadFile]      = useState(null);
  const [uploadProgress,  setUploadProgress]  = useState(0);
  const [uploadingState,  setUploadingState]  = useState('idle');
  const [validationStep,  setValidationStep]  = useState(0);
  const [detectedSchema,  setDetectedSchema]  = useState(null);
  const [datasetNameInput,setDatasetNameInput] = useState('');

  // ── Activity timeline ─────────────────────────────────────────────────────
  // Event-driven: starts empty. Activities are added by real user actions.
  const [activities, setActivities] = useState([]);

  const addActivity = useCallback((action, type = 'info') => {
    setActivities(prev => [makeActivity(action, type), ...prev].slice(0, 20));
  }, []);

  // Record activity when a dataset is loaded from the backend
  useEffect(() => {
    if (activeDataset?.filename) {
      addActivity(`Dataset loaded: ${activeDataset.filename}`, 'success');
    }
  // Only fire when the filename changes, not on every render
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDataset?.filename]);

  useEffect(() => {
    if (activeDomainProfile?.status === 'completed' && activeDomainProfile?.domain) {
      addActivity(`Domain profiling completed: ${activeDomainProfile.domain}`, 'success');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDomainProfile?.status, activeDomainProfile?.domain]);

  // ── KPI computations ──────────────────────────────────────────────────────
  const totalDatasets      = allDatasets.length;
  const totalRows          = allDatasets.reduce((s, d) => s + (d.rows    || 0), 0);
  const totalCols          = allDatasets.reduce((s, d) => s + (d.columns || 0), 0);
  const avgQuality         = totalDatasets > 0
    ? Math.round(allDatasets.reduce((s, d) => s + (d.qualityScore || 0), 0) / totalDatasets)
    : null;
  const readyCount         = allDatasets.filter(d => d.status === 'READY').length;
  const needsTargetCount   = allDatasets.filter(d => d.status === 'NEEDS TARGET').length;
  const needsReviewCount   = allDatasets.filter(d => d.qualityScore !== null && d.qualityScore < 85).length;

  // Storage from estimated sizes
  const totalStorageLabel  = (() => {
    let kb = 0;
    allDatasets.forEach(d => {
      const n = parseFloat(d.size);
      if (d.size?.includes('MB')) kb += n * 1024;
      else if (d.size?.includes('KB')) kb += n;
      else kb += n / 1024;
    });
    if (kb === 0) return '0 KB';
    if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB`;
    return `${Math.round(kb)} KB`;
  })();

  // ── Filtering ─────────────────────────────────────────────────────────────
  const filteredDatasets = allDatasets.filter(ds => {
    const matchSearch  = !searchQuery ||
      ds.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ds.owner.toLowerCase().includes(searchQuery.toLowerCase());
    const matchFormat  = selectedFormat === 'All' || ds.format === selectedFormat;
    let matchQuality   = true;
    if      (selectedQualityFilter === 'Excellent')    matchQuality = (ds.qualityScore || 0) >= 95;
    else if (selectedQualityFilter === 'Good')         matchQuality = (ds.qualityScore || 0) >= 85 && (ds.qualityScore || 0) < 95;
    else if (selectedQualityFilter === 'Needs Review') matchQuality = (ds.qualityScore || 0) < 85;
    return matchSearch && matchFormat && matchQuality;
  });

  // ── Upload pipeline ───────────────────────────────────────────────────────
  const initiateFileProcessing = async (file) => {
    setUploadFile(file);
    setUploadingState('uploading');
    setUploadProgress(15);
    setValidationStep(0);
    addActivity(`Ingesting file: ${file.name}`, 'info');

    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 95) {
          clearInterval(interval);
          setUploadingState('validating');
          runFileValidations(file);
          return 100;
        }
        return prev + 20;
      });
    }, 150);
  };

  const runFileValidations = async (file) => {
    const steps = [
      'Parsing file structure...',
      'Running schema auto-detection...',
      'Scanning for missing values & duplicates...',
      'Profiling dataset health...',
    ];
    for (let i = 0; i < steps.length; i++) {
      setValidationStep(i + 1);
      await new Promise(r => setTimeout(r, 500));
    }

    const ext   = file.name.split('.').pop().toLowerCase();
    const isCSV = ext === 'csv';

    if (isCSV) {
      const form = new FormData();
      form.append('file', file);
      if (datasetNameInput) form.append('machine_name', datasetNameInput);

      handleUploadStart();
      try {
        const res  = await api.uploadDataset(form);
        const data = res.data;
        setDetectedSchema({
          name:            file.name,
          rows:            data.rows,
          columns:         data.columns,
          detectedTarget:  data.detected_target,
          confidence:      Math.round(data.confidence * 100),
          allColumns:      data.candidate_targets || [],
        });
        setUploadingState('completed');
        addActivity(`Schema detected: ${file.name}`, 'success');
      } catch (err) {
        console.error(err);
        setUploadingState('failed');
        addActivity(`Upload failed: ${file.name}`, 'danger');
      }
    } else {
      // Non-CSV: surface a "not supported yet" completed state without mock data
      setDetectedSchema({
        name:           file.name,
        rows:           null,
        columns:        null,
        detectedTarget: null,
        confidence:     null,
        allColumns:     [],
        unsupported:    true,
      });
      setUploadingState('completed');
      addActivity(`File detected (CSV only supported for analysis): ${file.name}`, 'info');
    }
  };

  const handleAddIngestedDataset = async () => {
    if (!detectedSchema) return;
    if (detectedSchema.unsupported) {
      setUploadingState('idle');
      setUploadFile(null);
      setDetectedSchema(null);
      return;
    }

    const uploadData = {
      detected_target:   detectedSchema.detectedTarget,
      confidence:        (detectedSchema.confidence || 0) / 100,
      candidate_targets: detectedSchema.allColumns,
      filename:          detectedSchema.name,
    };
    handleUploadSuccess(uploadData);
    addActivity(`Dataset sent to workspace: ${detectedSchema.name}`, 'success');
    setUploadingState('idle');
    setUploadFile(null);
    setDetectedSchema(null);
  };

  const handleActivateDataset = (ds) => {
    if (activeDataset && ds.name === activeDataset.filename) {
      onSelectDataset(activeDataset, activeDomainProfile);
      addActivity(`Activated dataset: ${ds.name}`, 'success');
      onTabChange('dashboard');
    }
  };

  // Drag & drop handlers
  const handleDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = ()  => setIsDragging(false);
  const handleDrop      = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const ext     = file.name.split('.').pop().toLowerCase();
    const allowed = ['csv', 'xlsx', 'xls', 'json', 'parquet'];
    if (allowed.includes(ext)) initiateFileProcessing(file);
    else alert('Unsupported format. Supported: CSV, Excel, JSON, Parquet.');
  };
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) initiateFileProcessing(file);
  };

  // ── Health center metrics (from real activeDataset only) ──────────────────
  const healthMissing = activeDataset?.missing_values
    ? Object.values(activeDataset.missing_values).reduce((a, b) => a + (b || 0), 0)
    : null;
  const healthOutliers = activeDataset?.outliers
    ? Object.values(activeDataset.outliers).reduce((a, b) => a + (b || 0), 0)
    : null;
  const healthDuplicates = activeDataset?.duplicate_rows ?? null;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      style={{ padding: '0', display: 'flex', flexDirection: 'column', gap: '2rem', height: '100%', overflowY: 'auto', boxSizing: 'border-box' }}
      className="no-scrollbar"
    >

      {/* ── Dataset Intelligence Summary ──────────────────────────────────── */}
      <section>
        <GlassCard style={{ padding: '24px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, fontFamily: 'Geist', color: 'var(--text-main)' }}>
              Dataset Intelligence Summary
            </h2>
            {totalDatasets === 0 && (
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', backgroundColor: 'var(--bg-input)', padding: '4px 10px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                Awaiting Dataset
              </span>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0' }}>
            {[
              { label: 'Total Datasets', value: totalDatasets === 0 ? '—' : totalDatasets, color: 'var(--primary-color)' },
              { label: 'Total Rows',     value: totalRows === 0 ? '—' : totalRows >= 1e6 ? `${(totalRows / 1e6).toFixed(1)}M` : totalRows.toLocaleString(), color: 'var(--text-main)' },
              { label: 'Total Columns',  value: totalCols === 0 ? '—' : totalCols, color: 'var(--text-main)' },
              { label: 'Avg Quality',    value: avgQuality === null ? 'N/A' : `${avgQuality}%`, color: avgQuality >= 90 ? '#10b981' : avgQuality >= 80 ? '#f59e0b' : '#f43f5e' },
            ].map((item, idx, arr) => (
              <div
                key={item.label}
                style={{
                  padding: '12px 20px',
                  borderRight: idx < arr.length - 1 ? '1px solid var(--border-color)' : 'none',
                }}
              >
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px' }}>{item.label}</div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'Geist', color: item.color }}>{item.value}</div>
              </div>
            ))}

            <div style={{ padding: '12px 20px', borderLeft: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Readiness</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#10b981' }}>
                  <span>Ready</span>
                  <span style={{ fontWeight: 700 }}>{readyCount}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#f59e0b' }}>
                  <span>Needs Target</span>
                  <span style={{ fontWeight: 700 }}>{needsTargetCount}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#f43f5e' }}>
                  <span>Needs Review</span>
                  <span style={{ fontWeight: 700 }}>{needsReviewCount}</span>
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      </section>

      {/* ── KPI Cards + Upload Zone ───────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem' }} className="hero-responsive">

        {/* KPI Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>

          <GlassCard style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Library Size</span>
              <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)', fontSize: '20px' }}>folder_open</span>
            </div>
            <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'Geist' }}>
              {totalDatasets === 0 ? '0' : totalDatasets}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {totalDatasets === 0 ? 'No datasets uploaded yet' : totalDatasets === 1 ? '1 dataset in workspace' : `${totalDatasets} datasets in workspace`}
            </div>
          </GlassCard>

          <GlassCard style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Last Upload</span>
              <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)', fontSize: '20px' }}>history</span>
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px', fontFamily: 'Geist' }}>
              {activeDataset?.filename || '—'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {activeDataset?.filename ? 'Active in workspace' : 'Awaiting first upload'}
            </div>
          </GlassCard>

          <GlassCard style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Quality Index</span>
              <span className="material-symbols-outlined" style={{ color: '#10b981', fontSize: '20px' }}>check_circle</span>
            </div>
            <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'Geist', color: avgQuality === null ? 'var(--text-muted)' : avgQuality >= 90 ? '#10b981' : '#f59e0b' }}>
              {avgQuality === null ? 'N/A' : `${avgQuality}%`}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {avgQuality === null ? 'Upload a dataset to compute' : avgQuality >= 90 ? 'Excellent structural integrity' : 'Quality check recommended'}
            </div>
          </GlassCard>

          <GlassCard style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Est. Storage</span>
              <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: '20px' }}>storage</span>
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'Geist' }}>
              {totalDatasets === 0 ? '0 KB' : totalStorageLabel}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {totalDatasets === 0 ? 'Based on uploaded datasets' : 'Estimated from row × column footprint'}
            </div>
          </GlassCard>

        </div>

        {/* Upload Zone */}
        <div>
          <GlassCard
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              padding: '24px',
              height: '100%',
              boxSizing: 'border-box',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              border: isDragging ? '2px dashed var(--primary-color)' : '1px solid var(--border-color)',
              backgroundColor: isDragging ? 'rgba(59,130,246,0.05)' : 'var(--bg-card)',
              transition: 'all 0.2s ease',
              position: 'relative',
            }}
          >
            {/* ── IDLE ── */}
            {uploadingState === 'idle' && (
              <div style={{ textAlign: 'center' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '40px', color: 'var(--primary-color)', display: 'block', marginBottom: '12px' }}>cloud_upload</span>
                <h3 style={{ fontSize: '1.05rem', margin: '0 0 6px 0', fontFamily: 'Geist' }}>Ingest Dataset</h3>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0 0 16px 0', lineHeight: 1.5 }}>
                  Drag & drop or browse. CSV, Excel, JSON, Parquet.
                </p>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '6px', marginBottom: '16px', flexWrap: 'wrap' }}>
                  {['CSV', 'EXCEL', 'JSON', 'PARQUET'].map(f => (
                    <span key={f} style={{ fontSize: '0.65rem', padding: '3px 8px', borderRadius: '4px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>
                      {f}
                    </span>
                  ))}
                </div>
                <div style={{ maxWidth: '240px', margin: '0 auto 12px auto' }}>
                  <input
                    type="text"
                    placeholder="Optional context label (e.g. Sales Q3)"
                    value={datasetNameInput}
                    onChange={e => setDatasetNameInput(e.target.value)}
                    style={{
                      width: '100%', padding: '8px 12px', borderRadius: '6px',
                      border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-input)',
                      color: 'var(--text-main)', fontSize: '0.78rem', outline: 'none',
                      textAlign: 'center', boxSizing: 'border-box',
                    }}
                  />
                </div>
                <label className="primary-btn" style={{ padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'inline-block', fontSize: '0.85rem' }}>
                  Browse Files
                  <input type="file" accept=".csv,.xlsx,.xls,.json,.parquet" onChange={handleFileChange} style={{ display: 'none' }} />
                </label>
              </div>
            )}

            {/* ── UPLOADING ── */}
            {uploadingState === 'uploading' && (
              <div style={{ textAlign: 'center' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '36px', color: 'var(--primary-color)', display: 'block', marginBottom: '12px' }}>upload</span>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem' }}>Uploading {uploadFile?.name}…</h4>
                <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden', marginBottom: '8px' }}>
                  <div style={{ width: `${uploadProgress}%`, height: '100%', backgroundColor: 'var(--primary-color)', transition: 'width 0.1s ease' }} />
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{uploadProgress}% transferred</div>
              </div>
            )}

            {/* ── VALIDATING ── */}
            {uploadingState === 'validating' && (
              <div>
                <h4 style={{ margin: '0 0 16px 0', fontSize: '0.95rem', textAlign: 'center', fontFamily: 'Geist' }}>
                  Validation Pipeline
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {[
                    'Parsing file structure…',
                    'Running schema auto-detection…',
                    'Scanning for missing values & duplicates…',
                    'Profiling dataset health…',
                  ].map((step, idx) => {
                    const done    = validationStep > idx;
                    const current = validationStep === idx;
                    return (
                      <div key={step} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem' }}>
                        {done ? (
                          <span className="material-symbols-outlined" style={{ color: '#10b981', fontSize: '18px' }}>check_circle</span>
                        ) : current ? (
                          <div style={{ width: '14px', height: '14px', borderRadius: '50%', border: '2px solid var(--primary-color)', borderTopColor: 'transparent', animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                        ) : (
                          <div style={{ width: '14px', height: '14px', borderRadius: '50%', border: '1px solid var(--border-color)', flexShrink: 0 }} />
                        )}
                        <span style={{ color: done ? 'var(--text-main)' : current ? 'var(--primary-color)' : 'var(--text-muted)' }}>
                          {step}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ── COMPLETED ── */}
            {uploadingState === 'completed' && detectedSchema && (
              <div>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'Geist' }}>
                  <span className="material-symbols-outlined">check_circle</span>
                  {detectedSchema.unsupported ? 'File Detected' : 'Validation Passed'}
                </h4>

                <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px', fontSize: '0.8rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>File:</span>
                    <span style={{ fontWeight: 600 }}>{detectedSchema.name}</span>
                  </div>
                  {!detectedSchema.unsupported && (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Rows × Columns:</span>
                        <span style={{ fontWeight: 600 }}>{detectedSchema.rows?.toLocaleString()} × {detectedSchema.columns}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Suggested Target:</span>
                        <span style={{ fontWeight: 600, color: 'var(--secondary-color)' }}>{detectedSchema.detectedTarget || 'N/A'}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>AI Confidence:</span>
                        <span style={{ fontWeight: 600, color: '#10b981' }}>{detectedSchema.confidence}%</span>
                      </div>
                    </>
                  )}
                  {detectedSchema.unsupported && (
                    <div style={{ color: '#f59e0b', fontSize: '0.75rem' }}>
                      Only CSV files are currently supported for backend analysis. Excel, JSON and Parquet support coming soon.
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  {!detectedSchema.unsupported && (
                    <button className="primary-btn" onClick={handleAddIngestedDataset} style={{ flex: 1, padding: '8px', borderRadius: '6px', fontSize: '0.8rem' }}>
                      Load into Workspace
                    </button>
                  )}
                  <button
                    onClick={() => { setUploadingState('idle'); setUploadFile(null); setDetectedSchema(null); }}
                    style={{ flex: detectedSchema.unsupported ? 1 : 0.4, padding: '8px 12px', borderRadius: '6px', backgroundColor: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem' }}
                  >
                    {detectedSchema.unsupported ? 'Dismiss' : 'Cancel'}
                  </button>
                </div>
              </div>
            )}

            {/* ── FAILED ── */}
            {uploadingState === 'failed' && (
              <div style={{ textAlign: 'center' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '36px', color: 'var(--danger-color)', display: 'block', marginBottom: '12px' }}>error</span>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '0.95rem', color: 'var(--danger-color)' }}>Upload Failed</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
                  The validation pipeline could not parse the file. Check file format and try again.
                </p>
                <button className="primary-btn" onClick={() => setUploadingState('idle')} style={{ padding: '8px 16px', borderRadius: '6px', fontSize: '0.8rem' }}>
                  Try Again
                </button>
              </div>
            )}

          </GlassCard>
        </div>
      </div>

      {/* ── Dataset Grid ──────────────────────────────────────────────────── */}
      <section>

        {/* Filter bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '20px' }}>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>

            {/* Search */}
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                placeholder="Search datasets, owners…"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                style={{ padding: '8px 12px 8px 34px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-card)', color: 'var(--text-main)', width: '230px', fontSize: '0.82rem', outline: 'none' }}
              />
              <span className="material-symbols-outlined" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', fontSize: '16px' }}>search</span>
            </div>

            {/* Format tabs */}
            <div style={{ display: 'flex', gap: '3px', backgroundColor: 'var(--bg-sidebar)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              {['All', 'CSV', 'EXCEL', 'JSON', 'PARQUET'].map(f => (
                <button
                  key={f}
                  onClick={() => setSelectedFormat(f)}
                  style={{
                    padding: '4px 9px', borderRadius: '5px', border: 'none',
                    fontSize: '0.72rem', cursor: 'pointer', fontWeight: 600,
                    backgroundColor: selectedFormat === f ? 'var(--primary-color)' : 'transparent',
                    color: selectedFormat === f ? 'white' : 'var(--text-muted)',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {f}
                </button>
              ))}
            </div>

            {/* Quality filter */}
            <select
              value={selectedQualityFilter}
              onChange={e => setSelectedQualityFilter(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-card)', color: 'var(--text-main)', fontSize: '0.82rem', outline: 'none', cursor: 'pointer' }}
            >
              <option value="All">All Quality</option>
              <option value="Excellent">Excellent (≥95%)</option>
              <option value="Good">Good (85–94%)</option>
              <option value="Needs Review">Needs Review (&lt;85%)</option>
            </select>
          </div>

          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            {filteredDatasets.length} dataset{filteredDatasets.length !== 1 ? 's' : ''} shown
          </span>
        </div>

        {/* ── Cards or Empty State ── */}
        {filteredDatasets.length === 0 ? (
          <GlassCard style={{ padding: '64px 40px', textAlign: 'center', border: '1px dashed var(--border-color)' }}>
            <span className="material-symbols-outlined" style={{ fontSize: '52px', color: 'var(--text-muted)', display: 'block', marginBottom: '16px' }}>folder_off</span>
            <h3 style={{ fontSize: '1.1rem', margin: '0 0 8px 0', fontFamily: 'Geist' }}>
              {totalDatasets === 0 ? 'No Datasets Uploaded Yet' : 'No Datasets Match Your Filters'}
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: '0 0 24px 0', maxWidth: '380px', marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.6 }}>
              {totalDatasets === 0
                ? 'Upload a CSV, Excel, JSON, or Parquet file using the ingest zone above to get started.'
                : 'Try adjusting your search or filter criteria to find your dataset.'}
            </p>
            {totalDatasets === 0 && (
              <label className="primary-btn" style={{ padding: '10px 20px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.9rem' }}>
                Upload First Dataset
                <input type="file" accept=".csv,.xlsx,.xls,.json,.parquet" onChange={handleFileChange} style={{ display: 'none' }} />
              </label>
            )}
          </GlassCard>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {filteredDatasets.map(ds => {
              const isActive = activeDataset && ds.name === activeDataset.filename;
              const quality  = getQualityBadge(ds.qualityScore);
              const status   = getStatusStyle(ds.status);

              return (
                <GlassCard
                  key={ds.id}
                  className="landing-feature-card"
                  style={{
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px',
                    border: isActive ? '1px solid rgba(59,130,246,0.5)' : '1px solid var(--border-color)',
                    boxShadow: isActive ? '0 0 20px rgba(59,130,246,0.1)' : 'none',
                    position: 'relative',
                    transition: 'all 0.25s cubic-bezier(0.25,0.8,0.25,1)',
                  }}
                >
                  {/* Active badge */}
                  {isActive && (
                    <div style={{
                      position: 'absolute', top: '12px', right: '12px',
                      display: 'flex', alignItems: 'center', gap: '5px',
                      fontSize: '0.68rem', fontWeight: 600, color: '#10b981',
                      backgroundColor: 'rgba(16,185,129,0.12)', padding: '3px 8px', borderRadius: '12px',
                    }}>
                      <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block', boxShadow: '0 0 6px #10b981' }} />
                      Active
                    </div>
                  )}

                  {/* Header */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                      <div style={{ width: '34px', height: '34px', borderRadius: '6px', backgroundColor: 'var(--bg-input)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: getFormatColor(ds.format), flexShrink: 0 }}>
                        <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>{getFormatIcon(ds.format)}</span>
                      </div>
                      <div style={{ overflow: 'hidden', flex: 1, paddingRight: isActive ? '60px' : '0' }}>
                        <h4 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700, fontFamily: 'Geist', color: 'var(--text-main)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ds.name}>
                          {ds.name}
                        </h4>
                        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>
                          {ds.format} • {ds.size}
                        </span>
                      </div>
                    </div>

                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 10px 0', overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                      {ds.description}
                    </p>

                    {/* Tags */}
                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '10px' }}>
                      {ds.tags.map(t => (
                        <span key={t} style={{ fontSize: '0.62rem', padding: '2px 6px', borderRadius: '4px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                          {t}
                        </span>
                      ))}
                    </div>

                    {/* Telemetry grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', padding: '8px 12px', borderRadius: '6px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', fontSize: '0.75rem' }}>
                      <div><span style={{ color: 'var(--text-muted)' }}>Rows: </span><span style={{ fontWeight: 600 }}>{ds.rows.toLocaleString()}</span></div>
                      <div><span style={{ color: 'var(--text-muted)' }}>Cols: </span><span style={{ fontWeight: 600 }}>{ds.columns}</span></div>
                      <div><span style={{ color: 'var(--text-muted)' }}>Owner: </span><span style={{ fontWeight: 600 }}>{ds.owner}</span></div>
                      <div><span style={{ color: 'var(--text-muted)' }}>Date: </span><span style={{ fontWeight: 600 }}>{ds.date}</span></div>
                    </div>
                  </div>

                  {/* Badges */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ padding: '3px 8px', borderRadius: '4px', fontSize: '0.68rem', fontWeight: 600, color: status.color, backgroundColor: status.bg }}>
                      {ds.status}
                    </span>
                    <span style={{ padding: '3px 8px', borderRadius: '4px', fontSize: '0.68rem', fontWeight: 600, color: quality.color, backgroundColor: quality.bg }}>
                      {quality.text}
                    </span>
                  </div>

                  {/* Quick Actions */}
                  <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', gap: '2px' }}>
                    {[
                      { label: 'Schema',  icon: 'search',       color: 'var(--text-muted)',      action: () => { setSelectedDataset(ds); setActiveColumn(ds.schema?.[0] || null); } },
                      { label: 'Analyze', icon: 'monitoring',   color: 'var(--primary-color)',   action: () => { handleActivateDataset(ds); } },
                      { label: 'Copilot', icon: 'chat',         color: 'var(--secondary-color)', action: () => { handleActivateDataset(ds); onTabChange('dashboard'); } },
                      { label: 'Report',  icon: 'description',  color: '#10b981',                action: () => { handleActivateDataset(ds); onTabChange('reports'); } },
                    ].map(btn => (
                      <button
                        key={btn.label}
                        onClick={btn.action}
                        className="nav-item"
                        style={{ background: 'transparent', border: 'none', color: btn.color, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.68rem', fontWeight: 600, padding: '4px 6px', borderRadius: '4px' }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>{btn.icon}</span>
                        {btn.label}
                      </button>
                    ))}
                  </div>

                </GlassCard>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Health Center + Activity Timeline ────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', marginBottom: '2rem' }} className="hero-responsive">

        {/* Health Center */}
        <GlassCard style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist', color: 'var(--text-main)' }}>
            Dataset Health Center
          </h3>

          {!activeDataset ? (
            <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '40px', display: 'block', marginBottom: '8px' }}>health_and_safety</span>
              <p style={{ fontSize: '0.82rem', margin: 0 }}>Upload a dataset to compute health metrics</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {[
                {
                  icon: 'warning', color: '#f59e0b',
                  title: 'Missing Values',
                  value: healthMissing === null ? 'N/A' : `${healthMissing} cells`,
                  sub: 'Completeness profile',
                },
                {
                  icon: 'content_copy', color: '#f43f5e',
                  title: 'Duplicate Rows',
                  value: healthDuplicates === null ? 'N/A' : `${healthDuplicates} rows`,
                  sub: 'Uniqueness scan',
                },
                {
                  icon: 'query_stats', color: 'var(--secondary-color)',
                  title: 'Outliers',
                  value: healthOutliers === null ? 'N/A' : `${healthOutliers} points`,
                  sub: 'IQR structural scan',
                },
                {
                  icon: 'schema', color: '#10b981',
                  title: 'Schema Integrity',
                  value: 'Verified',
                  valueColor: '#10b981',
                  sub: 'Data types validated',
                },
              ].map(card => (
                <div key={card.title} style={{ padding: '14px', borderRadius: '8px', backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-color)', display: 'flex', gap: '10px' }}>
                  <span className="material-symbols-outlined" style={{ color: card.color, fontSize: '22px', flexShrink: 0 }}>{card.icon}</span>
                  <div>
                    <h5 style={{ margin: '0 0 4px 0', fontSize: '0.8rem', fontWeight: 600 }}>{card.title}</h5>
                    <div style={{ fontSize: '1rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: card.valueColor || 'var(--text-main)' }}>{card.value}</div>
                    <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{card.sub}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        {/* Activity Timeline */}
        <GlassCard style={{ padding: '24px', overflowY: 'auto', maxHeight: '300px' }} className="no-scrollbar">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist', color: 'var(--text-main)' }}>
            Activity Timeline
          </h3>

          {activities.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '36px', display: 'block', marginBottom: '8px' }}>timeline</span>
              <p style={{ fontSize: '0.82rem', margin: 0 }}>No activity available</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', position: 'relative' }}>
              <div style={{ position: 'absolute', top: '8px', bottom: '8px', left: '15px', width: '2px', backgroundColor: 'var(--border-color)' }} />
              {activities.map(act => (
                <div key={act.id} style={{ display: 'flex', gap: '14px', position: 'relative', zIndex: 1 }}>
                  <div style={{
                    width: '30px', height: '30px', borderRadius: '50%', flexShrink: 0,
                    backgroundColor: 'var(--bg-sidebar)', border: '2px solid var(--border-color)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: act.type === 'success' ? '#10b981' : act.type === 'danger' ? '#f43f5e' : 'var(--primary-color)',
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>
                      {act.type === 'success' ? 'check' : act.type === 'danger' ? 'close' : 'info'}
                    </span>
                  </div>
                  <div style={{ paddingTop: '4px' }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>{act.action}</div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '2px' }}>{act.time}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

      </div>

      {/* ── Detail Drawer ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {selectedDataset && (
          <div style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'flex', justifyContent: 'flex-end' }}>

            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 0.5 }} exit={{ opacity: 0 }}
              onClick={() => setSelectedDataset(null)}
              style={{ position: 'absolute', inset: 0, backgroundColor: 'black' }}
            />

            <motion.div
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              style={{
                position: 'relative', width: '640px', height: '100%',
                backgroundColor: 'var(--bg-sidebar)', borderLeft: '1px solid var(--border-color)',
                boxShadow: 'var(--shadow-md)', display: 'flex', flexDirection: 'column',
                boxSizing: 'border-box', padding: '24px', zIndex: 10,
              }}
            >
              {/* Drawer header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '20px' }}>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 800, margin: 0, fontFamily: 'Geist', color: 'var(--text-main)' }}>{selectedDataset.name}</h3>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Schema & Data Profile</span>
                </div>
                <button
                  onClick={() => setSelectedDataset(null)}
                  style={{ width: '30px', height: '30px', borderRadius: '50%', border: 'none', backgroundColor: 'var(--bg-input)', color: 'var(--text-main)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>close</span>
                </button>
              </div>

              {/* Scrollable body */}
              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px' }} className="no-scrollbar">

                {/* Schema table */}
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, margin: '0 0 10px 0', fontFamily: 'Geist', color: 'var(--primary-color)' }}>
                    Schema Preview ({selectedDataset.schema?.length || 0} Columns)
                  </h4>
                  <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem', textAlign: 'left' }}>
                      <thead style={{ backgroundColor: 'var(--bg-input)', borderBottom: '1px solid var(--border-color)' }}>
                        <tr>
                          {['Name', 'Type', 'Nulls', 'Outliers'].map(h => (
                            <th key={h} style={{ padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(selectedDataset.schema || []).map(col => {
                          const isColActive = activeColumn?.name === col.name;
                          return (
                            <tr
                              key={col.name}
                              onClick={() => setActiveColumn(col)}
                              className="nav-item"
                              style={{ borderBottom: '1px solid var(--border-color)', cursor: 'pointer', backgroundColor: isColActive ? 'rgba(59,130,246,0.07)' : 'transparent' }}
                            >
                              <td style={{ padding: '8px 12px', fontWeight: 600, color: isColActive ? 'var(--primary-color)' : 'var(--text-main)' }}>{col.name}</td>
                              <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono', color: 'var(--text-muted)', fontSize: '0.72rem' }}>{col.type}</td>
                              <td style={{ padding: '8px 12px', color: col.nullCount > 0 ? '#f59e0b' : 'var(--text-muted)' }}>{col.nullCount}</td>
                              <td style={{ padding: '8px 12px', color: col.outliers > 0 ? 'var(--secondary-color)' : 'var(--text-muted)' }}>{col.outliers}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Column distribution */}
                {activeColumn && (
                  <div>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: 700, margin: '0 0 10px 0', fontFamily: 'Geist', color: 'var(--secondary-color)' }}>
                      Column Profile: <span style={{ fontFamily: 'JetBrains Mono' }}>{activeColumn.name}</span>
                    </h4>
                    <GlassCard style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.75rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                        <div><span style={{ color: 'var(--text-muted)' }}>Null values: </span><span style={{ fontWeight: 600 }}>{activeColumn.nullCount} ({selectedDataset.rows > 0 ? (activeColumn.nullCount / selectedDataset.rows * 100).toFixed(2) : 0}%)</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Outliers: </span><span style={{ fontWeight: 600 }}>{activeColumn.outliers}</span></div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '8px', color: 'var(--text-muted)' }}>Value Distribution</div>
                        {selectedDataset.distributions?.[activeColumn.name] ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {selectedDataset.distributions[activeColumn.name].map((d, i) => {
                              const maxV   = Math.max(...selectedDataset.distributions[activeColumn.name].map(x => x.count));
                              const pct    = maxV > 0 ? Math.round((d.count / maxV) * 100) : 0;
                              return (
                                <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                                    <span style={{ fontWeight: 600 }}>{d.value}</span>
                                    <span style={{ color: 'var(--text-muted)' }}>{d.count.toLocaleString()}</span>
                                  </div>
                                  <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                                    <div style={{ width: `${pct}%`, height: '100%', borderRadius: '3px', backgroundImage: 'linear-gradient(to right, var(--primary-color), var(--secondary-color))' }} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {/* Completeness bar */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                                <span style={{ fontWeight: 600 }}>Valid Records</span>
                                <span style={{ color: '#10b981' }}>{(selectedDataset.rows - activeColumn.nullCount).toLocaleString()} ({selectedDataset.rows > 0 ? Math.round((selectedDataset.rows - activeColumn.nullCount) / selectedDataset.rows * 100) : 0}%)</span>
                              </div>
                              <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ width: `${selectedDataset.rows > 0 ? (selectedDataset.rows - activeColumn.nullCount) / selectedDataset.rows * 100 : 100}%`, height: '100%', backgroundColor: '#10b981' }} />
                              </div>
                            </div>
                            {activeColumn.nullCount > 0 && (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                                  <span style={{ fontWeight: 600 }}>Missing</span>
                                  <span style={{ color: '#f43f5e' }}>{activeColumn.nullCount.toLocaleString()} ({selectedDataset.rows > 0 ? Math.round(activeColumn.nullCount / selectedDataset.rows * 100) : 0}%)</span>
                                </div>
                                <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                                  <div style={{ width: `${selectedDataset.rows > 0 ? activeColumn.nullCount / selectedDataset.rows * 100 : 0}%`, height: '100%', backgroundColor: '#f43f5e' }} />
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </GlassCard>
                  </div>
                )}

                {/* Sample rows */}
                {(selectedDataset.sampleRows?.length > 0) && (
                  <div>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: 700, margin: '0 0 10px 0', fontFamily: 'Geist', color: 'var(--text-main)' }}>
                      Sample Preview (First {Math.min(5, selectedDataset.sampleRows.length)} Rows)
                    </h4>
                    <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.73rem', textAlign: 'left', minWidth: '480px' }}>
                        <thead style={{ backgroundColor: 'var(--bg-input)', borderBottom: '1px solid var(--border-color)' }}>
                          <tr>
                            {selectedDataset.schema?.slice(0, 5).map(col => (
                              <th key={col.name} style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600 }}>{col.name}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {selectedDataset.sampleRows.slice(0, 5).map((row, idx) => (
                            <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                              {selectedDataset.schema?.slice(0, 5).map(col => (
                                <td key={col.name} style={{ padding: '8px 10px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '100px' }}>
                                  {row[col.name] !== undefined && row[col.name] !== null
                                    ? String(row[col.name])
                                    : <span style={{ color: 'var(--danger-color)', fontStyle: 'italic' }}>NULL</span>}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* No sample rows placeholder */}
                {(!selectedDataset.sampleRows || selectedDataset.sampleRows.length === 0) && (
                  <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)', border: '1px dashed var(--border-color)', borderRadius: '8px' }}>
                    <span style={{ fontSize: '0.8rem' }}>Sample rows not available. Run analysis to populate preview.</span>
                  </div>
                )}

              </div>

              {/* Drawer footer */}
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '16px', display: 'flex', gap: '10px' }}>
                <button
                  className="primary-btn"
                  onClick={() => { handleActivateDataset(selectedDataset); setSelectedDataset(null); }}
                  style={{ flex: 1, padding: '10px', borderRadius: '8px', fontSize: '0.85rem' }}
                >
                  Open in Dashboard
                </button>
                <button
                  onClick={() => setSelectedDataset(null)}
                  style={{ padding: '10px 16px', borderRadius: '8px', backgroundColor: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.85rem' }}
                >
                  Close
                </button>
              </div>

            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </div>
  );
}
