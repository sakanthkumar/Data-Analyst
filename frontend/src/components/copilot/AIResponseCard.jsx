import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function AIResponseCard({ responseData, plots, onSelectFollowUp, onPinInsight, isPinned, activeDataset, domainProfile }) {
  const [traceOpen, setTraceOpen] = useState(false);
  const [errorDetailsOpen, setErrorDetailsOpen] = useState(false);

  // Extract variables
  const {
    question = '',
    analysis = '',
    evidence = [],
    confidence = 90,
    visualization_type = null,
    recommendations = [],
    suggested_follow_ups = [],
    reasoning_trace = [],
    processingTime = null,
    timestamp = null,
    isError = false,
    errorDetails = ''
  } = responseData;

  // Retrieve base64 plot if it matches visualization_type
  const plotBase64 = plots && visualization_type ? plots[visualization_type] : null;

  // Resolve dataset meta info
  const datasetFilename = activeDataset?.filename || 'Unknown Dataset';
  const analysisType = domainProfile?.analysis_type || 'General';

  // 1. Error Recovery Template
  if (isError) {
    const displayError = errorDetails || "An unexpected network or backend processing error occurred.";
    return (
      <div 
        className="glass-card fade-in"
        style={{
          background: 'rgba(244, 63, 94, 0.03)',
          border: '1px solid rgba(244, 63, 94, 0.2)',
          borderRadius: '12px',
          padding: '24px',
          margin: '16px 0 32px 0',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          boxShadow: 'var(--shadow-md)',
          boxSizing: 'border-box',
          width: '100%'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="material-symbols-outlined" style={{ color: 'var(--danger-color)', fontSize: '24px' }}>
            error
          </span>
          <h4 style={{ margin: 0, fontFamily: 'Geist', fontSize: '1.05rem', color: 'var(--text-main)', fontWeight: 600 }}>
            Unable to generate analysis
          </h4>
        </div>

        <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          The analyst backend returned an error or is unreachable. Please verify your connection and try again.
        </p>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            onClick={() => onSelectFollowUp(question)}
            style={{
              backgroundColor: 'var(--danger-color)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '6px 14px',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'background 0.2s'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>refresh</span>
            Retry
          </button>

          <button
            onClick={() => setErrorDetailsOpen(!errorDetailsOpen)}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 14px',
              fontSize: '0.8rem',
              color: 'var(--text-main)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>info</span>
            {errorDetailsOpen ? "Hide Details" : "View Details"}
          </button>

          <button
            onClick={() => {
              navigator.clipboard.writeText(`Error: ${displayError}`);
              alert("Error details copied to clipboard!");
            }}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 14px',
              fontSize: '0.8rem',
              color: 'var(--text-main)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>content_copy</span>
            Copy Error
          </button>
        </div>

        {errorDetailsOpen && (
          <pre 
            style={{ 
              background: 'var(--bg-sidebar)', 
              border: '1px solid rgba(244, 63, 94, 0.1)', 
              borderRadius: '6px', 
              padding: '12px', 
              fontFamily: 'JetBrains Mono, monospace', 
              fontSize: '0.78rem', 
              color: 'var(--text-muted)',
              overflowX: 'auto',
              margin: 0
            }}
          >
            {displayError}
          </pre>
        )}
      </div>
    );
  }

  // 2. Normal Response Card
  return (
    <div 
      className="glass-card fade-in"
      style={{
        background: 'var(--bg-card)',
        backdropFilter: 'blur(20px)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '24px',
        margin: '16px 0 32px 0',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
        boxShadow: 'var(--shadow-md)',
        overflow: 'visible',
        boxSizing: 'border-box',
        width: '100%'
      }}
    >
      {/* Compact Metadata Header */}
      <div 
        style={{ 
          fontSize: '0.78rem',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div 
              style={{ 
                width: '26px', 
                height: '26px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--primary-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>
                analytics
              </span>
            </div>
            <span style={{ fontFamily: 'Geist', fontWeight: 600, color: 'var(--text-main)', fontSize: '0.9rem' }}>
              Analyst.AI Assistant
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Pin action */}
            <button
              onClick={onPinInsight}
              title={isPinned ? "Pinned to memory" : "Pin to memory"}
              style={{
                background: 'transparent',
                border: 'none',
                borderRadius: '4px',
                padding: '4px',
                color: isPinned ? 'var(--primary-color)' : 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                cursor: 'pointer',
                transition: 'color 0.2s'
              }}
            >
              <span 
                className="material-symbols-outlined" 
                style={{ 
                  fontSize: '18px',
                  fontVariationSettings: isPinned ? "'FILL' 1" : "'FILL' 0" 
                }}
              >
                push_pin
              </span>
            </button>
          </div>
        </div>

        {/* Structured Metadata Row */}
        <div 
          style={{ 
            display: 'flex', 
            gap: '12px', 
            flexWrap: 'wrap', 
            color: 'var(--text-muted)',
            fontSize: '0.75rem',
            fontFamily: 'JetBrains Mono, monospace'
          }}
        >
          <div>
            Dataset: <span style={{ color: 'var(--text-main)' }}>{datasetFilename}</span>
          </div>
          <div>•</div>
          <div>
            Type: <span style={{ color: 'var(--text-main)', textTransform: 'capitalize' }}>{analysisType}</span>
          </div>
          <div>•</div>
          <div>
            Conf: <span style={{ color: 'var(--accent-color)', fontWeight: 600 }}>{confidence}%</span>
          </div>
          {processingTime && (
            <>
              <div>•</div>
              <div>
                Time: <span style={{ color: 'var(--text-main)' }}>{processingTime}s</span>
              </div>
            </>
          )}
          {timestamp && (
            <>
              <div>•</div>
              <div>
                Logged: <span style={{ color: 'var(--text-main)' }}>{timestamp}</span>
              </div>
            </>
          )}
        </div>

        {/* Reference Question Box */}
        <div 
          style={{ 
            background: 'rgba(255,255,255,0.01)', 
            borderLeft: '2px solid var(--primary-color)', 
            paddingLeft: '8px', 
            margin: '4px 0', 
            fontStyle: 'italic', 
            color: 'var(--text-muted)',
            lineHeight: '1.4',
            fontSize: '0.825rem'
          }}
        >
          Q: "{question}"
        </div>
      </div>

      {/* Progressive Render Section 1: Analysis explanation */}
      <div className="progressive-section" style={{ animationDelay: '0.1s' }}>
        <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 10px 0', fontWeight: 700 }}>
          Analysis
        </h4>
        <div style={{ wordBreak: 'break-word', overflow: 'visible' }}>
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({node, ...props}) => (
                <div style={{ overflowX: 'auto', margin: '12px 0', width: '100%' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }} {...props} />
                </div>
              ),
              th: ({node, ...props}) => (
                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, background: 'var(--bg-sidebar)', borderBottom: '2px solid var(--border-color)', borderRight: '1px solid var(--border-color)' }} {...props} />
              ),
              td: ({node, ...props}) => (
                <td style={{ padding: '8px 12px', borderBottom: '1px solid var(--border-color)', borderRight: '1px solid var(--border-color)' }} {...props} />
              ),
              code: ({node, inline, className, children, ...props}) => {
                const isBlock = !inline && (className || String(children).includes('\n'));
                return isBlock ? (
                  <pre style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '12px 16px', overflowX: 'auto', margin: '10px 0', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.82rem', color: 'var(--text-main)', width: '100%', boxSizing: 'border-box' }}>
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                ) : (
                  <code style={{ fontFamily: 'JetBrains Mono, monospace', background: 'var(--bg-sidebar)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.82rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }} {...props}>
                    {children}
                  </code>
                );
              },
              h1: ({node, children, ...props}) => <h1 style={{ fontFamily: 'Geist', fontSize: '1.3rem', color: 'var(--text-main)', margin: '18px 0 8px 0', fontWeight: 600 }} {...props}>{children}</h1>,
              h2: ({node, children, ...props}) => <h2 style={{ fontFamily: 'Geist', fontSize: '1.15rem', color: 'var(--text-main)', margin: '14px 0 6px 0', fontWeight: 600 }} {...props}>{children}</h2>,
              h3: ({node, children, ...props}) => <h3 style={{ fontFamily: 'Geist', fontSize: '1.05rem', color: 'var(--text-main)', margin: '12px 0 4px 0', fontWeight: 600 }} {...props}>{children}</h3>,
              ul: ({node, ...props}) => <ul style={{ margin: '8px 0', paddingLeft: '24px', display: 'flex', flexDirection: 'column', gap: '6px' }} {...props} />,
              ol: ({node, ...props}) => <ol style={{ margin: '8px 0', paddingLeft: '24px', display: 'flex', flexDirection: 'column', gap: '6px' }} {...props} />,
              li: ({node, ...props}) => <li style={{ fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.5' }} {...props} />,
              p: ({node, ...props}) => <p style={{ margin: '6px 0', fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.55', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }} {...props} />
            }}
          >
            {analysis}
          </ReactMarkdown>
        </div>
      </div>

      {/* Progressive Render Section 2: Evidence Columns */}
      {evidence && evidence.length > 0 && (
        <div className="progressive-section" style={{ animationDelay: '0.3s' }}>
          <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 10px 0', fontWeight: 700 }}>
            Evidence
          </h4>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {evidence.map((col) => (
              <span
                key={col}
                className="label-technical"
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  padding: '4px 10px',
                  color: 'var(--text-main)',
                  fontSize: '0.75rem',
                  fontWeight: 500
                }}
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Progressive Render Section 3: Confidence Score */}
      {confidence !== undefined && confidence !== null && (
        <div className="progressive-section" style={{ animationDelay: '0.5s' }}>
          <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 10px 0', fontWeight: 700 }}>
            Confidence
          </h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div 
              style={{
                backgroundColor: 'rgba(16, 185, 129, 0.08)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                color: 'var(--accent-color)',
                padding: '4px 12px',
                borderRadius: '12px',
                fontSize: '0.825rem',
                fontWeight: 700,
                fontFamily: 'JetBrains Mono, monospace',
                display: 'inline-block'
              }}
            >
              {confidence}%
            </div>
            <span style={{ fontSize: '0.825rem', color: 'var(--text-muted)' }}>
              Statistical validation based on data consistency and diagnostic profiling.
            </span>
          </div>
        </div>
      )}

      {/* Progressive Render Section 4: Visualization Panel */}
      <div className="progressive-section" style={{ animationDelay: '0.7s' }}>
        <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 10px 0', fontWeight: 700 }}>
          Visualization
        </h4>
        {plotBase64 ? (
          <div 
            style={{ 
              background: 'var(--bg-input)', 
              borderRadius: '8px', 
              padding: '12px', 
              border: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <img 
              src={`data:image/png;base64,${plotBase64}`} 
              alt="Data Analysis Visualizer" 
              style={{ 
                maxWidth: '100%', 
                maxHeight: '320px', 
                borderRadius: '6px',
                objectFit: 'contain'
              }} 
            />
          </div>
        ) : (
          <div 
            style={{ 
              background: 'var(--bg-input)', 
              borderRadius: '8px', 
              padding: '20px', 
              border: '1px solid var(--border-color)',
              textAlign: 'center',
              color: 'var(--text-muted)',
              fontSize: '0.825rem'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '22px', color: 'rgba(255,255,255,0.15)', marginBottom: '4px' }}>
              insert_chart_off
            </span>
            <div>No visualization generated for this query.</div>
          </div>
        )}
      </div>

      {/* Progressive Render Section 5: Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="progressive-section" style={{ animationDelay: '0.9s' }}>
          <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', margin: '0 0 10px 0', fontWeight: 700 }}>
            Recommendations
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {recommendations.map((rec, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                <span 
                  className="material-symbols-outlined" 
                  style={{ 
                    fontSize: '16px', 
                    color: 'var(--primary-color)',
                    marginTop: '2px'
                  }}
                >
                  lightbulb
                </span>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.45' }}>
                  {rec}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progressive Render Section 6: Suggested Follow-up Questions */}
      {suggested_follow_ups && suggested_follow_ups.length > 0 && (
        <div className="progressive-section" style={{ animationDelay: '1.1s', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
          <div 
            style={{ 
              fontSize: '0.75rem', 
              color: 'var(--text-muted)', 
              fontWeight: 600, 
              textTransform: 'uppercase', 
              letterSpacing: '0.05em',
              marginBottom: '10px' 
            }}
          >
            Suggested Follow-up
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {suggested_follow_ups.map((qText, idx) => (
              <button
                key={idx}
                onClick={() => onSelectFollowUp(qText)}
                style={{
                  background: 'rgba(59, 130, 246, 0.05)',
                  border: '1px solid rgba(59, 130, 246, 0.15)',
                  borderRadius: '16px',
                  padding: '5px 12px',
                  color: 'var(--primary-color)',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  fontFamily: 'Inter, sans-serif'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.12)';
                  e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.35)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.05)';
                  e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.15)';
                }}
              >
                {qText}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Progressive Render Section 7: Collapsible Reasoning Trace */}
      {reasoning_trace && reasoning_trace.length > 0 && (
        <div className="progressive-section" style={{ animationDelay: '1.3s', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
          <div 
            onClick={() => setTraceOpen(!traceOpen)}
            style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              cursor: 'pointer',
              color: 'var(--text-muted)'
            }}
          >
            <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
              Reasoning Trace
            </span>
            <span className="material-symbols-outlined" style={{ fontSize: '18px', transform: traceOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
              expand_more
            </span>
          </div>

          {traceOpen && (
            <div 
              className="fade-in"
              style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '8px', 
                marginTop: '12px',
                paddingLeft: '8px',
                borderLeft: '2px solid rgba(59, 130, 246, 0.15)'
              }}
            >
              {reasoning_trace.map((step, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span 
                    className="material-symbols-outlined" 
                    style={{ 
                      fontSize: '14px', 
                      color: 'var(--accent-color)',
                      fontWeight: 'bold'
                    }}
                  >
                    done
                  </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
