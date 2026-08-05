import React, { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import AIResponseCard from './AIResponseCard';
import SuggestedPrompts from './SuggestedPrompts';
import PromptInput from './PromptInput';
import LoadingAnalysis from './LoadingAnalysis';

// Dynamic prompt compiler from metadata
export function generateDatasetPrompts(activeDataset, domainProfile) {
  const target = activeDataset?.target_column || domainProfile?.target_column || '';
  const numeric = activeDataset?.numeric_cols || [];
  const categorical = activeDataset?.categorical_cols || [];
  const analysisType = domainProfile?.analysis_type || '';
  const domain = domainProfile?.domain || '';

  const prompts = [];

  // 1. Target column specific prompts
  if (target && target !== 'None' && target !== 'Not Defined') {
    prompts.push(`Which features influence ${target} the most?`);
    prompts.push(`Show ${target} distribution.`);
    
    if (categorical.length > 0) {
      const catCol = categorical.find(c => c.toLowerCase() !== target.toLowerCase()) || categorical[0];
      prompts.push(`Compare ${target} across different groups of ${catCol}.`);
    }
  }

  // 2. Domain & Analysis Type specific prompts
  if (domain && domain !== 'dataset' && domain !== 'General Analysis' && domain !== 'Domain detecting...') {
    prompts.push(`Summarize dataset health for this ${domain} domain.`);
  } else {
    prompts.push(`Summarize dataset health and data quality.`);
  }

  if (analysisType && analysisType !== 'general') {
    prompts.push(`Explain feature importance for our ${analysisType} task.`);
  } else {
    prompts.push(`Show feature importance relative to the target.`);
  }

  // 3. Numeric correlations & distributions
  if (numeric.length >= 2) {
    const numCols = numeric.filter(c => c.toLowerCase() !== target.toLowerCase());
    if (numCols.length >= 2) {
      prompts.push(`Explain the strongest correlations between ${numCols[0]} and ${numCols[1]}.`);
    } else {
      prompts.push(`Explain the strongest correlations among numerical columns.`);
    }
  }

  // 4. Missing values & anomalies
  const hasMissing = activeDataset?.missing_values && Object.values(activeDataset.missing_values).some(v => v > 0);
  if (hasMissing) {
    prompts.push(`Explain missing values and recommend a cleanup strategy.`);
  }

  const hasOutliers = activeDataset?.outliers && Object.values(activeDataset.outliers).some(v => v > 0);
  if (hasOutliers) {
    prompts.push(`Find anomalies and analyze outlier patterns.`);
  }

  prompts.push(`Generate executive summary.`);

  return Array.from(new Set(prompts)).slice(0, 6);
}

export default function ConversationPanel({ 
  activeDataset,
  domainProfile,
  conversationHistory, 
  loading, 
  apiFinished,
  onLoadingComplete,
  plots, 
  targetColumn, 
  pinnedInsightIds, 
  onSend, 
  onPinToggle, 
  onNewConversation,
  downloadPDF,
  drawerOpen,
  onToggleDrawer
}) {
  const bottomRef = useRef(null);
  const scrollContainerRef = useRef(null);

  // Auto-scroll to bottom on message list updates or loading state changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversationHistory, loading]);

  // Copy plain text conversation transcript
  const handleCopyConversation = () => {
    if (conversationHistory.length === 0) return;
    const text = conversationHistory.map(msg => {
      if (msg.role === 'user') {
        return `YOU:\n${msg.content}\n`;
      } else {
        return `ANALYST.AI:\n${msg.analysis}\n`;
      }
    }).join('\n---\n\n');
    navigator.clipboard.writeText(text);
    alert("Conversation transcript copied to clipboard!");
  };

  // Export transcript as Markdown file
  const handleExportMarkdown = () => {
    if (conversationHistory.length === 0) return;
    const text = conversationHistory.map(msg => {
      if (msg.role === 'user') {
        return `## User Question\n\n${msg.content}\n`;
      } else {
        return `## Analyst.AI Insight\n\n${msg.analysis}\n`;
      }
    }).join('\n\n---\n\n');
    
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analyst_ai_conversation_${activeDataset?.filename || 'export'}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Export transcript as Client-side PDF print preview
  const handleExportPDF = () => {
    if (conversationHistory.length === 0) return;
    
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;

    const datasetName = activeDataset?.filename || 'Dataset';
    const timestamp = new Date().toLocaleString();

    let transcriptHtml = `
      <html>
        <head>
          <title>Analyst.AI Conversation Transcript - ${datasetName}</title>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
              color: #1a202c;
              padding: 40px;
              line-height: 1.6;
            }
            .header {
              border-bottom: 2px solid #e2e8f0;
              padding-bottom: 20px;
              margin-bottom: 30px;
            }
            .title {
              font-size: 24px;
              font-weight: 700;
              margin: 0 0 8px 0;
              color: #3b82f6;
            }
            .meta {
              font-size: 13px;
              color: #718096;
            }
            .message {
              margin-bottom: 28px;
              padding-bottom: 24px;
              border-bottom: 1px solid #edf2f7;
            }
            .role-user {
              font-weight: 700;
              color: #2d3748;
              font-size: 14px;
              text-transform: uppercase;
              letter-spacing: 0.05em;
              margin-bottom: 6px;
            }
            .role-assistant {
              font-weight: 700;
              color: #3b82f6;
              font-size: 14px;
              text-transform: uppercase;
              letter-spacing: 0.05em;
              margin-bottom: 6px;
            }
            .content {
              font-size: 15px;
              white-space: pre-wrap;
            }
            .section-title {
              font-weight: 700;
              font-size: 12px;
              color: #718096;
              text-transform: uppercase;
              letter-spacing: 0.05em;
              margin-top: 14px;
              margin-bottom: 6px;
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 class="title">Analyst.AI Conversation Transcript</h1>
            <div class="meta">
              Dataset: ${datasetName} | Date: ${timestamp}
            </div>
          </div>
    `;

    conversationHistory.forEach(msg => {
      if (msg.role === 'user') {
        transcriptHtml += `
          <div class="message">
            <div class="role-user">You</div>
            <div class="content">${msg.content}</div>
          </div>
        `;
      } else {
        transcriptHtml += `
          <div class="message">
            <div class="role-assistant">Analyst.AI</div>
            <div class="content">${msg.analysis}</div>
        `;
        
        if (msg.evidence && msg.evidence.length > 0) {
          transcriptHtml += `
            <div class="section-title">Evidence Columns</div>
            <div style="font-size: 13px;">${msg.evidence.join(', ')}</div>
          `;
        }
        
        if (msg.recommendations && msg.recommendations.length > 0) {
          transcriptHtml += `
            <div class="section-title">Recommendations</div>
            <ul style="margin: 4px 0; padding-left: 20px; font-size: 14px;">
              ${msg.recommendations.map(r => `<li>${r}</li>`).join('')}
            </ul>
          `;
        }
        
        transcriptHtml += `</div>`;
      }
    });

    transcriptHtml += `
          <script>
            window.onload = function() {
              window.print();
              window.close();
            }
          </script>
        </body>
      </html>
    `;

    printWindow.document.write(transcriptHtml);
    printWindow.document.close();
  };

  // Pin latest assistant insight to memory
  const handlePinLatestInsight = () => {
    const latestAi = [...conversationHistory].reverse().find(msg => msg.role === 'assistant');
    if (latestAi) {
      onPinToggle(latestAi);
    } else {
      alert("No AI insights generated yet to pin.");
    }
  };

  // Resolve dynamic suggested prompts based on context
  const latestAssistantMsg = [...conversationHistory]
    .reverse()
    .find(msg => msg.role === 'assistant');

  const suggestedPrompts = conversationHistory.length === 0
    ? generateDatasetPrompts(activeDataset, domainProfile)
    : (latestAssistantMsg?.suggested_follow_ups || []);

  return (
    <div 
      style={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%',
        backgroundColor: 'var(--bg-body)',
        boxSizing: 'border-box',
        overflow: 'hidden'
      }}
    >
      {/* 0. Enterprise Conversation Toolbar */}
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          padding: '14px 24px', 
          borderBottom: '1px solid var(--border-color)',
          background: 'var(--bg-sidebar)',
          flexWrap: 'wrap',
          gap: '12px'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)' }}>
            forum
          </span>
          <span style={{ fontWeight: 600, fontFamily: 'Geist', fontSize: '0.925rem', color: 'var(--text-main)' }}>
            Dataset Intelligence Workspace
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            onClick={onNewConversation}
            title="Start New Conversation"
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s ease'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>add</span>
            New
          </button>

          <button
            onClick={handleCopyConversation}
            title="Copy conversation transcript"
            disabled={conversationHistory.length === 0}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: conversationHistory.length === 0 ? 'var(--text-muted)' : 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: conversationHistory.length === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              opacity: conversationHistory.length === 0 ? 0.5 : 1
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>content_copy</span>
            Copy
          </button>

          <button
            onClick={handleExportMarkdown}
            title="Export conversation to Markdown"
            disabled={conversationHistory.length === 0}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: conversationHistory.length === 0 ? 'var(--text-muted)' : 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: conversationHistory.length === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              opacity: conversationHistory.length === 0 ? 0.5 : 1
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>download</span>
            Markdown
          </button>

          <button
            onClick={handleExportPDF}
            title="Export conversation PDF"
            disabled={conversationHistory.length === 0}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: conversationHistory.length === 0 ? 'var(--text-muted)' : 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: conversationHistory.length === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              opacity: conversationHistory.length === 0 ? 0.5 : 1
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>picture_as_pdf</span>
            PDF
          </button>

          <button
            onClick={handlePinLatestInsight}
            title="Pin latest AI response card"
            disabled={conversationHistory.length === 0}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: conversationHistory.length === 0 ? 'var(--text-muted)' : 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: conversationHistory.length === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              opacity: conversationHistory.length === 0 ? 0.5 : 1
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>push_pin</span>
            Pin Latest
          </button>

          <button
            onClick={onNewConversation}
            title="Clear all messages in chat"
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: 'var(--text-main)',
              fontSize: '0.8rem',
              fontWeight: 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>delete_sweep</span>
            Clear
          </button>

          {!drawerOpen && (
            <button
              onClick={onToggleDrawer}
              title="Open Dataset Intelligence"
              style={{
                background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.2)',
                borderRadius: '6px',
                padding: '6px 12px',
                color: 'var(--primary-color)',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>analytics</span>
              Dataset Intel
            </button>
          )}
        </div>
      </div>

      {/* 1. Scrollable Message Feed */}
      <div 
        ref={scrollContainerRef}
        style={{ 
          flex: 1, 
          overflowY: 'auto', 
          padding: '24px 32px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <div style={{ width: '100%', maxWidth: '800px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {conversationHistory.length === 0 && !loading && (
            <div 
              style={{ 
                margin: '80px auto', 
                textAlign: 'center', 
                color: 'var(--text-muted)',
                maxWidth: '440px',
                padding: '2rem 0'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '48px', color: 'rgba(59,130,246,0.25)', marginBottom: '14px' }}>
                forum
              </span>
              <h3 style={{ fontFamily: 'Geist', margin: '0 0 8px 0', fontSize: '1.25rem', color: 'var(--text-main)', fontWeight: 600 }}>
                Dataset Intelligence Workspace
              </h3>
              <p style={{ fontSize: '0.875rem', lineHeight: '1.55', margin: 0 }}>
                Ask question chips below or enter natural queries to execute statistical tests, find correlations, scan anomalies, and compile boardroom-ready summaries.
              </p>
            </div>
          )}
    
          {conversationHistory.map((msg) => {
            if (msg.role === 'user') {
              return <MessageBubble key={msg.id} message={msg} />;
            } else {
              return (
                <AIResponseCard 
                  key={msg.id} 
                  responseData={msg} 
                  plots={plots} 
                  onSelectFollowUp={onSend}
                  onPinInsight={() => onPinToggle(msg)}
                  isPinned={pinnedInsightIds.includes(msg.id)}
                  activeDataset={activeDataset}
                  domainProfile={domainProfile}
                />
              );
            }
          })}
    
          {loading && (
            <LoadingAnalysis 
              apiFinished={apiFinished} 
              onComplete={onLoadingComplete} 
            />
          )}
          <div ref={bottomRef} />
        </div>
      </div>
  
      {/* 2. Suggestions & Input Panel */}
      <div 
        style={{ 
          padding: '16px 32px 24px 32px', 
          borderTop: '1px solid var(--border-color)',
          background: 'var(--bg-card)',
          backdropFilter: 'blur(20px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <div style={{ width: '100%', maxWidth: '800px' }}>
          {suggestedPrompts.length > 0 && (
            <SuggestedPrompts 
              prompts={suggestedPrompts} 
              onSelectPrompt={onSend} 
            />
          )}
          
          <PromptInput 
            onSend={onSend} 
            loading={loading}
            onAttachDataset={() => onSend("Summarize dataset health.")}
            onAttachReport={() => onSend("Generate executive summary.")}
          />
        </div>
      </div>
    </div>
  );
}
