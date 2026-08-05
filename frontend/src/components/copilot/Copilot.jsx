import React, { useState } from 'react';
import EmptyCopilotState from './EmptyCopilotState';
import ConversationPanel from './ConversationPanel';
import DatasetIntelligenceDrawer from './DatasetIntelligenceDrawer';
import { api } from '../../services/api';

export default function Copilot({ 
  activeDataset, 
  domainProfile, 
  plots, 
  onTabChange,
  conversationHistory,
  setConversationHistory,
  pinnedInsights,
  setPinnedInsights,
  recentQuestions,
  setRecentQuestions,
  downloadPDF
}) {
  const [loading, setLoading] = useState(false);
  const [apiFinished, setApiFinished] = useState(false);
  const [pendingResponse, setPendingResponse] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(true);

  // Handle message sending
  const handleSend = async (qText) => {
    if (!qText.trim() || loading) return;

    // 1. Add user question to history
    const userMsgId = `user-${Date.now()}`;
    const userMsg = {
      id: userMsgId,
      role: 'user',
      content: qText
    };

    setConversationHistory(prev => [...prev, userMsg]);

    // 2. Add to recent questions list
    setRecentQuestions(prev => {
      const filtered = prev.filter(q => q !== qText);
      return [qText, ...filtered].slice(0, 10);
    });

    setLoading(true);
    setApiFinished(false);
    setPendingResponse(null);

    const startTime = Date.now();

    try {
      const res = await api.postChat(qText);
      const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

      // 3. Prepare assistant structured card response payload (to be appended when animation finishes)
      const assistantMsg = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        question: qText,
        analysis: res.data.analysis || res.data.answer || "No response received.",
        evidence: res.data.evidence || [],
        confidence: res.data.confidence || 90,
        visualization_type: res.data.visualization_type || null,
        recommendations: res.data.recommendations || [],
        suggested_follow_ups: res.data.suggested_follow_ups || [],
        reasoning_trace: res.data.reasoning_trace || [],
        processingTime: processingTime,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setPendingResponse(assistantMsg);
      setApiFinished(true);
    } catch (e) {
      const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);
      const errorMsg = {
        id: `ai-error-${Date.now()}`,
        role: 'assistant',
        isError: true,
        question: qText,
        errorDetails: e.response?.data?.detail || e.message || "Failed to connect to the analysis backend.",
        processingTime: processingTime,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setPendingResponse(errorMsg);
      setApiFinished(true);
    }
  };

  const handleLoadingComplete = () => {
    if (pendingResponse) {
      setConversationHistory(prev => [...prev, pendingResponse]);
      setPendingResponse(null);
      setLoading(false);
      setApiFinished(false);
    }
  };

  const handleNewConversation = () => {
    setConversationHistory([]);
  };

  const handlePinInsight = (insight) => {
    setPinnedInsights(prev => {
      if (prev.some(p => p.id === insight.id)) return prev;
      return [...prev, insight];
    });
  };

  const handleUnpinInsight = (insightId) => {
    setPinnedInsights(prev => prev.filter(insight => insight.id !== insightId));
  };

  const handlePinToggle = (msg) => {
    const isPinned = pinnedInsights.some(p => p.id === msg.id);
    if (isPinned) {
      handleUnpinInsight(msg.id);
    } else {
      handlePinInsight({
        id: msg.id,
        question: msg.question,
        analysis: msg.analysis || msg.answer
      });
    }
  };

  // 1. Empty State
  if (!activeDataset) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <EmptyCopilotState onTabChange={onTabChange} />
      </div>
    );
  }

  // 2. Three-Panel Workspace Layout
  return (
    <div 
      style={{ 
        display: 'flex', 
        height: 'calc(100vh - 64px)', // Deducts header height
        width: '100%',
        overflow: 'hidden',
        boxSizing: 'border-box'
      }}
    >
      {/* Center Panel: AI Conversation Workspace */}
      <ConversationPanel 
        activeDataset={activeDataset}
        domainProfile={domainProfile}
        conversationHistory={conversationHistory}
        loading={loading}
        apiFinished={apiFinished}
        onLoadingComplete={handleLoadingComplete}
        plots={plots}
        targetColumn={activeDataset.target_column || domainProfile?.target_column}
        pinnedInsightIds={pinnedInsights.map(p => p.id)}
        onSend={handleSend}
        onPinToggle={handlePinToggle}
        onNewConversation={handleNewConversation}
        downloadPDF={downloadPDF}
        drawerOpen={drawerOpen}
        onToggleDrawer={() => setDrawerOpen(!drawerOpen)}
      />

      {/* Right Panel: Collapsible Dataset Intelligence Panel */}
      {drawerOpen && (
        <DatasetIntelligenceDrawer 
          activeDataset={activeDataset}
          domainProfile={domainProfile}
          conversationHistory={conversationHistory}
          recentQuestions={recentQuestions}
          pinnedInsights={pinnedInsights}
          onSelectQuestion={handleSend}
          onUnpinInsight={handleUnpinInsight}
          onClose={() => setDrawerOpen(false)}
        />
      )}
    </div>
  );
}
