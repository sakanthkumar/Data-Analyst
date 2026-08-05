import axios from 'axios';
import API_BASE from '../config';

const client = axios.create({
  baseURL: API_BASE
});

export const api = {
  // Uploads
  uploadDataset: (formData) => client.post('/upload', formData),
  
  // Chat
  postChat: (question) => client.post('/chat', { question }),
  
  // Data Grid
  getLogs: (page, limit) => client.get(`/data?page=${page}&limit=${limit}`),
  
  // Reports History
  getReportsList: () => client.get('/reports'),
  getReportDetail: (id) => client.get(`/reports/${id}`),
  
  // Manuals (RAG)
  getManuals: () => client.get('/manuals'),
  uploadManual: (formData) => client.post('/manuals/upload', formData),
  
  // Settings
  getConfig: () => client.get('/settings/config'),
  getModels: () => client.get('/settings/models'),
  updateModel: (model) => client.post('/settings/model', { model }),
  updateTemperature: (temperature) => client.post('/settings/temperature', { temperature }),
  clearKnowledgeBase: () => client.post('/manuals/clear'),
  updateExpertSettings: (systemPrompt, ollamaUrl) => client.post('/settings/expert', { system_prompt: systemPrompt, ollama_url: ollamaUrl }),
  updateRagSettings: (ragDepth) => client.post('/settings/rag', { n_results: ragDepth }),
  
  // Dashboard / Analysis
  getDomainProfile: () => client.get(`/domain_profile?t=${Date.now()}`),
  getFailures: () => client.get(`/failures?t=${Date.now()}`),
  saveReportHistory: (analysisType) => client.post('/reports/save', { analysis_type: analysisType }),
  startAnalysis: () => client.post('/analysis/start'),
  confirmTarget: (targetColumn) => client.post('/analysis/confirm_target', { target_column: targetColumn }),
  updateAcronyms: (acronyms) => client.post('/settings/acronyms', { acronyms }),
  exportPDF: () => client.get('/reports/export/pdf', { responseType: 'blob' }),
  getEDA: () => client.get(`/eda?t=${Date.now()}`),
  getEDAPlots: () => client.get(`/eda_plots?t=${Date.now()}`),
  getFastFailureReport: () => client.get(`/analysis/fast_failure?t=${Date.now()}`),
  getAnalysisReport: (type) => client.get(`/analysis/report?type=${type}&t=${Date.now()}`)
};
