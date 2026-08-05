import { useEffect, useState, useRef } from "react";
import Upload from "./Upload";
import Copilot from "./components/copilot/Copilot";
import DatasetLibrary from "./pages/DatasetLibrary";
import Manuals from "./Manuals";
import Sidebar from "./Sidebar";
import Header from "./Header";
import Reports from './Reports';
import Settings from './Settings';
import "./App.css";
import { api } from "./services/api";
import TargetModal from "./TargetModal";
import AcronymModal from "./AcronymModal";
import FailureModal from "./FailureModal";

// Cognitive Enterprise Layout & Component imports
import AppLayout from "./components/AppLayout";
import AnalysisWorkspace from "./pages/AnalysisWorkspace";
import DashboardSkeleton from "./components/skeleton/DashboardSkeleton";
import GlassCard from "./components/GlassCard";
import LandingPage from "./pages/LandingPage";
import AuthPage from "./pages/AuthPage";
import WelcomeDashboard from "./pages/WelcomeDashboard";

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [plots, setPlots] = useState(null);
  const [reports, setReports] = useState({});
  const [reportLoading, setReportLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showFailures, setShowFailures] = useState(false);
  const [failures, setFailures] = useState([]);
  const [domainProfile, setDomainProfile] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);

  // Lifted Copilot states
  const [conversationHistory, setConversationHistory] = useState([]);
  const [pinnedInsights, setPinnedInsights] = useState([]);
  const [recentQuestions, setRecentQuestions] = useState([]);
  const lastLoadedFilenameRef = useRef(null);

  // Resolve dataset specific memory key
  const datasetId = data?.dataset_id || data?.filename || 'empty';
  const storageKey = datasetId !== 'empty' ? `copilot_memory_${datasetId}` : null;

  // Load dataset-specific memory on change
  useEffect(() => {
    if (storageKey) {
      try {
        const savedRaw = localStorage.getItem(storageKey);
        if (savedRaw) {
          const parsed = JSON.parse(savedRaw);
          setConversationHistory(parsed.conversationHistory || []);
          setPinnedInsights(parsed.pinnedInsights || []);
          setRecentQuestions(parsed.recentQuestions || []);
        } else {
          setConversationHistory([]);
          setPinnedInsights([]);
          setRecentQuestions([]);
        }
      } catch (e) {
        console.error("Failed to load dataset memory", e);
        setConversationHistory([]);
        setPinnedInsights([]);
        setRecentQuestions([]);
      }
      lastLoadedFilenameRef.current = data?.filename;
    } else {
      setConversationHistory([]);
      setPinnedInsights([]);
      setRecentQuestions([]);
      lastLoadedFilenameRef.current = null;
    }
  }, [storageKey, data?.filename]);

  // Persist dataset-specific memory on updates
  useEffect(() => {
    if (storageKey && lastLoadedFilenameRef.current === data?.filename) {
      const payload = {
        conversationHistory,
        pinnedInsights,
        recentQuestions,
        lastOpened: Date.now()
      };
      localStorage.setItem(storageKey, JSON.stringify(payload));
    }
  }, [conversationHistory, pinnedInsights, recentQuestions, storageKey, data?.filename]);

  // Authentication & Navigation view states
  // eslint-disable-next-line no-unused-vars
  const [isAuthenticated, setIsAuthenticated] = useState(() => localStorage.getItem('isAuthenticated') === 'true');
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [view, setView] = useState(() => {
    return localStorage.getItem('isAuthenticated') === 'true' ? 'workspace' : 'landing';
  });
  const [authState, setAuthState] = useState('login');

  const handleAuthSuccess = (userData) => {
    setIsAuthenticated(true);
    setUser(userData);
    localStorage.setItem('isAuthenticated', 'true');
    localStorage.setItem('user', JSON.stringify(userData));
    setView('workspace');
  };

  const handleLogout = () => {
    // Purge dataset memories from local storage
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('copilot_memory_')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(k => localStorage.removeItem(k));

    setIsAuthenticated(false);
    setUser(null);
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('user');

    // Clear React states
    setConversationHistory([]);
    setPinnedInsights([]);
    setRecentQuestions([]);
    if (lastLoadedFilenameRef.current) {
      lastLoadedFilenameRef.current = null;
    }

    setView('landing');
    setActiveTab('dashboard');
  };

  // Target Selection State
  const [showTargetModal, setShowTargetModal] = useState(false);
  const [detectedTarget, setDetectedTarget] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [candidateTargets, setCandidateTargets] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState("");

  // Polling reference
  const pollingTimeoutRef = useRef(null);

  // Centralized polling function
  const startDomainProfilePolling = () => {
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
      pollingTimeoutRef.current = null;
      console.log("pollDomainProfile stop (cleared previous)");
    }

    console.log("pollDomainProfile start");

    const poll = async () => {
      try {
        const profileRes = await api.getDomainProfile();
        const status = profileRes.data.status;
        console.log("pollDomainProfile tick, status:", status);

        if (status === "completed") {
          setDomainProfile(profileRes.data);
          pollingTimeoutRef.current = null;
          console.log("pollDomainProfile stop (completed)");
        } else if (status === "failed") {
          console.error("Dataset profiling failed on the backend.");
          setDomainProfile(null);
          pollingTimeoutRef.current = null;
          console.log("pollDomainProfile stop (failed)");
        } else {
          // "running" or "idle" (waiting to start) - keep polling
          setDomainProfile(null);
          pollingTimeoutRef.current = setTimeout(poll, 2000);
        }
      } catch (e) {
        console.error("Failed to poll domain profile:", e);
        pollingTimeoutRef.current = setTimeout(poll, 2000);
      }
    };

    poll();
  };

  const handleUploadStart = () => {
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
      pollingTimeoutRef.current = null;
      console.log("pollDomainProfile stop (upload started)");
    }
    console.log("setData invocation: null (upload start)");
    setData(null);
    setPlots(null);
    setReports({});
    setFailures([]);
    setShowFailures(false);
    setDomainProfile(null);
    setUnknownAcronyms([]);
    setAcronymInputs({});
    setReportLoading(false);
    setDatasetLoading(true);
    
    // Clear target state
    setShowTargetModal(false);
    setDetectedTarget("");
    setConfidence(0);
    setCandidateTargets([]);
    setSelectedTarget("");
  };

  const loadFailures = async () => {
    try {
      const res = await api.getFailures();
      if (res.data.failures) {
        setFailures(res.data.failures);
        setShowFailures(true);
        // Auto-save report for history
        await api.saveReportHistory("Target Driver Scan");
      } else {
        alert("No highlights detected in the dataset!");
      }
    } catch (e) { console.error(e); }
  };


  const [showAcronymModal, setShowAcronymModal] = useState(false);
  const [unknownAcronyms, setUnknownAcronyms] = useState([]);
  const [acronymInputs, setAcronymInputs] = useState({});

  const startAnalysis = async () => {
    try {
      await api.startAnalysis();
      // Optional: Add toast "Analysis Started"
      console.log("Analysis started in background");
    } catch (e) {
      console.error("Failed to start analysis", e);
    }
  };

  const handleUploadSuccess = (uploadData) => {
    // Clear old state immediately to prevent stale data display
    console.log("setData invocation: null (upload success)");
    setData(null);
    setPlots(null);
    setReports({});
    setFailures([]);
    setShowFailures(false);
    setDomainProfile(null);
    setDatasetLoading(false);

    setDetectedTarget(uploadData.detected_target);
    setConfidence(uploadData.confidence);
    setCandidateTargets(uploadData.candidate_targets || []);
    setSelectedTarget(uploadData.detected_target);
    setShowTargetModal(true);
  };

  const handleConfirmTarget = async () => {
    try {
      setShowTargetModal(false);
      setDatasetLoading(true);
      const res = await api.confirmTarget(selectedTarget);
      
      // Fetch EDA to populate page immediately after confirming target
      fetchEDA();
      setActiveTab('dashboard');

      if (res.data.unknown_acronyms && res.data.unknown_acronyms.length > 0) {
        setUnknownAcronyms(res.data.unknown_acronyms);
        setShowAcronymModal(true);
      } else {
        // No missing definitions, start immediately
        startAnalysis();
      }
    } catch (e) {
      alert("Failed to confirm target column selection.");
      setDatasetLoading(false);
    }
  };

  const handleAcronymSubmit = async () => {
    try {
      await api.updateAcronyms(acronymInputs);
      setShowAcronymModal(false);
      alert("Definitions saved! Analysis starting...");
      startAnalysis();
    } catch (e) { alert("Failed to save definitions"); }
  };

  const handleAcronymSkip = () => {
    setShowAcronymModal(false);
    startAnalysis();
  };

  const downloadPDF = async () => {
    try {
      const response = await api.exportPDF();
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Analyst_AI_Executive_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      alert("Failed to download PDF report. Please ensure the analysis has completed.");
    }
  };

  // Data Fetching
  const fetchEDA = async () => {
    console.log("fetchEDA start");
    setReportLoading(false); // No auto-loading
    setDatasetLoading(true);
    
    let isDatasetLoaded = false;

    // 1. Fetch EDA stats
    try {
      const edaRes = await api.getEDA();
      if (edaRes.data.error) {
        console.log("setData invocation: null (fetchEDA error)");
        setData(null);
      } else {
        console.log("setData invocation:", edaRes.data);
        setData(edaRes.data);
        isDatasetLoaded = true;
      }
    } catch (e) {
      console.error("Failed to fetch EDA data:", e);
      console.log("setData invocation: null (fetchEDA catch)");
      setData(null);
    }

    // 2. Fetch EDA plots in background
    try {
      const plotsRes = await api.getEDAPlots();
      if (plotsRes.data.error) {
        setPlots(null);
      } else {
        setPlots(plotsRes.data);
      }
    } catch (e) {
      console.error("Failed to fetch EDA plots:", e);
      setPlots(null);
    }

    // 3. Poll Domain Profile if dataset is loaded
    if (isDatasetLoaded) {
      startDomainProfilePolling();
    } else {
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
        console.log("pollDomainProfile stop (no dataset loaded)");
      }
      setDomainProfile(null);
    }

    setDatasetLoading(false);
    console.log("fetchEDA complete");
  };

  useEffect(() => { 
    fetchEDA(); 
    return () => {
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
        console.log("pollDomainProfile stop (unmount)");
      }
    };
  // fetchEDA is intentionally omitted — it must only run once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Log dashboard visibility state changes
  useEffect(() => {
    const isVisible = data !== null && activeTab === 'dashboard';
    console.log("dashboard visibility state changes:", isVisible ? "VISIBLE" : "HIDDEN");
  }, [data, activeTab]);

  // Manual Analysis Handler
  const runAnalysis = async (type) => {
    setReportLoading(true);
    // Don't clear previous reports, just add/update

    try {
      if (type === 'what') {
        // ULTRA-FAST PATH: Use deterministic Python analysis
        const res = await api.getFastFailureReport();
        if (res.data.answer) {
          setReports(prev => ({ ...prev, 'Target Driver Scan': res.data.answer }));
        }
        loadFailures(); // Auto-open logs
      } else {
        // CACHED PATH: Fetch pre-computed analysis
        const res = await api.getAnalysisReport(type);

        const titles = { why: "Executive Insights Report", impact: "Executive Insights Report", fix: "Executive Insights Report" };
        const title = titles[type] || "Analysis Report";

        if (res.data.answer) {
          setReports(prev => ({ ...prev, [title]: res.data.answer }));
        }
      }
    } catch (e) {
      alert("Analysis failed. Please check backend connection.");
    } finally {
      setReportLoading(false);
    }
  };

  // View Components
  const renderDashboard = () => {
    if (datasetLoading) return (
      <div style={{ padding: '24px' }}>
        <DashboardSkeleton />
      </div>
    );

    if (!data) return (
      <WelcomeDashboard
        user={user}
        onNavigateToTab={setActiveTab}
        handleUploadStart={handleUploadStart}
        handleUploadSuccess={handleUploadSuccess}
      />
    );

    return (
      <AnalysisWorkspace
        data={data}
        domainProfile={domainProfile}
        plots={plots}
        runAnalysis={runAnalysis}
        loadFailures={loadFailures}
        reports={reports}
        reportLoading={reportLoading}
        downloadPDF={downloadPDF}
        onNavigateToTab={setActiveTab}
      />
    );
  };

  if (view === 'landing') {
    return (
      <LandingPage 
        onNavigateToAuth={(state) => {
          setAuthState(state);
          setView('auth');
        }}
      />
    );
  }

  if (view === 'auth') {
    return (
      <AuthPage 
        defaultState={authState} 
        onAuthSuccess={handleAuthSuccess}
        onNavigateHome={() => setView('landing')}
      />
    );
  }

  const headerTitleMap = {
    dashboard: 'Analysis Workspace',
    copilot: 'AI Copilot',
    logs: 'Dataset Library',
    reports: 'Advanced Insights',
    manuals: 'Reference Documents',
    analysis: 'New Analysis',
    settings: 'Settings',
  };

  return (
    <AppLayout
      sidebar={<Sidebar activeTab={activeTab} onTabChange={setActiveTab} domainProfile={domainProfile} />}
      header={<Header title={headerTitleMap[activeTab] ?? activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} domainProfile={domainProfile} user={user} onLogout={handleLogout} />}
    >
      {activeTab === 'dashboard' && renderDashboard()}
      {activeTab === 'analysis' && (
        <div style={{ maxWidth: '600px', margin: '3rem auto 0 auto' }}>
          <GlassCard style={{ padding: '32px' }}>
            <h2 style={{ fontFamily: 'Geist', margin: '0 0 10px 0', color: 'var(--text-main)' }}>New Analysis</h2>
            <p style={{ margin: '0 0 24px 0', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              Upload a new CSV dataset file to start a fresh profiling and evaluation run.
            </p>
            <Upload onUploadSuccess={handleUploadSuccess} onUploadStart={handleUploadStart} />
          </GlassCard>
        </div>
      )}
      {activeTab === 'logs' && (
        <DatasetLibrary
          user={user}
          activeDataset={data}
          activeDomainProfile={domainProfile}
          onSelectDataset={(newDataset, newProfile) => {
            setData(newDataset);
            setDomainProfile(newProfile);
          }}
          onTabChange={setActiveTab}
          handleUploadStart={handleUploadStart}
          handleUploadSuccess={handleUploadSuccess}
        />
      )}
      {activeTab === 'copilot' && (
        <Copilot 
          key={datasetId} 
          activeDataset={data} 
          domainProfile={domainProfile} 
          plots={plots}
          onTabChange={setActiveTab}
          conversationHistory={conversationHistory}
          setConversationHistory={setConversationHistory}
          pinnedInsights={pinnedInsights}
          setPinnedInsights={setPinnedInsights}
          recentQuestions={recentQuestions}
          setRecentQuestions={setRecentQuestions}
          downloadPDF={downloadPDF}
        />
      )}
      {activeTab === 'reports' && <Reports />}
      {activeTab === 'manuals' && <Manuals />}
      {activeTab === 'settings' && <Settings />}

      {/* Target Column Confirmation Modal */}
      <TargetModal
        isOpen={showTargetModal}
        onClose={() => setShowTargetModal(false)}
        detectedTarget={detectedTarget}
        confidence={confidence}
        candidateTargets={candidateTargets}
        selectedTarget={selectedTarget}
        setSelectedTarget={setSelectedTarget}
        onConfirm={handleConfirmTarget}
      />

      {/* Acronym Definition Modal */}
      <AcronymModal
        isOpen={showAcronymModal}
        onClose={handleAcronymSkip}
        unknownAcronyms={unknownAcronyms}
        acronymInputs={acronymInputs}
        setAcronymInputs={setAcronymInputs}
        onSubmit={handleAcronymSubmit}
        onSkip={handleAcronymSkip}
      />

      {/* Highlighted Records Modal */}
      <FailureModal
        isOpen={showFailures}
        onClose={() => setShowFailures(false)}
        failures={failures}
      />
    </AppLayout>
  );
}
