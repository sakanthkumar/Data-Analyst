from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import json
import io
import re
from uuid import uuid4
from typing import Optional
from agent import agent_instance as agent
from analyzer import auto_eda, generate_plots, clean_for_json, get_failure_stats, get_correlation_stats, TargetAnalysisEngine, find_target_column
from reporting import get_failures, save_report, list_reports, get_report, generate_pdf_report

from logging_config import logger

app = FastAPI()

# Add CORS middleware with production-safe origins
raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://localhost")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
allow_all = "*" in origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else origins,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime

def validate_startup_config():
    """Fail-fast validation for startup configuration and runtime environments."""
    backend_provider = os.getenv("LLM_BACKEND", "groq").lower()
    timeout = os.getenv("LLM_REQUEST_TIMEOUT", "60")
    logger.info(f"Validating startup configuration for LLM_BACKEND={backend_provider}...")
    
    if backend_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            logger.error("GROQ_API_KEY is missing while LLM_BACKEND is set to 'groq'.")
            raise RuntimeError("Configuration Error: GROQ_API_KEY environment variable is required.")
        reasoning_model = os.getenv("GROQ_MODEL_REASONING", "llama-3.3-70b-versatile").strip()
        code_model = os.getenv("GROQ_MODEL_CODE", "llama-3.3-70b-versatile").strip()
        
        logger.info(f"✓ Provider: groq")
        logger.info(f"✓ Models: Reasoning={reasoning_model}, Code={code_model}")
        logger.info(f"✓ Timeout: {timeout}s")
        logger.info(f"✓ Configuration Valid")
    elif backend_provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        if not ollama_url:
            logger.error("OLLAMA_BASE_URL is missing while LLM_BACKEND is set to 'ollama'.")
            raise RuntimeError("Configuration Error: OLLAMA_BASE_URL environment variable is required.")
        reasoning_model = os.getenv("OLLAMA_REASONING_MODEL", "llama3").strip()
        code_model = os.getenv("OLLAMA_CODE_MODEL", "deepseek-coder:6.7b").strip()
        
        logger.info(f"✓ Provider: ollama")
        logger.info(f"✓ Endpoint: {ollama_url}")
        logger.info(f"✓ Models: Reasoning={reasoning_model}, Code={code_model}")
        logger.info(f"✓ Timeout: {timeout}s")
        logger.info(f"✓ Configuration Valid")
    else:
        logger.error(f"Unsupported LLM_BACKEND: '{backend_provider}'. Supported: 'groq', 'ollama'.")
        raise RuntimeError(f"Configuration Error: Unsupported LLM_BACKEND '{backend_provider}'.")
    
    reports_dir = os.getenv("REPORTS_DIR", "reports")
    os.makedirs(reports_dir, exist_ok=True)

@app.on_event("startup")
def startup_event():
    validate_startup_config()

@app.get("/health")
def health_check():
    """Standard health check endpoint for AWS ALB, ECS, EC2, and Docker."""
    return {
        "status": "healthy",
        "service": "Data Analyst Agent",
        "provider": os.getenv("LLM_BACKEND", "groq"),
        "reasoning_model": agent.llm_service.analysis_model,
        "coding_model": agent.llm_service.code_model,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected server error occurred."}
    )

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path.rstrip("/")
    if request.method == "GET" and path in ["/eda", "/eda_plots", "/analysis/report", "/failures", "/domain_profile"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Analysis Cache: Stores pre-computed reports for instant access
# Structure: { "why": "Report Text...", "fix": "Report Text..." }
# Analysis Cache: Stores pre-computed reports for instant access
# Structure: { "why": "Report Text...", "fix": "Report Text..." }
ANALYSIS_CACHE = {}

DATASTORE = {}

# Store for user-defined acronyms
DATASTORE["acronyms"] = {}
DATASTORE["profiling_status"] = "idle"
DATASTORE["report_generation_status"] = "idle"
DATASTORE["domain_profile"] = None

# Secure Upload Lifecycle state
DATASTORE["dataset_session_id"] = None
DATASTORE["executive_report"] = None
DATASTORE["highlighted_records"] = None
DATASTORE["target_analysis"] = None
DATASTORE["driver_analysis"] = None
DATASTORE["report_cache"] = {}
DATASTORE["chat_history"] = []
DATASTORE["background_analysis_results"] = None
DATASTORE["target_column"] = None

def secure_filename(filename: str) -> str:
    """
    Sanitizes filenames to prevent path traversal attacks.
    Excludes all directory components and filters non-alphanumeric/dot/hyphen/underscore.
    """
    basename = os.path.basename(filename)
    basename = re.sub(r'[^a-zA-Z0-9._-]', '_', basename)
    basename = basename.lstrip('.')
    if not basename:
        basename = "uploaded_file"
    return basename

import threading
from tools import search_web



def run_background_profiling_and_analysis(df, machine_name, start_analysis_flag, session_id):
    """
    Runs the DomainAgent profiling in the background, updates status,
    and optionally triggers the background analysis thread.
    """
    global DATASTORE
    logger.info(f"[Background] Profiling Started for session {session_id}...")
    if DATASTORE.get("dataset_session_id") != session_id:
        logger.info(f"[Background] Profiling aborted: session mismatch. Active: {DATASTORE.get('dataset_session_id')}, Task: {session_id}")
        return
        
    try:
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        sample_rows = df.head(3).to_dict(orient="records")
        
        profile = agent.profile_dataset(columns, dtypes, sample_rows)
        
        if DATASTORE.get("dataset_session_id") != session_id:
            logger.info("[Background] Profiling discarded: session mismatch after profile computation.")
            return
            
        DATASTORE["domain_profile"] = profile
        DATASTORE["profiling_status"] = "completed"
        from datetime import datetime
        DATASTORE["domain_profile_timestamp"] = datetime.now().isoformat()
        
        logger.info(
            f"[DomainAgent] Dataset Ingested and Profiling Completed: "
            f"Domain={profile.get('domain')}, Confidence={profile.get('confidence')}, AnalysisType={profile.get('analysis_type')}"
        )
    except Exception as e:
        if DATASTORE.get("dataset_session_id") != session_id:
            logger.info("[Background] Profiling exception discarded: session mismatch.")
            return
        logger.error(f"Error during DomainAgent profiling: {e}", exc_info=True)
        DATASTORE["profiling_status"] = "failed"
        
    if start_analysis_flag:
        run_background_analysis(df, machine_name, session_id)


def explain_correlation(feature: str, val: float, target_col: str, df: pd.DataFrame) -> str:
    direction = "positive" if val > 0 else "negative"
    
    # Try to determine a friendly semantic meaning
    meaning = ""
    feature_lower = feature.lower()
    target_lower = target_col.lower()
    
    # Check for passenger class (Pclass)
    if "pclass" in feature_lower or "passenger class" in feature_lower:
        if val < 0:
            meaning = f"As class number increases (from first class to third class), {target_col} decreases (first-class passengers had higher rates)."
        else:
            meaning = f"As class number increases (from first class to third class), {target_col} increases."
            
    # Check for Age
    elif "age" in feature_lower:
        if val < 0:
            meaning = f"Older age is associated with lower {target_col} rates (survival/target decreases as age increases)."
        else:
            meaning = f"Older age is associated with higher {target_col} rates (survival/target increases as age increases)."
            
    # Check for Fare
    elif "fare" in feature_lower:
        if val > 0:
            meaning = f"Passengers who paid higher fares had higher {target_col} rates."
        else:
            meaning = f"Passengers who paid higher fares had lower {target_col} rates."
            
    # Check for sibling/spouse/parents/children count (SibSp, Parch)
    elif "sibsp" in feature_lower or "parch" in feature_lower:
        if val < 0:
            meaning = f"Having more family members onboard is associated with a lower {target_col} rate."
        else:
            meaning = f"Having more family members onboard is associated with a higher {target_col} rate."
            
    # General / Fallback explanations
    if not meaning:
        is_binary = False
        if df is not None and feature in df.columns:
            unique_vals = df[feature].dropna().unique()
            if len(unique_vals) <= 2:
                is_binary = True
                
        if is_binary:
            meaning = f"Presence/higher value of '{feature}' is {'positively' if val > 0 else 'negatively'} associated with '{target_col}'."
        else:
            meaning = f"As '{feature}' increases, '{target_col}' tends to {'increase' if val > 0 else 'decrease'}."
            
    return f"""Feature: {feature}
Correlation: {val:.2f}
Direction: {direction}
Business Meaning: {meaning}"""


def run_background_analysis(df, machine_name, session_id):
    """
    Runs key analyses in the background so they are ready when requested.
    """
    global DATASTORE, ANALYSIS_CACHE
    if DATASTORE.get("dataset_session_id") != session_id:
        print(f"[Background] Analysis aborted: session mismatch on start. Active: {DATASTORE.get('dataset_session_id')}, Task: {session_id}")
        return
        
    print(f"Background Analysis Started for session {session_id}...")
    DATASTORE["report_generation_status"] = "running"
    
    # Initialize placeholders
    ANALYSIS_CACHE['why'] = "Analyzing..."
    ANALYSIS_CACHE['impact'] = "Analyzing..."
    ANALYSIS_CACHE['fix'] = "Analyzing..."
    
    # 1. Driver Analysis (Why)
    print("Pre-computing Driver Analysis...")
    try:
        agent.set_df(df, context_data={"machine_name": machine_name})
        
        # Build Statistical Context
        target_override = DATASTORE.get("target_column")
        f_stats = TargetAnalysisEngine.get_target_stats(df, target_override=target_override)
        c_stats = get_correlation_stats(df, target_override=target_override)
        
        if DATASTORE.get("dataset_session_id") != session_id:
            print("[Background] Analysis discarded: session mismatch after statistical checks.")
            return

        if "error" in f_stats:
            msg = f"Analysis Skipped: {f_stats['error']}"
            if DATASTORE.get("dataset_session_id") != session_id:
                print("[Background] Analysis discarded: session mismatch.")
                return
            ANALYSIS_CACHE['combined'] = msg
            ANALYSIS_CACHE['why'] = msg
            ANALYSIS_CACHE['impact'] = msg
            ANALYSIS_CACHE['fix'] = msg
            DATASTORE["report_generation_status"] = "completed"
        elif f_stats["total_failures"] == 0:
            msg = "No highlighted target instances detected. Driver analysis not required."
            if DATASTORE.get("dataset_session_id") != session_id:
                print("[Background] Analysis discarded: session mismatch.")
                return
            ANALYSIS_CACHE['combined'] = msg
            ANALYSIS_CACHE['why'] = msg
            ANALYSIS_CACHE['impact'] = msg
            ANALYSIS_CACHE['fix'] = msg
            DATASTORE["report_generation_status"] = "completed"
        else:
            target_col = f_stats["target_column"]
            target_type = f_stats["target_type"]
            # Build Knowledge Context (Definitions ONLY)
            knowledge_context = ""
            if f_stats["modes"]:
                knowledge_context += "\nSemantic Definitions:\n"
                for mode in f_stats["modes"]:
                    name = mode['name']
                    definition = None
                    source = None
                    
                    # 1. Manuals/Documents
                    rag_hits = kb.search_manuals(f"What does term {name} mean?", k=1)
                    if rag_hits:
                        definition = rag_hits[0]
                        source = "Documents"
                    
                    # 2. User Acronyms
                    if not definition and name in DATASTORE.get("acronyms", {}):
                        definition = DATASTORE["acronyms"][name]
                        source = "User Definition"
                        
                    # 3. Web Search
                    if not definition:
                        try:
                            web_res = search_web(f"meaning of {name} in the context of {target_col}")
                            if web_res:
                                definition = web_res[:200]
                                source = "Web Search"
                        except: pass
                            
                    if definition:
                        knowledge_context += f"- **{name}**: {definition} [Source: {source}]\n"
                    else:
                        knowledge_context += f"- **{name}**: No semantic definition available.\n"

            if DATASTORE.get("dataset_session_id") != session_id:
                print("[Background] Analysis discarded: session mismatch after definitions search.")
                return

            # Construct Combined Prompt (Data + Knowledge)
            prompt_failure = f"""
TASK: Perform a complete Target Variable Analysis.

DATA CONTEXT:
Target Variable: {target_col} ({target_type})
Dataset summary:
- Total records: {f_stats['total_records']}
- Highlighted target instances: {f_stats['total_failures']}

Target category breakdown:
"""
            for m in f_stats["modes"]:
                prompt_failure += f"- {m['name']}: {m['count']} ({m['percent']:.1f}%)\n"

            prompt_failure += "\nStatistical observations:\n"
            if f_stats["modes"]:
                prompt_failure += f"- Most frequent indicator: {f_stats['modes'][0]['name']}\n"
            
            prompt_failure += "\nCorrelation insights:\n"
            if "top_correlations" in c_stats and c_stats["top_correlations"]:
                for item in c_stats["top_correlations"]:
                    explanation = explain_correlation(item['feature'], item['value'], target_col, df)
                    prompt_failure += explanation + "\n\n"

            prompt_failure += knowledge_context

            # RAG for strategy (Global search)
            hits = kb.search_manuals(f"Actionable strategies and response guide for {target_col}", k=3)
            if hits:
                prompt_failure += "\nREFERENCE EXCERPTS:\n" + "\n".join(hits)

            # SINGLE CALL
            domain_profile = DATASTORE.get("domain_profile")
            if not domain_profile:
                columns = df.columns.tolist()
                domain_profile = agent.domain_agent._generate_fallback_profile(columns)
                
            target_col = f_stats["target_column"]
            columns = df.columns.tolist()
            kpis = domain_profile.get("recommended_kpis", [])
            correlations = c_stats.get("top_correlations", [])
            
            selected_prompt = agent.llm_service.get_executive_report_prompt(
                domain_profile=domain_profile,
                target_column=target_col,
                columns=columns,
                kpis=kpis,
                correlations=correlations
            )
            
            if DATASTORE.get("dataset_session_id") != session_id:
                print("[Background] Analysis discarded: session mismatch before LLM execution.")
                return

            full_report = agent.generate_direct(prompt_failure, system_type="failure", system_prompt=selected_prompt)
            
            if DATASTORE.get("dataset_session_id") != session_id:
                print("[Background] Analysis discarded: session mismatch after LLM execution.")
                return

            # Store in cache (all keys point to valid report to support legacy endpoints)
            ANALYSIS_CACHE['combined'] = full_report
            ANALYSIS_CACHE['why'] = full_report
            ANALYSIS_CACHE['impact'] = full_report
            ANALYSIS_CACHE['fix'] = full_report
            
            # Populate lifecycle-specific attributes in DATASTORE
            DATASTORE["executive_report"] = full_report
            DATASTORE["driver_analysis"] = full_report
            DATASTORE["target_analysis"] = full_report
            DATASTORE["background_analysis_results"] = {
                "combined": full_report,
                "why": full_report,
                "impact": full_report,
                "fix": full_report
            }
            
        print("Failure Analysis Computed (Combined).")
        DATASTORE["report_generation_status"] = "completed"
    except Exception as e:
        if DATASTORE.get("dataset_session_id") != session_id:
            print("[Background] Analysis exception update discarded: session mismatch.")
            return
        print(f"Error computing Failure Analysis: {e}")
        ANALYSIS_CACHE['combined'] = f"Analysis Failed: {str(e)}"
        ANALYSIS_CACHE['why'] = f"Analysis Failed: {str(e)}"
        ANALYSIS_CACHE['impact'] = f"Analysis Failed: {str(e)}"
        ANALYSIS_CACHE['fix'] = f"Analysis Failed: {str(e)}"
        DATASTORE["report_generation_status"] = "failed"
    
    print("Background Analysis Complete! Cache populated.")

class Query(BaseModel):
    question: str

class AcronymPayload(BaseModel):
    acronyms: dict

@app.post("/settings/acronyms")
def update_acronyms(payload: AcronymPayload):
    DATASTORE["acronyms"].update(payload.acronyms)
    return {"message": "Acronyms updated", "total": len(DATASTORE["acronyms"])}

@app.get("/settings/acronyms/unknown")
def get_unknown_acronyms():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset loaded"}
        
    target_override = DATASTORE.get("target_column")
    stats = TargetAnalysisEngine.get_target_stats(df, target_override=target_override)
    if "error" in stats:
        return {"unknown": []}
        
    unknown_candidates = []
    known = DATASTORE.get("acronyms", {})
    
    for m in stats.get("modes", []):
        name = m["name"]
        if name not in known:
            unknown_candidates.append(name)
            
    # Apply semantic filtering using LLM
    unknown = agent.filter_acronyms(unknown_candidates)
    return {"unknown": unknown}

@app.post("/upload")
def upload_csv(file: UploadFile = File(...), machine_name: Optional[str] = Form(None)):
    try:
        # Validate extension
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .csv files are supported.")

        # Validate file size (20MB)
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds size limit. Maximum allowed size is 20MB.")

        # Verify CSV headers / parse preview
        try:
            df_preview = pd.read_csv(file.file, nrows=2)
            file.file.seek(0)
            if df_preview.empty or len(df_preview.columns) == 0:
                raise ValueError("CSV has no columns or rows.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format or corrupted file headers: {str(e)}")

        df = pd.read_csv(file.file)
        # Basic sanitization: strip whitespace from headers
        df.columns = df.columns.str.strip()
        
        sanitized_filename = secure_filename(file.filename)
        
        DATASTORE["df"] = df
        DATASTORE["machine_name"] = machine_name # Store the context
        DATASTORE["filename"] = sanitized_filename
        DATASTORE["file_size_bytes"] = size  # Actual file size in bytes
        
        # Generate new session ID
        dataset_session_id = str(uuid4())
        DATASTORE["dataset_session_id"] = dataset_session_id
        DATASTORE["target_column"] = None
        
        # Clear agent memory & update agent df immediately
        agent.dataset_session_id = dataset_session_id
        agent.memory = []
        agent.set_df(df, context_data={"machine_name": machine_name})
        
        # Clear old domain profile & cache, set profiling status to idle
        DATASTORE["domain_profile"] = None
        DATASTORE["domain_profile_timestamp"] = None
        DATASTORE["profiling_status"] = "idle"
        DATASTORE["report_generation_status"] = "idle"
        DATASTORE["acronyms"] = {}
        ANALYSIS_CACHE.clear()
        
        # Clear audit lifecycle artifacts
        DATASTORE["executive_report"] = None
        DATASTORE["highlighted_records"] = None
        DATASTORE["target_analysis"] = None
        DATASTORE["driver_analysis"] = None
        DATASTORE["report_cache"] = {}
        DATASTORE["chat_history"] = []
        DATASTORE["background_analysis_results"] = None
        
        # Guess the target column
        detected_target = find_target_column(df)
        possible_cols = [
            "Target", "target", "label", "Label", "y", "Machine failure", "Failure", "failure", 
            "Survived", "survived", "churn", "Churn", "default", "Default", "class", "Class",
            "output", "Output", "response", "Response", "clicked", "Clicked", "decision", "Decision"
        ]
        is_priority = detected_target in possible_cols or (detected_target and detected_target.lower() in [c.lower() for c in possible_cols])
        confidence = 0.95 if is_priority else 0.50
        candidate_targets = df.columns.tolist()

        return {
            "message": "File uploaded successfully.",
            "filename": sanitized_filename,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "detected_target": detected_target,
            "confidence": confidence,
            "candidate_targets": candidate_targets,
            "session_id": dataset_session_id
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        DATASTORE["profiling_status"] = "failed"
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

@app.post("/analysis/start")
def start_analysis():
    df = DATASTORE.get("df")
    machine_name = DATASTORE.get("machine_name")
    session_id = DATASTORE.get("dataset_session_id")
    
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
        
    # Set status to running so frontend starts polling it
    DATASTORE["profiling_status"] = "running"
    
    # Check if already running? (Optional optimization)
    # We just restart/overwrite the thread. Python threads can't be killed easily, 
    # but run_background_analysis checks cache keys so it might overlap.
    # However, for this single-user local app, it's fine.
    
    thread = threading.Thread(
        target=run_background_profiling_and_analysis, 
        args=(df, machine_name, True, session_id)
    )
    thread.daemon = True
    thread.start()
    
    return {"message": "Analysis started", "status": "started"}



from analyzer import auto_eda, generate_plots, clean_for_json

@app.get("/eda")
def get_eda():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset has been uploaded"}
    res = auto_eda(df)
    res["filename"] = DATASTORE.get("filename")
    res["file_size_bytes"] = DATASTORE.get("file_size_bytes")  # Actual upload size
    res["target_column"] = DATASTORE.get("target_column")    # Confirmed target
    res["dataset_id"] = DATASTORE.get("dataset_session_id")
    return res

@app.get("/domain_profile")
def get_domain_profile():
    status = DATASTORE.get("profiling_status", "idle")
    profile = DATASTORE.get("domain_profile")
    if status == "completed" and profile is not None:
        res = dict(profile)
        res["status"] = "completed"
        res["timestamp"] = DATASTORE.get("domain_profile_timestamp")
        return res
    return {
        "status": status,
        "error": "No domain profile loaded"
    }


@app.get("/eda_plots")
def get_eda_plots():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset has been uploaded"}
    try:
        plots = generate_plots(df)
        return plots
    except Exception as e:
        return {"error": str(e)}

@app.get("/data")
def get_data(page: int = 1, limit: int = 50):
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset has been uploaded"}
    
    start = (page - 1) * limit
    end = start + limit
    
    # Slice and clean
    subset = df.iloc[start:end]
    data = subset.to_dict(orient="records")
    return {
        "page": page,
        "limit": limit,
        "total_rows": len(df),
        "data": clean_for_json(data)
    }

import time
from fastapi import HTTPException

import os
import shutil
from knowledge import kb

LAST_CHAT_TIME = 0

@app.post("/manuals/upload")
async def upload_manual(file: UploadFile = File(...)):
    try:
        # Validate extension
        if not file.filename.lower().endswith(".pdf"):
            return {"error": "Invalid file type. Only PDF files are supported."}
            
        # Check file size (10MB)
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > 10 * 1024 * 1024:
            return {"error": "File exceeds size limit. Maximum allowed size is 10MB."}
            
        # Validate PDF signature
        pdf_sig = file.file.read(4)
        file.file.seek(0)
        if pdf_sig != b'%PDF':
            return {"error": "Invalid PDF format or corrupted file headers."}
            
        sanitized_name = secure_filename(file.filename)
        manuals_dir = os.path.join("backend", "manuals")
        os.makedirs(manuals_dir, exist_ok=True)
        file_path = os.path.join(manuals_dir, sanitized_name)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest into Knowledge Base (RAG)
        success, message = kb.ingest_manual(file_path)
        
        if success:
            return {"message": f"Manual uploaded and indexed: {message}"}
        else:
            return {"error": f"Upload successful but indexing failed: {message}"}
            
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

@app.get("/manuals")
def list_manuals():
    try:
        manuals_dir = os.path.join("backend", "manuals")
        if not os.path.exists(manuals_dir):
            return []
        return [f for f in os.listdir(manuals_dir) if f.endswith(".pdf")]
    except Exception as e:
        return {"error": str(e)}

def structure_chat_response(question: str, raw_answer: str, columns: list) -> dict:
    prompt = f"""
Given a user's analytical question, the raw markdown response from the data analyst assistant, and the dataset columns:
User Question: {question}
Raw Answer: {raw_answer}
Dataset Columns: {columns}

Structure the response into a JSON object matching the following fields. Do not fabricate statistics, facts, or charts. Only extract or translate what is present or logically implied:
- "analysis" (string): The primary explainable text response/explanation. Format nicely as markdown without duplicating other sections.
- "evidence" (array of strings): A list of column names from the Dataset Columns list that are relevant, discussed, or serve as evidence for this answer.
- "confidence" (integer): A confidence percentage (between 50 and 100) reflecting how well the data supports the answer. If the answer is unsure or data is missing, score it lower.
- "visualization_type" (string or null): The key of a visualization that would best represent this answer. Choose from "heatmap" (if correlations or multi-variable associations are discussed), or "dist_<col_name>" (if a specific numeric column's distribution is relevant, e.g. "dist_Age"), or null. It must match one of the active column names.
- "recommendations" (array of strings): 2-3 data-driven recommendations or business action items derived directly from this analysis.
- "suggested_follow_ups" (array of strings): 2-3 short contextual follow-up question chips the user can click next (e.g. "Explain [Column]", "Compare [Column]", "Show correlations").
- "reasoning_trace" (array of strings): 3-5 steps of analysis executed to produce this response (e.g. "Inspected schema", "Analyzed target drivers", "Evaluated correlations", "Synthesized recommendations").

Ensure you return ONLY valid JSON. Avoid any text before or after the JSON.
"""
    try:
        res = agent.generate_direct(prompt, system_type="analysis")
        # Clean response
        cleaned = res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end != -1:
            import json
            structured = json.loads(cleaned[start:end])
            
            # Sanitize visualization_type
            viz = structured.get("visualization_type")
            if viz and viz != "heatmap" and not viz.startswith("dist_"):
                # Clean up if it's just a column name
                if viz in columns:
                    structured["visualization_type"] = f"dist_{viz}"
                else:
                    structured["visualization_type"] = None
            
            # Ensure evidence columns are valid
            ev = structured.get("evidence", [])
            structured["evidence"] = [c for c in ev if c in columns]
            
            return structured
    except Exception as e:
        print(f"Error structuring chat response: {e}")
        
    # Fallback structure if LLM fails or doesn't return JSON
    return {
        "analysis": raw_answer,
        "evidence": [c for c in columns if c.lower() in question.lower() or c.lower() in raw_answer.lower()][:3],
        "confidence": 90,
        "visualization_type": "heatmap" if "correlation" in raw_answer.lower() else None,
        "recommendations": ["Inspect data columns further", "Refine analytical queries"],
        "suggested_follow_ups": [f"Explain {columns[0]}" if columns else "Show feature importance"],
        "reasoning_trace": ["Schema inspected", "Query executed", "Insight generated"]
    }

@app.post("/chat")
def chat(query: Query):
    global LAST_CHAT_TIME
    current_time = time.time()
    
    # 5-second rate limit
    if current_time - LAST_CHAT_TIME < 5:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please wait 5 seconds."
        )
    
    LAST_CHAT_TIME = current_time

    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset has been uploaded"}
    
    # Verify/Attach session id to agent chat memory
    active_session_id = DATASTORE.get("dataset_session_id")
    if getattr(agent, "dataset_session_id", None) != active_session_id:
        agent.memory = []
        agent.dataset_session_id = active_session_id
        DATASTORE["chat_history"] = []
    
    # Update Agent Environment
    agent.set_df(df, context_data={"machine_name": DATASTORE.get("machine_name")})
    
    # Run Agent Loop
    answer = agent.run(query.question, chat_history=list(DATASTORE.get("chat_history", [])))
    
    # Save to chat history
    if DATASTORE.get("chat_history") is None:
        DATASTORE["chat_history"] = []
    DATASTORE["chat_history"].append({"role": "user", "content": query.question})
    DATASTORE["chat_history"].append({"role": "assistant", "content": answer})
    
    # Sync agent memory
    agent.memory = list(DATASTORE["chat_history"])
    
    # Structure response
    structured = structure_chat_response(query.question, answer, df.columns.tolist())
    # Ensure raw answer is also present for backward compatibility
    structured["answer"] = answer
    
    return structured

@app.get("/auto_analysis")
def auto_analysis():
    # ... restored previously ...
    df = DATASTORE.get("df")
    agent.set_df(df, context_data={"machine_name": DATASTORE.get("machine_name")})
    prompt = "Perform a comprehensive target variable driver and impact analysis..."
    report = agent.run(prompt)
    return {"report": report}

@app.get("/analysis/fast_failure")
def fast_failure_analysis():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No data loaded"}
    
    from analyzer import analyze_failure_modes
    report = analyze_failure_modes(df)
    return {"answer": report}

@app.get("/analysis/report")
def get_cached_report(type: str = "why"):
    """
    Returns the pre-computed analysis from the cache.
    Types: 'why' (Root Cause), 'impact' (Impact), 'fix' (Repair)
    """
    if type in ANALYSIS_CACHE:
        answer = ANALYSIS_CACHE[type]
        if answer == "Analyzing...":
             return {"answer": "Background analysis in progress. Please wait...", "status": "pending"}
        elif "Analysis Failed" in answer:
             return {"answer": answer, "status": "error"}
        else:
             return {"answer": answer, "status": "ready"}
    else:
        # Cache missing entirely - means upload never happened or server restarted
        return {"answer": "No analysis data found. Please re-upload CSV.", "status": "error"}

@app.get("/analysis/status")
def get_analysis_status():
    """
    Returns the current background report generation status.
    """
    return {"status": DATASTORE.get("report_generation_status", "idle")}

class ConfirmTargetPayload(BaseModel):
    target_column: str

@app.post("/analysis/confirm_target")
def confirm_target(payload: ConfirmTargetPayload):
    df = DATASTORE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
        
    target_column = payload.target_column
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found in dataset")
        
    DATASTORE["target_column"] = target_column
    
    # Calculate stats with override
    stats = TargetAnalysisEngine.get_target_stats(df, target_override=target_column)
    
    # Identify unknown acronyms for this target column
    known = DATASTORE.get("acronyms", {})
    unknown_candidates = []
    for m in stats.get("modes", []):
        name = m["name"]
        if name not in known:
            unknown_candidates.append(name)
            
    # Apply semantic filtering using LLM
    unknown = agent.filter_acronyms(unknown_candidates)
    
    return {
        "message": "Target column confirmed.",
        "target_column": target_column,
        "unknown_acronyms": unknown,
        "status": "waiting_for_definitions" if unknown else "ready_to_start"
    }

@app.get("/reports/export/pdf")
def export_report_pdf():
    report_text = DATASTORE.get("executive_report")
    if not report_text:
        raise HTTPException(status_code=400, detail="No executive report generated yet. Please run analysis first.")
        
    context_name = DATASTORE.get("machine_name") or DATASTORE.get("filename") or "Generic Dataset"
    
    domain_profile = DATASTORE.get("domain_profile") or {}
    domain_name = domain_profile.get("domain") or "General Analysis"
    
    target_column = DATASTORE.get("target_column") or "N/A"
    
    pdf_buffer = io.BytesIO()
    try:
        generate_pdf_report(
            pdf_buffer,
            context_name=context_name,
            domain_name=domain_name,
            target_column=target_column,
            report_text=report_text
        )
        pdf_buffer.seek(0)
        
        # Clean context_name for filename
        safe_ctx = re.sub(r'[^a-zA-Z0-9_-]', '_', context_name)
        filename = f"Analyst_AI_Executive_Report_{safe_ctx}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/failures")
def get_failure_list():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No data loaded"}
    
    failures = get_failures(df)
    DATASTORE["highlighted_records"] = failures
    return {"failures": failures}

@app.post("/reports/save")
def save_current_report(analysis_type: str = Body(..., embed=True)):
    df = DATASTORE.get("df")
    machine_name = DATASTORE.get("machine_name")
    
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
        
    report_id, msg = save_report(df, machine_name, analysis_type)
    return {"id": report_id, "message": msg}

@app.get("/reports")
def get_all_reports():
    return list_reports()

@app.get("/reports/{report_id}")
def get_single_report(report_id: str):
    data = get_report(report_id)
    if data:
        return data
    raise HTTPException(status_code=404, detail="Report not found")

# --- Settings API ---

@app.get("/settings/config")
def get_settings_config():
    conf = agent.get_config()
    conf["rag_depth"] = kb.n_results
    return conf

@app.get("/settings/models")
def get_settings_models():
    return {"models": agent.get_available_models()}

class ModelUpdate(BaseModel):
    model: str

@app.post("/settings/model")
def update_settings_model(update: ModelUpdate):
    msg = agent.set_model(update.model)
    return {"message": msg, "current_model": agent.model}

class TempUpdate(BaseModel):
    temperature: float

@app.post("/settings/temperature")
def update_settings_temp(update: TempUpdate):
    msg = agent.set_temperature(update.temperature)
    return {"message": msg, "temperature": agent.temperature}

@app.post("/manuals/clear")
def clear_manuals_kb():
    success, msg = kb.clear_index()
    if success:
        return {"message": msg}
    else:
        raise HTTPException(status_code=500, detail=msg)

class ExpertConfig(BaseModel):
    system_prompt: str = None
    ollama_url: str = None

@app.post("/settings/expert")
def update_expert_settings(config: ExpertConfig):
    msg = agent.set_config(config.dict(exclude_none=True))
    return {"message": msg}

class RagUpdate(BaseModel):
    n_results: int

@app.post("/settings/rag")
def update_rag_settings(update: RagUpdate):
    msg = kb.set_depth(update.n_results)
    return {"message": msg, "depth": kb.n_results}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
