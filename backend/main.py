from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import json
from agent import agent_instance as agent
from analyzer import auto_eda, generate_plots
from reporting import get_failures, save_report, list_reports, get_report

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASTORE = {}

# Analysis Cache: Stores pre-computed reports for instant access
# Structure: { "why": "Report Text...", "fix": "Report Text..." }
ANALYSIS_CACHE = {}

import threading

def run_background_analysis(df, machine_name):
    """
    Runs key analyses in the background so they are ready when requested.
    """
    print("Background Analysis Started...")
    
    # Initialize placeholders - This prevents duplicate runs
    ANALYSIS_CACHE['why'] = "Analyzing..."
    ANALYSIS_CACHE['impact'] = "Analyzing..."
    ANALYSIS_CACHE['fix'] = "Analyzing..."
    
    # 1. Root Cause (Why)
    # Hybrid Mode: Logic + RAG
    print("Pre-computing Root Cause...")
    try:
        agent.set_df(df, context_data={"machine_name": machine_name})
        prompt_why = "Diagnose the root cause of the identified failures using the manuals."
        ANALYSIS_CACHE['why'] = agent.run(prompt_why)
        print("Root Cause Computed.")
    except Exception as e:
        print(f"Error computing Root Cause: {e}")
        ANALYSIS_CACHE['why'] = f"Analysis Failed: {str(e)}"
    
    # 2. Impact Assessment
    print("Pre-computing Impact...")
    try:
        prompt_impact = "What is the operational impact of these failures?"
        ANALYSIS_CACHE['impact'] = agent.run(prompt_impact)
        print("Impact Computed.")
    except Exception as e:
        print(f"Error computing Impact: {e}")
        ANALYSIS_CACHE['impact'] = f"Analysis Failed: {str(e)}"

    # 3. Repair Guide (Fix)
    print("Pre-computing Repair Guide...")
    try:
        prompt_fix = "Provide step-by-step repair instructions for this issue."
        ANALYSIS_CACHE['fix'] = agent.run(prompt_fix)
        print("Repair Guide Computed.")
    except Exception as e:
        print(f"Error computing Repair: {e}")
        ANALYSIS_CACHE['fix'] = f"Analysis Failed: {str(e)}"
    
    print("Background Analysis Complete! Cache populated.")

class Query(BaseModel):
    question: str

from fastapi import Form
from typing import Optional

@app.post("/upload")
def upload_csv(file: UploadFile = File(...), machine_name: Optional[str] = Form(None)):
    try:
        df = pd.read_csv(file.file)
        # Basic sanitization: strip whitespace from headers
        df.columns = df.columns.str.strip()
        DATASTORE["df"] = df
        DATASTORE["machine_name"] = machine_name # Store the context
        
        # Clear old cache
        ANALYSIS_CACHE.clear()
        
        # Start Background Analysis Thread
        thread = threading.Thread(target=run_background_analysis, args=(df, machine_name))
        thread.daemon = True # Ensure thread dies if server stops
        thread.start()
        
        # Calculate true failures
        failure_count = 0
        possible_cols = ["Machine failure", "Failure", "Target", "failure", "target"]
        found_col = next((c for c in possible_cols if c in df.columns), None)
        
        if found_col:
            failure_count = int(df[found_col].sum())
        else:
            # Fallback for unlabeled data (assume row count if unsure, but user said it's wrong)
            # Better to return 0 or explicitly say standard rows if no failure col found.
            # Let's default to row count ONLY if no failure column exists, but label it "Rows"
            failure_count = df.shape[0] 

        return {
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "rows": df.shape[0],
            "failure_count": failure_count,
            "columns": df.shape[1]
        }
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}


from analyzer import auto_eda, generate_plots, clean_for_json

@app.get("/eda")
def get_eda():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No dataset has been uploaded"}
    return auto_eda(df)

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
        if not file.filename.endswith(".pdf"):
            return {"error": "Only PDF files are supported."}
            
        file_path = os.path.join("backend", "manuals", file.filename)
        
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
    
    # Update Agent Environment
    agent.set_df(df, context_data={"machine_name": DATASTORE.get("machine_name")})
    
    # Run Agent Loop
    answer = agent.run(query.question)
    return {"answer": answer}

@app.get("/auto_analysis")
def auto_analysis():
    # ... restored previously ...
    df = DATASTORE.get("df")
    agent.set_df(df, context_data={"machine_name": DATASTORE.get("machine_name")})
    prompt = "Perform a comprehensive reliability analysis..."
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

@app.get("/failures")
def get_failure_list():
    df = DATASTORE.get("df")
    if df is None:
        return {"error": "No data loaded"}
    
    failures = get_failures(df)
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
