import os
import json
import uuid
from datetime import datetime
import pandas as pd

REPORTS_DIR = "reports"

# Ensure reports directory exists
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

def get_failures(df: pd.DataFrame):
    """
    Extracts rows where failure occurred.
    """
    possible_cols = ["Machine failure", "Failure", "Target", "failure", "target"]
    found_col = next((c for c in possible_cols if c in df.columns), None)
    
    if found_col:
        # Filter where value is 1 (True)
        failure_df = df[df[found_col] == 1]
    else:
        # If no explicit column, return empty
        failure_df = pd.DataFrame()
        
    # Convert to list of dicts for JSON
    # Use standard date format for timestamp if exists
    if not failure_df.empty:
        # Limit to top 1000 to prevent huge payloads
        return failure_df.head(1000).to_dict(orient="records")
    return []

def save_report(df: pd.DataFrame, machine_name: str, analysis_type: str = "Manual Scan"):
    """
    Saves a snapshot of failures to a JSON file.
    """
    failures = get_failures(df)
    
    if not failures:
        return None, "No failures found to save."

    report_id = str(uuid.uuid4())
    filename = f"{REPORTS_DIR}/{report_id}.json"
    
    report_data = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "machine_name": machine_name or "Unknown Machine",
        "analysis_type": analysis_type,
        "total_failures": len(failures),
        "failures": failures
    }
    
    with open(filename, "w") as f:
        json.dump(report_data, f, indent=2)
        
    return report_id, "Report saved successfully."

def list_reports():
    """
    Lists all saved reports (metadata only).
    """
    reports = []
    if not os.path.exists(REPORTS_DIR):
        return []
        
    for f in os.listdir(REPORTS_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(REPORTS_DIR, f), "r") as file:
                    data = json.load(file)
                    reports.append({
                        "id": data.get("id"),
                        "timestamp": data.get("timestamp"),
                        "machine_name": data.get("machine_name"),
                        "analysis_type": data.get("analysis_type"),
                        "total_failures": data.get("total_failures")
                    })
            except:
                pass
                
    # Sort by timestamp desc
    reports.sort(key=lambda x: x["timestamp"], reverse=True)
    return reports

def get_report(report_id: str):
    """
    Retrieves full report details.
    """
    filename = f"{REPORTS_DIR}/{report_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None
