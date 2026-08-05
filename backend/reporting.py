import os
import json
import uuid
from datetime import datetime
import pandas as pd

from pathlib import Path

REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_failures(df: pd.DataFrame):
    """
    Extracts highlighted rows (formerly failure rows) based on target variable.
    """
    from analyzer import TargetAnalysisEngine
    return TargetAnalysisEngine.get_highlighted_records(df)

def save_report(df: pd.DataFrame, machine_name: str, analysis_type: str = "Manual Scan"):
    """
    Saves a snapshot of failures to a JSON file.
    """
    failures = get_failures(df)
    
    if not failures:
        return None, "No failures found to save."

    report_id = str(uuid.uuid4())
    filename = os.path.join(REPORTS_DIR, f"{report_id}.json")
    
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

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect
import re

def generate_pdf_report(dest_io, context_name, domain_name, target_column, report_text):
    """
    Generates a professional PDF report from the analysis markdown text.
    """
    doc = SimpleDocTemplate(
        dest_io, 
        pagesize=letter,
        rightMargin=54, 
        leftMargin=54,
        topMargin=54, 
        bottomMargin=54
    )
    story = []
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    text_color = colors.HexColor("#2D3748")      # Dark gray
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=primary_color,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=text_color,
        leading=13,
        leftIndent=15,
        spaceAfter=5
    )
    
    # 1. Title Block
    story.append(Paragraph("Analyst.AI Executive Analysis Report", title_style))
    story.append(Spacer(1, 5))
    
    # Horizontal Divider
    d = Drawing(504, 2)
    d.add(Rect(0, 0, 504, 2, fillColor=primary_color, strokeColor=None))
    story.append(d)
    story.append(Spacer(1, 10))
    
    # 2. Metadata Table
    meta_data = [
        [Paragraph("<b>Dataset/Context Name:</b>", body_style), Paragraph(context_name or "Generic Dataset", body_style)],
        [Paragraph("<b>Identified Domain:</b>", body_style), Paragraph(domain_name or "General Analysis", body_style)],
        [Paragraph("<b>Target Variable Analyzed:</b>", body_style), Paragraph(target_column or "N/A", body_style)],
    ]
    t = Table(meta_data, colWidths=[160, 344])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#EDF2F7")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # 3. Content Parsing
    if not report_text:
        story.append(Paragraph("No report content available.", body_style))
    else:
        lines = report_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple markdown cleaning
            cleaned = line
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cleaned)
            cleaned = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cleaned)
            cleaned = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', cleaned)
            
            if cleaned.startswith('# '):
                story.append(Spacer(1, 10))
                story.append(Paragraph(cleaned[2:], h1_style))
            elif cleaned.startswith('## '):
                story.append(Spacer(1, 8))
                story.append(Paragraph(cleaned[3:], h1_style))
            elif cleaned.startswith('### '):
                story.append(Spacer(1, 6))
                story.append(Paragraph(cleaned[4:], h1_style))
            elif cleaned.startswith('- ') or cleaned.startswith('* '):
                story.append(Paragraph(f"&bull; {cleaned[2:]}", bullet_style))
            else:
                story.append(Paragraph(cleaned, body_style))
                
    doc.build(story)
