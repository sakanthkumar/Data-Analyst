import requests
import pandas as pd
import os
import json
import time
import random
from dotenv import load_dotenv
from executor import execute_pandas_code
from knowledge import kb
from executor import execute_pandas_code
from knowledge import kb
from tools import search_web
from analyzer import analyze_correlations

# Load environment variables
load_dotenv()

class DataAnalystAgent:
    def __init__(self):
        """
        Part 2.3: Internal State (Memory)
        Initialize the agent with empty memory and no dataset.
        """
        self.memory = []  # Stores conversation history: [{"role": "user", "content": ...}, ...]
        self.df = None    # The 'Environment' (Dataset)
        
        # Ollama Configuration
        # Ollama Configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:1.5b"
        # Auto-download model if missing
        self._ensure_model_downloaded()

        self.system_prompt_code = """
You are an Industrial Maintenance Data Analyst.
Your goal is to write VALID, SAFE pandas code to analyze machine failure data.
The dataframe is named 'df'.

Rules:
1. ONLY write python code. No markdown, no explanation.
2. Assign the final answer/result to a variable named `result`.
3. `result` can be a string, number, dataframe, or dictionary.
4. Do NOT import os, sys, or subprocess. Use only pandas (pd) and numpy (np).
5. Use the provided column info and sample data.
"""

        self.system_prompt_analysis = """
You are a Senior reliability Engineer.
Your goal is to explain the analysis results and provide actionable maintenance recommendations.
Use the provided 'Knowledge Base Context' and 'Web Search Results' to diagnose issues and suggest fixes (`How to fix`).

Structure your answer:
1. **Analysis**: What did the data show?
2. **Diagnosis**: Why did it fail? (Use manual/knowledge base)
3. **Recommendation**: How to fix it? (Use web search/manuals)
"""

    def set_df(self, df: pd.DataFrame, context_data: dict = None):
        """Sets the current dataset (Environment) and additional context."""
        self.df = df
        self.context_data = context_data or {}

    def _ensure_model_downloaded(self):
        """Checks local Ollama for the model and downloads if missing."""
        try:
            # 1. Check existing models
            res = requests.get("http://localhost:11434/api/tags")
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                if self.model in models:
                    return # Already exists
            
            # 2. Pull if missing
            print(f"Model {self.model} not found locally. Initiating download... (Please Wait)")
            pull_res = requests.post("http://localhost:11434/api/pull", json={"name": self.model}, stream=True)
            # We iterate content to ensure we wait for completion
            for line in pull_res.iter_lines():
                pass 
            print(f"Model {self.model} downloaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not verify or download auto-download model: {e}")

    def _call_ollama(self, prompt: str):
        """Helper to call Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            # 300 second (5 min) timeout to prevent infinite hangs (Increased for 7b model CPU inference)
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            raise Exception("Ollama API timed out (300s). Model might be overloaded.")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama is not running. Please run 'ollama serve' in your terminal.")
        except Exception as e:
            raise Exception(f"Ollama API Error: {str(e)}")

    def perceive(self, question: str):
        """
        Part 2.2: Perception (Input Understanding)
        Gathers context from:
        1. Environment (DataFrame)
        2. Knowledge Base (RAG)
        3. Web Search (if needed)
        4. Internal State (Memory)
        """
        if self.df is None:
            raise ValueError("No dataframe loaded in environment.")

        # 1. Inspect Data Structure
        cols_preview = list(self.df.columns)
        dtypes = self.df.dtypes.to_dict()
        sample_head = self.df.head(1).to_string()
        if len(sample_head) > 1000:
            sample_head = sample_head[:1000] + "... (truncated)"
        
        # Smart Context Detection: Check for Machine/Device columns
        machine_col = next((col for col in cols_preview if any(x in col.lower() for x in ['machine', 'device', 'equipment', 'asset', 'robot'])), None)
        
        detected_machines = []
        if machine_col:
            # If a column exists, get the unique machines (top 10 to avoid huge context)
            try:
                unique_vals = self.df[machine_col].unique()
                detected_machines = list(unique_vals[:10])
                if len(unique_vals) > 10:
                    detected_machines.append("...")
            except:
                pass

        # Fallback to manual context
        manual_tag = self.context_data.get("machine_name")
        
        # Determine effective context
        if machine_col:
            machine_context_str = f"Dataset contains multiple machines in column '{machine_col}': {detected_machines}"
            active_machine_context = ", ".join(map(str, detected_machines[:3])) # Use first few for RAG bias
        elif manual_tag:
            machine_context_str = f"User tagged entire dataset as: {manual_tag}"
            active_machine_context = manual_tag
        else:
            machine_context_str = "No machine name detected in columns or manual tag."
            active_machine_context = "Machine"
            
        environment_context = f"""
        Context: {machine_context_str}
        Columns: {cols_preview}
        Dtypes: {dtypes}
        Sample Data:
        {sample_head}
        
        {analyze_correlations(self.df)}
        """

        # 2. RAG Retrieval (Manuals)
        # Optimization: Only search RAG if question asks for "Why", "Fix", "How", "Manual", "Code", "Error"
        # For "Identify" or "Count" or "Show", we usually just need data.
        needs_knowledge = any(w in question.lower() for w in ["why", "fix", "how", "manual", "code", "error", "cause", "diagnose", "recommend", "impact"])
        
        rag_context = "Skipped (Not required for this query)."
        if needs_knowledge:
            rag_query = question
            if active_machine_context != "Machine":
                 rag_query = f"{active_machine_context} {question}"
            
            # Additional check: If query is very short/generic, RAG might be noise.
            rag_results = kb.search_manuals(rag_query)
            rag_context = "\n".join(rag_results) if rag_results else "No relevant manuals found."

        # 3. Web Search (for fixes/errors not in manuals)
        web_context = "Skipped."
        if any(w in question.lower() for w in ["fix", "repair", "solution", "replacement", "market price"]):
             web_context = search_web(question)
        
        # 4. Retrieve Memory
        memory_context = json.dumps(self.memory[-6:]) if self.memory else "No previous context."

        combined_context = f"""
        Dataset Context:
        {environment_context}
        
        Knowledge Base (Manuals):
        {rag_context}
        
        Web Search Results:
        {web_context}

        Conversation History:
        {memory_context}

        User Question: {question}
        """
        return combined_context

    def decide(self, context: str, question: str = ""):
        """
        Part 2.4: Decision-Making (The Brain)
        Decides on the analysis steps (Code Generation).
        """
        # SPEED OPTIMIZATION: 
        # For purely qualitative/RAG questions (Explain, Repair, Impact, Diagnose with Manuals),
        # we do NOT need to generate pandas code. We can skip directly to the 'Explain' phase.
        # NOTE: "Diagnose"/ "Root Cause" needs code to find correlations, BUT we now inject correlations
        # into the context in 'perceive' (analyze_correlations). So the LLM has the data it needs.
        # We can safely skip the redundant code generation step now.
        skip_keywords = ["diagnose", "root cause", "impact", "repair", "fix", "recommend", "instruction", "manual"]
        if any(k in question.lower() for k in skip_keywords) and "calculate" not in question.lower():
             return "result = 'No analysis needed'"

        # We only need code if the user asks for data analysis. 
        # Sometimes they might just ask "How to fix error 101?", which requires no pandas code.
        # But our current architecture assumes pandas execution first. 
        # We can instruct the model to print a dummy string if no analysis is needed.
        
        prompt = f"{self.system_prompt_code}\n\nContext:\n{context}\n\nGenerate Python Code (or result='No analysis needed' if purely a RAG/Web question):"
        
        max_retries = 3
        for i in range(max_retries):
            try:
                response_text = self._call_ollama(prompt)
                code = response_text.strip().replace("```python", "").replace("```", "").strip()
                return code
            except Exception as e:
                if "Ollama is not running" in str(e): raise e
                if i == max_retries - 1: raise e
                time.sleep(1)
        return None

    def act(self, code: str):
        """
        Part 2.5: Action (Execution)
        """
        if "No analysis needed" in code:
            # The 'analyze_correlations' data is already in the Context.
            # We tell the 'explain' step to look there.
            return True, "REFER_TO_CONTEXT_STATS"
            
        success, result = execute_pandas_code(self.df, code)
        return success, result

    def explain(self, question, code, result, previous_context=None):
        """
        Explanation step (part of Action/Reporting).
        Now reuses the 'previous_context' to avoid re-fetching RAG.
        """
        
        prompt = f"""
        {self.system_prompt_analysis}
        
        User Question: {question}
        
        Context Used (Environment + RAG + Web):
        {previous_context}
        
        Analysis Code Executed:
        {code}
        
        Analysis Result:
        {result}
        
        INSTRUCTIONS:
        - If 'Analysis Result' is "REFER_TO_CONTEXT_STATS", it means the math was already done.
        - You MUST look at the 'Statistical Root Cause Analysis' section in the 'Context Used' above.
        - Use those numbers (e.g. "HDF: 58%") to answer the user.
        - Do NOT say "Refer to context". Just give the answer.
        
        Provide the final answer:
        """
        try:
            return self._call_ollama(prompt)
        except Exception:
            # Fallback if LLM fails/timeouts: Return the raw data neatly
            return f"⚠️ **AI Interpretation Timed Out**\n\nHere is the raw data found in the analysis:\n\n{str(result)}"

    def update_memory(self, question, code, result, final_answer):
        """
        Part 2.3: Internal State Update
        """
        self.memory.append({"role": "user", "content": question})
        self.memory.append({
            "role": "agent", 
            "content": final_answer, 
            "metadata": {"code": code, "result_summary": str(result)[:100]}
        })

    def run(self, question: str):
        """
        Part 4: Agent Loop
        """
        try:
            # 1. Perceive (Time intensive)
            context = self.perceive(question)
            
            # 2. Decide (LLM Call 1)
            code = self.decide(context, question)
            if not code:
                return "Failed to generate analysis code."
            
            # 3. Act (Fast)
            success, result = self.act(code)
            
            if not success:
                return f"Error executing analysis: {result}"
            
            # FIX FOR LAZY MODEL:
            # If the action said "REFER_TO_CONTEXT_STATS", we manually inject the stats into the result.
            # This prevents the model from just repeating the instruction.
            if "REFER_TO_CONTEXT_STATS" in str(result):
                # Extract stats from context if possible, or just provide a generic prompt
                if "Statistical Root Cause Analysis" in context:
                    # Grab the relevant section (hacky but effective)
                    try:
                        stats_section = context.split("Statistical Root Cause Analysis")[1].split("Manuals & Knowledge:")[0]
                        result = f"STATISTICAL DATA FOUND:\n{stats_section}\n\nTask: Explain these statistics to the user."
                    except:
                        result = "Statistical data is in the context above. Please summarize it."
                else:
                    result = "Please check the context for failure statistics."

            # 4. Explain (LLM Call 2) - Reusing context!
            final_answer = self.explain(question, code, result, previous_context=context)
            
            # 5. Update Memory
            self.update_memory(question, code, result, final_answer)
            
            return final_answer
            
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                return "Rate limit reached. Please wait a moment."
            return f"Agent Logic Error: {str(e)}"

# Instantiate a global instance for simple imports if needed
agent_instance = DataAnalystAgent()
