import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any

from agents.llm_service import LLMService
from agents.schema_agent import SchemaAgent
from agents.code_generator_agent import CodeGeneratorAgent
from agents.executor_agent import ExecutorAgent
from agents.insight_agent import InsightAgent
from agents.analytics_agent import AnalyticsAgent
from agents.domain_agent import DomainAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.knowledge_agent import KnowledgeAgent

load_dotenv()

# --- EXECUTION CONTROL ---
RUN_LLM_ANALYSIS = True


class DataAnalystAgent:
    def __init__(self):
        self.memory = []
        self.active_memory = [] # Sliding window of dicts: {'question', 'code', 'result_summary', 'response'}
        self.df = None
        self.context_data = {}
        self.domain_profile = {}

        # Configuration
        import os
        self.backend = os.getenv("LLM_BACKEND", "groq")
        self.temperature = 0.1

        # Initialize LLM service (provider resolved internally from LLM_BACKEND)
        self.llm_service = LLMService(
            temperature=self.temperature
        )
        self.schema_agent = SchemaAgent()
        self.code_generator_agent = CodeGeneratorAgent(self.llm_service)
        self.executor_agent = ExecutorAgent()
        self.insight_agent = InsightAgent(self.llm_service)
        self.analytics_agent = AnalyticsAgent()
        self.domain_agent = DomainAgent(self.llm_service)
        self.orchestrator_agent = OrchestratorAgent(self.llm_service)
        self.knowledge_agent = KnowledgeAgent()

    def get_config(self):
        return {
            "backend": self.backend,
            "provider": self.llm_service.provider.get_provider_name(),
            "reasoning_model": self.llm_service.analysis_model,
            "code_model": self.llm_service.code_model,
            "connected": {self.llm_service.provider.get_provider_name(): True}
        }

    def set_model(self, model: str):
        self.llm_service.set_models(analysis_model=model)
        return f"Analysis model switched to {model}"

    @property
    def analysis_model(self):
        return self.llm_service.analysis_model

    @property
    def code_model(self):
        return self.llm_service.code_model
        
    def get_available_models(self):
        return [self.llm_service.code_model, self.llm_service.analysis_model]
        
    def set_temperature(self, temp: float):
        self.temperature = temp
        self.llm_service.temperature = temp
        return f"Temperature set to {temp}"
        
    def set_config(self, config: dict):
        if "system_prompt" in config:
            self.llm_service.system_prompt_common = config["system_prompt"]
        if "ollama_url" in config:
            self.ollama_url = config["ollama_url"]
            self.llm_service.ollama_url = config["ollama_url"]
        return "Expert configuration updated"

    # ---------------- LLM CALL (DELEGATED TO SERVICE) ----------------

    def _call_llm(self, prompt: str, system_type="analysis", system_prompt: str = None):
        return self.llm_service.call_llm(prompt, system_type=system_type, system_prompt=system_prompt)

    def generate_direct(self, prompt: str, system_type: str = "analysis", system_prompt: str = None):
        return self._call_llm(prompt, system_type=system_type, system_prompt=system_prompt)

    def filter_acronyms(self, candidates: List[str]) -> List[str]:
        """
        Filters a list of candidate terms using LLM semantic analysis.
        Only keeps abbreviations, codes, and domain-specific abbreviations.
        Excludes common English words, dataset labels, and normal business terms.
        """
        if not candidates:
            return []
            
        prompt = f"""
You are a semantic analysis tool. Your task is to filter a list of candidate terms from a dataset and identify only those that are abbreviations, acronyms, codes, or domain-specific abbreviations.

DO NOT include:
- Common English words (e.g., "Survived", "Age", "Fare", "Sex", "Embarked", "Class", "Ticket", "Cabin", "Name", "Passenger", "Male", "Female")
- Standard dataset labels / column names (e.g., "PassengerId", "customer_id", "status", "active")
- Normal business terms (e.g., "Revenue", "Cost", "Profit", "Manager", "Store")

DO include:
- Acronyms or Abbreviations (e.g., "TPM", "OEE", "MTBF", "RPM", "SKU", "CRM", "TWF", "HDF", "PWF", "OSF", "RNF")
- Technical or domain-specific codes / abbreviations

Input list of candidate terms: {candidates}

Return a JSON object with a single key "acronyms" whose value is a list of only the terms that meet the inclusion criteria.
Return ONLY valid JSON.
"""
        try:
            response_text = self._call_llm(prompt, system_type="analysis")
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end != -1:
                import json
                data = json.loads(cleaned[start:end])
                result = data.get("acronyms", [])
                # Ensure all returned acronyms were actually in the candidates list
                return [c for c in result if c in candidates]
        except Exception as e:
            print(f"Error in LLM acronym filtering: {e}")
            
        return self._fallback_filter_acronyms(candidates)

    def _fallback_filter_acronyms(self, candidates: List[str]) -> List[str]:
        # Simple heuristic fallback
        common_non_acronyms = {
            "survived", "age", "fare", "sex", "embarked", "passengerid", "name", 
            "class", "ticket", "cabin", "pclass", "sibsp", "parch", "passenger",
            "male", "female", "yes", "no", "true", "false", "status", "active",
            "revenue", "cost", "profit", "manager", "store", "product", "type"
        }
        filtered = []
        for c in candidates:
            # Clean up candidate if it contains '=' or spaces
            parts = [p.strip() for p in c.replace("=", " ").split()]
            for part in parts:
                part_lower = part.lower()
                if part_lower in common_non_acronyms:
                    continue
                # If it's short and uppercase (like TPM, OEE, RNF), it's likely an acronym
                if part.isupper() and 2 <= len(part) <= 6:
                    if c not in filtered:
                        filtered.append(c)
                        break
        return filtered

    # ---------------- DATA ----------------#

    def set_df(self, df: pd.DataFrame, context_data: dict = None):
        self.df = df
        self.context_data = context_data or {}
        self.domain_profile = {}

    def profile_dataset(self, columns: List[str], dtypes: Dict[str, str], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the DomainAgent to profile the uploaded dataset.
        """
        try:
            profile = self.domain_agent.execute(columns, dtypes, sample_rows)
            self.domain_profile = profile
            return profile
        except Exception as e:
            fallback = self.domain_agent._generate_fallback_profile(columns)
            fallback["error"] = str(e)
            self.domain_profile = fallback
            return fallback

    # ---------------- ANALYTICS ---------------- #

    def analyze_dataset(self, df: pd.DataFrame = None, include_plots: bool = False, target_override: str = None):
        target_df = self.df if df is None else df
        return self.analytics_agent.execute(target_df, include_plots=include_plots, target_override=target_override)

    def get_target_stats(self, df: pd.DataFrame = None, target_override: str = None):
        return self.analyze_dataset(df, target_override=target_override)["failure_stats"]

    def get_failure_stats(self, df: pd.DataFrame = None, target_override: str = None):
        return self.get_target_stats(df, target_override=target_override)

    def get_correlation_stats(self, df: pd.DataFrame = None, target_override: str = None):
        return self.analyze_dataset(df, target_override=target_override)["correlation_stats"]

    def get_eda(self, df: pd.DataFrame = None):
        return self.analyze_dataset(df)["eda"]

    def get_plots(self, df: pd.DataFrame = None):
        return self.analyze_dataset(df, include_plots=True)["plots"]

    def get_target_drivers_analysis(self, df: pd.DataFrame = None, target_override: str = None):
        target_df = self.df if df is None else df
        return self.analytics_agent.get_failure_mode_analysis(target_df, target_override=target_override)

    def get_failure_mode_analysis(self, df: pd.DataFrame = None, target_override: str = None):
        return self.get_target_drivers_analysis(df, target_override=target_override)

    # ---------------- PERCEPTION ---------------- #

    def perceive(self, question: str):
        if self.df is None:
            raise ValueError("Dataset not loaded")

        return self.schema_agent.build_context(self.df, question)

    # ---------------- DECISION ---------------- CODE GENRERATOR #

    def decide(self, context: str, question: str, chat_history: list = None):
        return self.code_generator_agent.execute(context, question, chat_history=chat_history)

    # ---------------- ACTION ---------------- CODE EXECUTER #

    def act(self, code: str):
        return self.executor_agent.execute(self.df, code)

    # ---------------- EXPLAIN (ROUTER) INSIGHT GENERATOR ---------------- #

    def explain(self, question: str, result, chat_history: list = None):
        data_context = {}
        if self.df is not None:
            data_context = {
                "columns": self.df.columns.tolist(),
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "sample_rows": self.df.head(3).to_dict(orient="records"),
                "domain_profile": self.domain_profile,
                "chat_history": chat_history or []
            }
        return self.insight_agent.execute(question, result, data_context=data_context)

    # ---------------- MEMORY HELPERS ---------------- #

    def _get_memory_summary(self) -> str:
        if not self.active_memory:
            return "No previous conversation context."
        
        summary_lines = []
        for i, turn in enumerate(self.active_memory):
            summary_lines.append(f"Turn {i+1}:")
            summary_lines.append(f"  User Question: {turn.get('question', '')}")
            if turn.get('code'):
                summary_lines.append(f"  Generated Code: {turn.get('code', '')}")
            if turn.get('result_summary'):
                summary_lines.append(f"  Execution Outcome: {turn.get('result_summary', '')}")
            summary_lines.append(f"  Assistant Response: {turn.get('response', '')}")
        return "\n".join(summary_lines)

    def _summarize_execution_result(self, result) -> str:
        if result is None:
            return "None"
        if isinstance(result, pd.DataFrame):
            # Truncate and list columns and shape
            return f"DataFrame with shape {result.shape}, Columns: {list(result.columns)}. Preview of first row: {result.head(1).to_dict(orient='records')}"
        if isinstance(result, pd.Series):
            return f"Series of length {len(result)}, Name: {result.name}, Index: {list(result.index[:5])}"
        if isinstance(result, (list, tuple)):
            if len(result) > 5:
                return f"List of length {len(result)} containing items: {list(result[:5])}..."
            return str(result)
        if isinstance(result, dict):
            keys = list(result.keys())
            if len(keys) > 5:
                truncated_dict = {k: result[k] for k in keys[:5]}
                return f"Dict with keys {keys}. Preview of first 5 keys: {truncated_dict}..."
            return str(result)
        val_str = str(result)
        if len(val_str) > 200:
            return val_str[:200] + "..."
        return val_str

    # ---------------- RUN ---------------- #

    def run(self, question: str, chat_history: list = None):
        print("RUN METHOD CALLED")
        print("STEP 1 - run() entered")

        # Sync active memory sliding window if chat_history is cleared
        if not chat_history:
            self.active_memory = []

        chat_history_summary = self._get_memory_summary()

        # Call Orchestrator to generate plan route
        plan = self.orchestrator_agent.execute(question, chat_history_summary=chat_history_summary)
        route = plan.get("route", ["SchemaAgent", "CodeGeneratorAgent", "ExecutorAgent", "InsightAgent"])
        print(f"[Orchestrator] Execution Plan Route: {route}")

        context = ""
        code = ""
        result = None
        success = True
        response = ""

        # Sequential execution of the planned agents
        for agent_name in route:
            print(f"[Orchestrator] Executing Agent: {agent_name}")

            if agent_name == "SchemaAgent":
                context = self.perceive(question)
                print("STEP - perceive completed")

            elif agent_name == "CodeGeneratorAgent":
                code = self.decide(context, question, chat_history=chat_history)
                print("STEP - decide completed")
                print("\n========== GENERATED CODE ==========")
                print(code)
                print("====================================\n")

            elif agent_name == "ExecutorAgent":
                # Self-correction loop: max 3 retries
                max_retries = 3
                attempt = 0
                while attempt <= max_retries:
                    success, result = self.act(code)
                    if success:
                        break
                    
                    attempt += 1
                    if attempt <= max_retries:
                        print(f"[Self-Correction] Execution failed: {result}. Attempt {attempt}/{max_retries}. Retrying...")
                        # Gather full context to feed back to CodeGenerator
                        schema_context = self.perceive(question)
                        code = self.code_generator_agent.execute(
                            context=schema_context,
                            question=question,
                            chat_history=chat_history,
                            failed_code=code,
                            error_message=str(result)
                        )
                        print(f"[Self-Correction] Regenerated code:\n{code}")
                    else:
                        print(f"[Self-Correction] Execution failed permanently after {max_retries} retries.")
                
                print("STEP - execution completed")
                if not success:
                    print("EXECUTION FAILED PERMANENTLY")
                    print(result)
                    return f"Execution Error after retries: {result}"

            elif agent_name == "InsightAgent":
                # If InsightAgent is called after code execution or as direct conversational response
                response = self.explain(question, result, chat_history=chat_history)
                print("STEP - explain completed")

            elif agent_name == "AnalyticsAgent":
                # Execute full automated analysis on the current dataframe
                result = self.analyze_dataset(self.df, include_plots=True)
                response = f"EDA completed. Auto-EDA results: {str(result.get('eda'))[:500]}..."
                print("STEP - AnalyticsAgent execution completed")

            elif agent_name == "KnowledgeAgent":
                # Execute RAG query against local vector store
                result = self.knowledge_agent.execute(question)
                print("STEP - KnowledgeAgent execution completed")

        # Fallback if route did not generate response text
        if not response:
            if result is not None:
                response = str(result)
            else:
                response = "Agent pipeline completed execution but produced no textual explanation."

        # Update sliding window active memory
        result_summary = self._summarize_execution_result(result)
        self.active_memory.append({
            "question": question,
            "code": code,
            "result_summary": result_summary,
            "response": response
        })

        if len(self.active_memory) > 5:
            self.active_memory.pop(0)

        return response



# ---------- INSTANCE ----------
agent_instance = DataAnalystAgent()
