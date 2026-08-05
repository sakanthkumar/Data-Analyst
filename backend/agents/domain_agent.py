from typing import List, Dict, Any
import json
from agents.base_agent import BaseAgent
from agents.llm_service import LLMService

class DomainAgent(BaseAgent):
    """
    Profile dataset semantics (domain, target, types, entities, KPI ideas)
    without dictating how or in what order analyses should execute.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.system_prompt = (
            "You are a Senior Data Profiler Agent. Analyze column names, pandas types, "
            "and sample records. Output ONLY a valid JSON profile matching the requested "
            "schema constraints. Do not output markdown wrapping except when requested."
        )

    def get_name(self) -> str:
        return "domain"

    def execute(
        self,
        columns: List[str],
        dtypes: Dict[str, str],
        sample_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize the dataset metadata into a declarative Domain Profile.
        
        Args:
            columns: Full list of column headers.
            dtypes: Python/Pandas data types.
            sample_rows: Head 3 records from the dataset.
            
        Returns:
            Dict matching the requested JSON profile schema.
        """
        prompt = self._build_prompt(columns, dtypes, sample_rows)
        raw_response = self.llm_service.call_llm(prompt, system_type="analysis", system_prompt=self.system_prompt)
        return self._parse_response(raw_response, columns)


    def _build_prompt(
        self,
        columns: List[str],
        dtypes: Dict[str, str],
        sample_rows: List[Dict[str, Any]]
    ) -> str:
        schema_info = {
            "columns": columns,
            "dtypes": dtypes,
            "sample_records": sample_rows
        }
        
        prompt = f"""
Analyze this dataset structure and provide a semantic profile.

SCHEMA & RECORDS:
{json.dumps(schema_info, indent=2)}

You must return a JSON object with the following fields:
- "domain" (string): The business domain (e.g. "Customer Churn Analysis").
- "confidence" (float): Your confidence score from 0.0 to 1.0 indicating how certain you are about the detected domain.
- "analysis_type" (string): The primary analysis category. Must be one of: "descriptive", "classification", "regression", "forecasting", "segmentation", "anomaly_detection".
- "target_column" (string or null): The primary variable to predict or analyze. Must match an actual column name.
- "identifier_columns" (array of strings): Primary key or ID columns.
- "date_columns" (array of strings): Temporal/timestamp columns.
- "numeric_columns" (array of strings): Continuous numerical features.
- "categorical_columns" (array of strings): Categorical groups or textual tags.
- "business_entities" (array of strings): High-level business concepts/nouns represented (e.g. ["Customer", "Subscription"]).
- "recommended_kpis" (array of objects): 2-3 KPIs to calculate. Each object must have "name", "metric_type" (must be one of: "average", "sum", "count", "percentage", "min", "max"), "column" (must match a column name), and "description".
- "recommended_analytics_tasks" (array of objects): 2-3 analyses to run. Each object must have "title" and "description".

Constraint: All column names assigned must match the casing and spelling in the input schema exactly.
Return ONLY valid JSON.
"""
        return prompt

    def _parse_response(self, text: str, original_columns: List[str]) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            profile = json.loads(cleaned)
        except json.JSONDecodeError:
            # Simple bracket extractor fallback
            try:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start != -1 and end != -1:
                    profile = json.loads(cleaned[start:end])
                else:
                    raise ValueError("No json brackets found")
            except Exception:
                profile = self._generate_fallback_profile(original_columns)

        # Sanitize columns mapping against the original columns list
        target = profile.get("target_column")
        if target and target not in original_columns:
            profile["target_column"] = None
            
        for key in ["identifier_columns", "date_columns", "numeric_columns", "categorical_columns"]:
            if key in profile:
                profile[key] = [c for c in profile[key] if c in original_columns]
            else:
                profile[key] = []

        return profile

    def _generate_fallback_profile(self, columns: List[str]) -> Dict[str, Any]:
        return {
            "domain": "Generic Dataset Analysis",
            "confidence": 0.5,
            "analysis_type": "descriptive",
            "target_column": columns[-1] if columns else None,
            "identifier_columns": [c for c in columns if "id" in c.lower()],
            "date_columns": [c for c in columns if "date" in c.lower() or "time" in c.lower()],
            "numeric_columns": [],
            "categorical_columns": [],
            "business_entities": [],
            "recommended_kpis": [],
            "recommended_analytics_tasks": []
        }
