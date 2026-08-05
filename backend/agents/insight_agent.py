from agents.base_agent import BaseAgent
from typing import Dict, Any
import json


class InsightAgent(BaseAgent):
    target_keywords = [
        "why",
        "driver",
        "impact",
        "strategy",
        "mitigate",
        "factor",
        "reason",
        "root cause",
        "failure",
        "target variable"
    ]

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.system_prompt = (
            "You are a Grounded Data Analyst Agent. Explain the computed results of the analysis "
            "clearly and concisely. Follow these rules:\n"
            "- Directness: Provide a clear, direct answer to the user's question first.\n"
            "- Grounding: Only reference values, statistics, and trends that are explicitly present in the "
            "Computed Result or Dataset Context. Never make up outside historical details or cause-and-effect "
            "mechanisms not reflected in the schema/results.\n"
            "- Formatting: Use concise bullet points for key findings. Avoid filler text or storytelling.\n"
            "- Missing Data: If the computed result doesn't contain enough information to explain the cause, "
            "state this clearly rather than speculating."
        )

    def get_name(self) -> str:
        return "insight"

    def format_chat_history(self, chat_history: list, max_turns: int = 5) -> str:
        if not chat_history:
            return ""
        recent_messages = chat_history[-(max_turns * 2):]
        formatted = []
        for msg in recent_messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def execute(self, question: str, result, data_context: Dict[str, Any] = None):
        context_str = ""
        if data_context:
            domain_profile = data_context.get("domain_profile") or {}
            columns = data_context.get("columns") or []
            dtypes = data_context.get("dtypes") or {}
            sample_rows = data_context.get("sample_rows") or []
            
            context_str += "DATASET SCHEMA & CONTEXT:\n"
            if domain_profile:
                context_str += f"- Detected Domain: {domain_profile.get('domain')}\n"
                context_str += f"- Key Business Entities: {', '.join(domain_profile.get('business_entities', []))}\n"
                context_str += f"- Target Analyzed: {domain_profile.get('target_column')}\n"
            context_str += f"- Available Columns: {', '.join(columns)}\n"
            context_str += f"- Column Types: {dtypes}\n"
            
            if sample_rows:
                context_str += f"- Sample Records:\n{json.dumps(sample_rows[:2], indent=2)}\n"

        history_str = ""
        if data_context and data_context.get("chat_history"):
            history_str = "CONVERSATION HISTORY:\n" + self.format_chat_history(data_context.get("chat_history")) + "\n"

        prompt = ""
        if context_str:
            prompt += context_str + "\n"
        if history_str:
            prompt += history_str + "\n"

        prompt += f"""User Question:
{question}

Computed Result:
{result}
"""

        return self.llm_service.call_llm(prompt, system_type="analysis", system_prompt=self.system_prompt)

    def generate_direct(self, prompt: str, system_type: str = "analysis"):
        return self.llm_service.call_llm(prompt, system_type=system_type)

    def is_failure_question(self, question: str) -> bool:
        question_lower = question.lower()
        return any(k in question_lower for k in self.target_keywords)
