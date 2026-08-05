import json
from agents.base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.system_prompt = (
            "You are the Master Orchestrator Agent. Your task is to analyze a user query and "
            "determine the exact execution plan (route of agents) to resolve the query. "
            "You must return ONLY a JSON object with a single key 'route' containing a list of agent names.\n\n"
            "Available Agents:\n"
            "- SchemaAgent: Inspects data structure (columns, sample rows).\n"
            "- CodeGeneratorAgent: Generates python code for data queries.\n"
            "- ExecutorAgent: Executes generated python code on the dataframe.\n"
            "- InsightAgent: Provides textual explanation, summaries, answers from context, or conversational responses.\n"
            "- AnalyticsAgent: Performs pre-defined comprehensive EDA, correlation, or failure mode analysis.\n"
            "- KnowledgeAgent: Performs RAG searches against uploaded manuals/documents.\n\n"
            "Routing Rules:\n"
            "1. For questions requiring data calculations, aggregations, filtering, counts, stats, or plots: "
            "['SchemaAgent', 'CodeGeneratorAgent', 'ExecutorAgent', 'InsightAgent']\n"
            "2. For simple greetings, direct chat, follow-up explanations, general discussion, or questions "
            "that don't require pandas code execution: "
            "['InsightAgent']\n"
            "3. For pre-packaged automatic data analysis/EDA requests: "
            "['AnalyticsAgent']\n"
            "4. For questions about machine manuals, documentation, user manuals, or reference guides: "
            "['KnowledgeAgent', 'InsightAgent']\n\n"
            "Output constraints:\n"
            "- Do not include any explanations or conversational text before or after the JSON.\n"
            "- Return ONLY valid JSON: {\"route\": [...]}\n"
        )

    def get_name(self) -> str:
        return "orchestrator"

    def execute(self, question: str, chat_history_summary: str = "") -> dict:
        prompt = f"""Conversation History Summary:\n{chat_history_summary}\n\nUser Question:\n{question}\n\nDetermine the routing execution plan. Return ONLY JSON:"""
        try:
            raw = self.llm_service.call_llm(prompt, system_type="analysis", system_prompt=self.system_prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end != -1:
                plan = json.loads(cleaned[start:end])
                if "route" in plan and isinstance(plan["route"], list):
                    # Ensure agent names are valid
                    valid_agents = {"SchemaAgent", "CodeGeneratorAgent", "ExecutorAgent", "InsightAgent", "AnalyticsAgent", "KnowledgeAgent"}
                    filtered_route = [agent for agent in plan["route"] if agent in valid_agents]
                    if filtered_route:
                        return {"route": filtered_route}

        except Exception as e:
            print(f"Error in OrchestratorAgent planning: {e}")

        # Safe fallback: full data analysis pipeline
        return {"route": ["SchemaAgent", "CodeGeneratorAgent", "ExecutorAgent", "InsightAgent"]}

