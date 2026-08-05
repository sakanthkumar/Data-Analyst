from agents.base_agent import BaseAgent
from knowledge import kb


class KnowledgeAgent(BaseAgent):
    def get_name(self) -> str:
        return "knowledge"

    def execute(self, query: str, k=None):
        return kb.search_manuals(query, k=k)
