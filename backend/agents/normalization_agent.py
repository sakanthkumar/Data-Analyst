from agents.base_agent import BaseAgent
from normalizer import normalize_output


class NormalizationAgent(BaseAgent):
    def get_name(self) -> str:
        return "normalization"

    def execute(self, text: str, section_type: str) -> str:
        return normalize_output(text, section_type)
