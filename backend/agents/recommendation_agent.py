from agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    def get_name(self) -> str:
        return "recommendation"

    def execute(self, insights):
        if not insights:
            return []

        if isinstance(insights, str):
            return [line.strip("- ").strip() for line in insights.splitlines() if line.strip().startswith("-")]

        return insights
