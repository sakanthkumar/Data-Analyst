from agents.base_agent import BaseAgent
from reporting import save_report


class ReportAgent(BaseAgent):
    def get_name(self) -> str:
        return "report"

    def execute(self, report_data, machine_name: str = None, analysis_type: str = "Automated"):
        return save_report(report_data, machine_name, analysis_type)
