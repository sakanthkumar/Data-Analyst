from agents.base_agent import BaseAgent
from analyzer import (
    auto_eda,
    generate_plots,
    get_correlation_stats,
    TargetAnalysisEngine,
)


class AnalyticsAgent(BaseAgent):
    def get_name(self) -> str:
        return "analytics"

    def execute(self, df, include_plots: bool = False, target_override: str = None):
        if df is None:
            raise ValueError("Dataset not loaded")

        result = {
            "eda": auto_eda(df),
            "failure_stats": TargetAnalysisEngine.get_target_stats(df, target_override=target_override),
            "correlation_stats": get_correlation_stats(df, target_override=target_override)
        }

        if include_plots:
            result["plots"] = generate_plots(df)

        return result

    def get_failure_mode_analysis(self, df, target_override: str = None):
        if df is None:
            raise ValueError("Dataset not loaded")

        return TargetAnalysisEngine.analyze_target_drivers(df, target_override=target_override)
