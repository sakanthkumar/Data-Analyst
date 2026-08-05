from agents.base_agent import BaseAgent
from executor import execute_pandas_code


class ExecutorAgent(BaseAgent):
    def get_name(self) -> str:
        return "executor"

    def execute(self, df, code: str):
        if "NO_DATA_ANALYSIS_REQUIRED" in code:
            return True, "NO_DATA"

        return execute_pandas_code(df, code)
