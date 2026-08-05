from agents.base_agent import BaseAgent
from analyzer import analyze_correlations


class SchemaAgent(BaseAgent):
    def get_name(self) -> str:
        return "schema"

    def execute(self, df):
        if df is None:
            raise ValueError("Dataset not loaded")

        return {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.head(1).to_dict(orient="records")
        }

    def build_context(self, df, question: str) -> str:
        schema = self.execute(df)
        correlation_context = ""
        if any(k in question.lower() for k in ["cause", "correlation", "impact"]):
            correlation_context = analyze_correlations(df)

        return f"""
COLUMNS: {schema["columns"]}
SAMPLE:
{df.head(1).to_string()}

CORRELATIONS:
{correlation_context}
"""
