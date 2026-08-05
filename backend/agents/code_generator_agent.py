from agents.base_agent import BaseAgent


class CodeGeneratorAgent(BaseAgent):
    skip_words = ["explain", "summary", "recommend"]

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def get_name(self) -> str:
        return "code_generator"

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

    def execute(self, context: str, question: str, chat_history: list = None, failed_code: str = None, error_message: str = None) -> str:
        if any(w in question.lower() for w in self.skip_words) and not failed_code:
            return "result = 'NO_DATA_ANALYSIS_REQUIRED'"

        history_str = ""
        if chat_history:
            history_str = "CONVERSATION HISTORY:\n" + self.format_chat_history(chat_history) + "\n"

        reflection_str = ""
        if failed_code and error_message:
            reflection_str = f"""
====================================
SELF-CORRECTION / ERROR REFLECTION:
Your previous generated code failed to run.
Failed Code:
{failed_code}

Execution Error message:
{error_message}

Analyze the error (e.g. check for typos in column names, incorrect pandas method usage, or type mismatch), and generate corrected code.
====================================
"""

        prompt = f"""Given the following context about the pandas DataFrame `df`:
{context}
{history_str}
{reflection_str}
You must write ONLY executable Python code to answer the user's question: "{question}"

CRITICAL INSTRUCTIONS:
1. Output ONLY valid, executable Python code.
2. Absolutely NO comments, explanation, conversational introduction, markdown code blocks, or post-generation remarks. 
3. Assume the DataFrame already exists in local scope as `df`.
4. Store the final answer/result in a variable named `result`.
5. Do NOT import any libraries.
6. Do NOT create or redefine `df`.
7. Do NOT generate SQL, pseudo-code, or English explanation.
8. Keep the code minimal and direct.
9. The final line MUST assign the answer to the variable `result`.
10. If aggregation is required, return a Python scalar (int, float, str, dict, or list).
11. Never use print().

Prompt:
Question: How many passengers survived?
Output:
result = int(df["Survived"].sum())

Question: What is the average fare?
Output:
result = float(df["Fare"].mean())

Question: {question}
Output:"""
        response = self.llm_service.call_llm(prompt, system_type="code")
        return response.replace("```python", "").replace("```", "").strip()

