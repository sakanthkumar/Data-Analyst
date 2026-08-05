import os
import sys

# Create services directory
services_dir = os.path.join(os.path.dirname(__file__), 'services')
os.makedirs(services_dir, exist_ok=True)

# Create __init__.py
init_file = os.path.join(services_dir, '__init__.py')
with open(init_file, 'w') as f:
    f.write('# Services package\n')

# Create llm_service.py
llm_service_content = '''import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

# --- EXECUTION CONTROL ---
RUN_LLM_ANALYSIS = True


class LLMService:
    """Service for LLM interactions via Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate",
                 code_model: str = "deepseek-coder",
                 analysis_model: str = "llama3",
                 temperature: float = 0.1):
        """
        Initialize LLM service with model and endpoint configuration.
        
        Args:
            ollama_url: Ollama API endpoint (default from agent.py)
            code_model: Model for code generation (default "deepseek-coder")
            analysis_model: Model for analysis (default "llama3")
            temperature: Temperature parameter for generation (default 0.1)
        """
        self.ollama_url = ollama_url
        self.code_model = code_model
        self.analysis_model = analysis_model
        self.temperature = temperature
        
        # System prompts (from agent.py - preserved unchanged)
        self.system_prompt_common = """
You are a Data Analyst Agent designed for fast, reliable responses.

Primary: Accuracy over verbosity. Speed over style. Clear, actionable outputs.
Rules:
- Default to concise answers (bullet points preferred).
- Avoid repetition, filler, or storytelling.
- Do not hallucinate. State missing data clearly.
- Response length: 80–120 tokens default.
- Structure: 1) Short summary (mandatory), 2) Detail (only if asked).
- Code: Minimal, runnable, no explanations unless requested.
- Error handling: Never fail silently.
"""

        self.system_prompt_code = self.system_prompt_common + """
Goal: Write ONLY valid python pandas code to analyze the dataframe `df`.
Rules:
1. Assign final output to variable `result`.
2. No imports except pandas (pd) and numpy (np).
3. Do NOT explain the code. Just provide the code block.
"""

        self.system_prompt_analysis = self.system_prompt_common + """
Goal: Explain analysis findings and provide actionable recommendations.
Structure:
1. **Conclusion**: Direct answer.
2. **Analysis**: Key metrics/findings.
3. **Recommendation**: Specific action items.
"""

        self.system_prompt_failure_combined = """
================================
SECTION 1: ROOT CAUSE ANALYSIS
================================
ROLE:
Explain WHY failures occur at a system level.

RULES:
- Do NOT include frequencies, percentages, or correlations.
- Do NOT include impact, repair, or prevention.
- Use causal phrases such as "caused by", "driven by", or "resulting from".
- 4–5 bullet points only.
- One sentence per bullet.

================================
SECTION 2: IMPACT ASSESSMENT
================================
ROLE:
Explain WHAT happens due to the failures.

RULES:
- Do NOT explain causes.
- Use qualitative severity only.
- Exactly three lines:
  - Operational Impact: Low/Medium/High (one sentence)
  - Safety Risk: Low/Medium/High (one sentence)
  - Performance Degradation: Low/Medium/High (one sentence)

================================
SECTION 3: REPAIR GUIDE
================================
ROLE:
Explain HOW the issue can be corrected.

RULES:
- High-level corrective actions only.
- No root cause explanation.
- No prevention, scheduling, or maintenance plans.
- 3–5 bullet points only.
- Action-oriented language.

================================
GLOBAL RULES
================================
- Do NOT repeat information across sections.
- Do NOT add conclusions.
- Do NOT add extra titles or commentary.
- If any rule is violated, regenerate internally before responding.
"""

    def set_models(self, code_model: str = None, analysis_model: str = None):
        """Update model identifiers."""
        if code_model:
            self.code_model = code_model
        if analysis_model:
            self.analysis_model = analysis_model

    def call_ollama_with_model(self, prompt: str, system_prompt: str, model: str):
        """
        Call Ollama with specified model via subprocess.
        
        Args:
            prompt: User request text
            system_prompt: System instructions
            model: Model identifier
            
        Returns:
            str: Model response
            
        Raises:
            Exception: If subprocess returns non-zero exit code
        """
        full_prompt = f"{system_prompt}\\n\\nUser Request:\\n{prompt}"

        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        stdout, stderr = process.communicate(input=full_prompt)

        if process.returncode != 0:
            raise Exception(stderr)

        return stdout.strip()

    def call_llm(self, prompt: str, system_type: str = "analysis"):
        """
        Route LLM request to appropriate model with system prompt.
        
        Args:
            prompt: User request
            system_type: Type of prompt ("code", "failure", or "analysis")
            
        Returns:
            str: LLM response or disabled message
        """
        if not RUN_LLM_ANALYSIS:
            return "LLM disabled."

        if system_type == "code":
            model = self.code_model
            system_prompt = self.system_prompt_code
        elif system_type == "failure":
            model = self.analysis_model
            system_prompt = self.system_prompt_failure_combined
        else:
            model = self.analysis_model
            system_prompt = self.system_prompt_analysis

        return self.call_ollama_with_model(prompt, system_prompt, model)
'''

llm_service_file = os.path.join(services_dir, 'llm_service.py')
with open(llm_service_file, 'w') as f:
    f.write(llm_service_content)

print(f"Created {services_dir}")
print(f"Created {init_file}")
print(f"Created {llm_service_file}")
