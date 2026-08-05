import os
import logging
from dotenv import load_dotenv
from agents.providers import create_provider

load_dotenv()

logger = logging.getLogger("DataAnalystAgent.LLMService")

# --- EXECUTION CONTROL ---
RUN_LLM_ANALYSIS = True


class LLMService:
    """Provider-agnostic LLM service supporting Groq (cloud) and Ollama (local).

    Architecture:
        LLMService delegates all inference to a BaseLLMProvider implementation
        resolved at init time via the LLM_BACKEND environment variable.

        Two independent model slots are maintained:
        - reasoning_model: Used for analysis, domain profiling, insights, reports.
        - code_model: Used for Python code generation tasks.
    """

    def __init__(self, ollama_url: str = None,
                 code_model: str = None,
                 analysis_model: str = None,
                 temperature: float = 0.1):
        """Initialize the LLM service.

        Args:
            ollama_url: Legacy parameter preserved for backward compatibility.
            code_model: Override for code generation model name.
            analysis_model: Override for reasoning/analysis model name.
            temperature: Sampling temperature for generation.
        """
        self.temperature = temperature
        self.timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

        # Resolve backend and create provider
        self.backend = os.getenv("LLM_BACKEND", "groq").lower()
        self.provider = create_provider(self.backend)

        # Resolve model names from env with sensible defaults
        if self.backend == "groq":
            self.code_model = code_model or os.getenv("GROQ_MODEL_CODE", "qwen-qwq-32b")
            self.analysis_model = analysis_model or os.getenv("GROQ_MODEL_REASONING", "deepseek-r1-distill-llama-70b")
        else:
            self.analysis_model = analysis_model or os.getenv("OLLAMA_REASONING_MODEL", "llama3")
            self.code_model = code_model or os.getenv("OLLAMA_CODE_MODEL", "deepseek-coder:6.7b")

        # Legacy compat: store ollama_url for set_config
        if ollama_url:
            self.ollama_url = ollama_url
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            self.ollama_url = f"{base_url}/api/generate"

        logger.info(
            f"LLMService initialized: provider={self.provider.get_provider_name()}, "
            f"reasoning_model={self.analysis_model}, code_model={self.code_model}"
        )

        # ================================================================
        # System prompts (business logic - preserved EXACTLY unchanged)
        # ================================================================
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

        self.system_prompt_code = """Goal: Write ONLY valid, executable python pandas code to analyze the dataframe `df`.

Rules:
1. Assign the final output to the variable `result`.
2. Do NOT import any libraries.
3. Absolutely NO explanations, NO comments, NO conversational remarks, and NO markdown code fences.
4. Output syntactically valid Python code only.
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
SECTION 1: DRIVER ANALYSIS
================================
ROLE:
Explain WHY specific target events or anomalies occur at a system level.

RULES:
- Do NOT include frequencies, percentages, or correlations.
- Do NOT include business impact, optimization strategies, or action items.
- Use causal phrases such as "caused by", "driven by", "associated with", or "resulting from".
- 4–5 bullet points only.
- One sentence per bullet.

================================
SECTION 2: IMPACT ASSESSMENT
================================
ROLE:
Explain WHAT happens due to these target events/anomalies.

RULES:
- Do NOT explain causes or drivers.
- Use qualitative severity only.
- Exactly three lines:
  - Operational Impact: Low/Medium/High (one sentence)
  - Risk Level: Low/Medium/High (one sentence)
  - Performance Degradation: Low/Medium/High (one sentence)

================================
SECTION 3: ACTION GUIDE
================================
ROLE:
Explain HOW to optimize, correct, or respond to these events.

RULES:
- High-level corrective or optimization strategies only.
- No driver or root cause explanation.
- No scheduling, training, or preventative plans.
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
        self.system_prompt_executive_industrial = self.system_prompt_failure_combined

        self.system_prompt_executive_business = """
================================
SECTION 1: EXECUTIVE KEY DRIVER ANALYSIS
================================
ROLE:
Explain the key drivers behind the target variable values.

RULES:
- You MUST reference the actual quantitative statistics, correlation coefficients, and category percentages provided in the prompt context.
- Summarize the strongest statistical drivers clearly (e.g. "Sex has the highest correlation with survival, with females surviving at a X% rate vs males at Y%").
- 4-5 bullet points only.
- One sentence per bullet.

================================
SECTION 2: BUSINESS IMPACT ASSESSMENT
================================
ROLE:
Assess the business or operational implications of these findings.

RULES:
- Do NOT explain drivers or causes.
- Translate the statistical observations into high-level business risk or impact (e.g., loss of potential customers, customer satisfaction decline, passenger safety implications).
- Exactly three lines:
  - Strategic Impact: Low/Medium/High (one sentence)
  - Risk Exposure: Low/Medium/High (one sentence)
  - Outcome Variance: Low/Medium/High (one sentence)

================================
SECTION 3: ACTIONABLE STRATEGY GUIDE
================================
ROLE:
Provide actionable recommendations to optimize the target variable outcome.

RULES:
- Provide high-level recommendations that target the key drivers identified.
- Limit to 3-5 bullet points.
- Action-oriented business language (e.g., "Implement X to achieve Y").
- Do not repeat information from previous sections.

================================
GLOBAL RULES
================================
- Do NOT repeat information across sections.
- Do NOT add conclusions or introductory remarks.
- Do NOT add extra titles or commentary.
"""

    # ================================================================
    # Public interface (unchanged API contract)
    # ================================================================

    def set_models(self, code_model: str = None, analysis_model: str = None):
        """Update model identifiers."""
        if code_model:
            self.code_model = code_model
        if analysis_model:
            self.analysis_model = analysis_model

    def call_llm(self, prompt: str, system_type: str = "analysis", system_prompt: str = None):
        """Route LLM request to the active provider with the appropriate model.

        Model routing:
            system_type == "code"    -> self.code_model
            system_type == "analysis" or "failure" -> self.analysis_model
        """
        if not RUN_LLM_ANALYSIS:
            return "LLM disabled."

        if os.getenv("TESTING") == "true":
            if "acronyms" in prompt:
                return '{"acronyms": []}'
            if "domain" in prompt or "analysis_type" in prompt:
                return '{"domain": "Generic Test Domain", "confidence": 0.95, "analysis_type": "descriptive", "target_column": null, "identifier_columns": [], "date_columns": [], "numeric_columns": [], "categorical_columns": [], "business_entities": [], "recommended_kpis": [], "recommended_analytics_tasks": []}'
            return "Mocked test response content."

        # Resolve system prompt
        if system_prompt is None:
            if system_type == "code":
                system_prompt = self.system_prompt_code
            elif system_type == "failure":
                system_prompt = self.system_prompt_failure_combined
            else:
                system_prompt = self.system_prompt_analysis

        # Route to correct model
        model = self.code_model if system_type == "code" else self.analysis_model

        return self.provider.generate(prompt, system_prompt, model, self.temperature)

    # Legacy methods preserved for backward compatibility
    def call_ollama_with_model(self, prompt: str, system_prompt: str, model: str):
        """Legacy method: delegates to provider."""
        return self.provider.generate(prompt, system_prompt, model, self.temperature)

    def get_executive_report_prompt(self, domain_profile: dict, target_column: str, columns: list, kpis: list, correlations: list) -> str:
        domain = domain_profile.get("domain") or "Generic Analysis"
        analysis_type = domain_profile.get("analysis_type") or "descriptive"
        
        domain_lower = domain.lower()
        analysis_lower = analysis_type.lower()
        columns_lower = [c.lower() for c in columns]
        target_lower = target_column.lower() if target_column else ""
        
        # Classify the domain
        is_industrial = any(
            keyword in domain_lower or keyword in analysis_lower or keyword in target_lower
            for keyword in ["machine", "maintenance", "failure", "industrial", "hardware", "iot", "sensor", "telemetry", "equipment", "reliability", "downtime"]
        ) or any(
            keyword in columns_lower
            for keyword in ["tool wear", "rotational speed", "torque", "tool_wear", "air_temperature", "process_temperature"]
        )
        
        is_survival = not is_industrial and (any(
            keyword in domain_lower or keyword in analysis_lower or keyword in target_lower
            for keyword in ["survival", "survived", "mortality", "fatality", "rescue", "safety", "passenger", "evacuation", "titanic", "historical"]
        ) or any(
            keyword in columns_lower
            for keyword in ["survived", "passengerclass", "pclass", "sibsp", "parch", "embarked", "age", "sex"]
        ))
        
        columns_str = ", ".join(f"'{c}'" for c in columns)
        
        if is_survival:
            return f"""You are a Senior Demographic & Survival Analyst. Generate a domain-aware Executive Insights Report for the '{domain}' dataset.

Structure the report EXACTLY using these four markdown headers:

# Executive Summary
[A concise demographic/historical overview summarizing the study domain, the target survival/safety variable ('{target_column}'), and overall survival rate. 2-3 sentences.]

# Key Findings
- [Survival Driver 1: Explain the primary demographic or safety factor driving survival outcomes. One sentence.]
- [Survival Driver 2: Explain the second key demographic driver and its relationship. One sentence.]
- [Survival Driver 3: Explain the specific age group, class, or category most impacted. One sentence.]

# Statistical Evidence
- [Demographic Factor: Cite specific passenger class, demographics, or safety percentages from the data. One sentence.]
- [Statistical Finding: State a key quantitative correlation or statistical shift in evacuation/survival. One sentence.]
- [Evacuation Analysis: Detail specific statistical findings from the computed metrics. One sentence.]

# Recommendations
- [Safety/Evacuation Recommendation 1: Technical evacuation planning or safety outcome improvements. One sentence.]
- [Safety/Evacuation Recommendation 2: Policy/demographic equity recommendation based on historical safety analysis. One sentence.]

GLOBAL RULES:
- Terminology constraints:
  - ALLOWED terminology: "Survival Rate", "Passenger Class", "Demographics", "Age Groups", "Family Relationships", "Safety Outcomes", "Evacuation Analysis", "Statistical Correlations".
  - FORBIDDEN terminology (Do NOT use): "Revenue Impact", "Customer Impact", "KPI Impact", "Marketing Campaigns", "Loyalty Programs", "Revenue Optimization", "Business Driver", "Downtime Risk", "Maintenance Actions", "Operational Impact", "Performance Degradation", "Reliability".
- No Hallucinations: You must NEVER invent or guess percentages or numerical values. Every number, percentage, or correlation value mentioned in the report MUST come directly from the dataset statistics, computed metrics, or correlation analysis provided in the context.
- No estimations: Do NOT estimate survival rate increases, safety percentage improvements, or KPI changes.
- Recommendations and findings MUST be domain-specific and derived strictly from the available column list: [{columns_str}].
- NEVER propose actions referencing columns not present in the dataset.
- DO NOT add conclusions or introductory remarks.
- DO NOT add extra titles or commentary.
"""
        elif is_industrial:
            return f"""You are a Senior Predictive Maintenance & Reliability Engineer. Generate a domain-aware Executive Insights Report for the '{domain}' dataset.

Structure the report EXACTLY using these four markdown headers:

# Executive Summary
[A concise technical overview summarizing the system monitoring domain, the target failure variable ('{target_column}'), and overall failure rate. 2-3 sentences.]

# Key Findings
- [Failure Driver 1: Explain the primary engineering/telemetered factor causing failure. One sentence.]
- [Failure Driver 2: Explain the second key telemetry driver and its relationship. One sentence.]
- [Failure Driver 3: Explain the specific failure mode or component most impacted. One sentence.]

# Statistical Evidence
- [Failure Causes: Cite specific telemetry thresholds, correlations, or category percentages causing the event. One sentence.]
- [Risk Factors: Cite specific statistical probabilities or correlation coefficients linked to severe system risks. One sentence.]
- [Maintenance telemetry: Detail specific statistical findings (e.g. shifts in telemetry averages during failures). One sentence.]

# Recommendations
- [Maintenance Recommendation 1: Technical preventive maintenance scheduling or parameter adjustment. One sentence.]
- [Maintenance Recommendation 2: Corrective strategy to mitigate operational downtime. One sentence.]
- Operational Impact: [Low/Medium/High] - [Describe impact on production or cycle times in one sentence.]
- Risk Exposure: [Low/Medium/High] - [Describe safety or equipment damage risk in one sentence.]
- Performance Degradation: [Low/Medium/High] - [Describe loss of throughput or yield in one sentence.]

GLOBAL RULES:
- Terminology constraints:
  - ALLOWED terminology: "Failure Drivers", "Downtime Risk", "Maintenance Actions", "Operational Impact", "Performance Degradation", "Reliability".
  - FORBIDDEN terminology (Do NOT use): "Revenue Impact", "Customer Impact", "KPI Impact", "Marketing Campaigns", "Loyalty Programs", "Revenue Optimization", "Business Driver", "Evacuation Analysis", "Survival Rate", "Passenger Class", "Demographics", "Age Groups", "Family Relationships", "Safety Outcomes", "Statistical Correlations".
- No Hallucinations: You must NEVER invent or guess percentages or numerical values. Every number, percentage, or correlation value mentioned in the report MUST come directly from the dataset statistics, computed metrics, or correlation analysis provided in the context.
- No estimations: Do NOT estimate downtime decreases, reliability improvements, or KPI changes.
- Recommendations and findings MUST be domain-specific and derived strictly from the available column list: [{columns_str}].
- NEVER propose actions referencing columns not present in the dataset.
- DO NOT add conclusions or introductory remarks.
- DO NOT add extra titles or commentary.
"""
        else:
            return f"""You are a Senior Business & Growth Analyst. Generate a domain-aware Executive Insights Report for the '{domain}' dataset.

Structure the report EXACTLY using these four markdown headers:

# Executive Summary
[A concise high-level business overview summarizing the business domain, the target variable ('{target_column}'), and its overall prevalence/prevalence rate. 2-3 sentences.]

# Key Findings
- [Business Driver 1: Explain the primary business factor driving the target outcome. One sentence.]
- [Business Driver 2: Explain the second key driver and its direction. One sentence.]
- [Business Driver 3: Explain the customer/market segment most impacted. One sentence.]

# Statistical Evidence
- [Revenue Impact: Cite specific percentages and statistical correlation values showing how revenue/pricing is impacted. One sentence.]
- [Customer Impact: Cite category percentages or group counts showing which customer segment is most affected. One sentence.]
- [KPI Impact: State the quantitative correlation or statistical metric change for a key business KPI. One sentence.]

# Recommendations
- [Strategic Action 1: Customer retention or conversion strategy targeting the primary driver. One sentence.]
- [Strategic Action 2: Operational strategy to optimize business performance or pricing structure. One sentence.]

GLOBAL RULES:
- Terminology constraints:
  - ALLOWED terminology: "Revenue", "Customer Segments", "Churn", "Retention", "KPIs", "Profitability".
  - FORBIDDEN terminology (Do NOT use): "Failure Drivers", "Downtime Risk", "Maintenance Actions", "Operational Impact", "Performance Degradation", "Reliability", "Evacuation Analysis", "Survival Rate", "Passenger Class", "Demographics", "Age Groups", "Family Relationships", "Safety Outcomes", "Statistical Correlations".
- No Hallucinations: You must NEVER invent or guess percentages or numerical values. Every number, percentage, or correlation value mentioned in the report MUST come directly from the dataset statistics, computed metrics, or correlation analysis provided in the context.
- No estimations: Do NOT estimate revenue increases or KPI changes.
- Recommendations and findings MUST be domain-specific and derived strictly from the available column list: [{columns_str}].
- NEVER propose actions referencing columns not present in the dataset.
- DO NOT add conclusions or introductory remarks.
- DO NOT add extra titles or commentary.
"""


