# Task-to-Model Mapping Guide

This document provides a comprehensive mapping of which local **Ollama models** and **embeddings** are utilized for specific tasks and API endpoints within the Data Analyst Agent codebase.

---

## 🛠️ Summary of Configured Models

The system is designed to run entirely locally using **Ollama** and **ChromaDB**. It divides tasks between distinct models based on their specialized capabilities:

1. **Code Generation Model (`code_model`)**: `deepseek-coder:6.7b`
   - *Purpose*: Specialized in generating syntactically correct Python/Pandas code.
   - *Default*: Set to `deepseek-coder:6.7b` (initialized in [agent.py](file:///d:/Data_Analyst_Agent/backend/agent.py#L38)).
2. **Analysis & Reasoner Model (`analysis_model`)**: `llama3`
   - *Purpose*: Handles general reasoning, dataset profiling, explanations, structural formatting, and semantic filtering.
   - *Default*: Set to `llama3` (initialized in [agent.py](file:///d:/Data_Analyst_Agent/backend/agent.py#L39)).
3. **Text Embeddings Model (`EMBEDDING_MODEL`)**: `nomic-embed-text`
   - *Purpose*: Generates dense vectors for PDF manuals ingested in the RAG pipeline.
   - *Default*: Set to `nomic-embed-text` (defined in [knowledge.py](file:///d:/Data_Analyst_Agent/backend/knowledge.py#L14)).

---

## 📋 Task-to-Model Mapping Table

| Pipeline Stage / Task | Backend Agent / Component | Model Used | Ollama API Endpoint | System Prompt / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Semantics Profiling** | [DomainAgent](file:///d:/Data_Analyst_Agent/backend/agents/domain_agent.py) | `analysis_model` (`llama3`) | `/api/generate` | `Senior Data Profiler Agent` prompt. Discovers business domain, primary analysis type, target column, identifier, numerical and categorical columns, recommended KPIs, and suggested analytics tasks. |
| **Python Code Generation (Decide)** | [CodeGeneratorAgent](file:///d:/Data_Analyst_Agent/backend/agents/code_generator_agent.py) | `code_model` (`deepseek-coder:6.7b`) | `/api/generate` | Strict code generation prompt (`Goal: Write ONLY valid python pandas code...`). Excludes comments, explanations, markdown formatting, or conversational text. |
| **Grounded Insights (Explain)** | [InsightAgent](file:///d:/Data_Analyst_Agent/backend/agents/insight_agent.py) | `analysis_model` (`llama3`) | `/api/generate` | `Grounded Data Analyst Agent` prompt. Explains execution results without hallucinating outer domain context, limiting response size, and citing evidence. |
| **Target Variable / Failure Analysis & Executive Reporting** | [main.py](file:///d:/Data_Analyst_Agent/backend/main.py#L273) (`run_background_analysis`) | `analysis_model` (`llama3`) | `/api/generate` | Dynamically selected based on dataset domain:<br>• **Demographic & Survival**: `Senior Demographic & Survival Analyst`<br>• **Predictive Maintenance**: `Senior Predictive Maintenance & Reliability Engineer`<br>• **Business & Growth**: `Senior Business & Growth Analyst` |
| **Acronym Semantic Filtering** | [agent.py](file:///d:/Data_Analyst_Agent/backend/agent.py#L93) (`filter_acronyms`) | `analysis_model` (`llama3`) | `/api/generate` | `Semantic analysis tool` prompt. Distinguishes technical acronyms (e.g. TPM, OEE, RPM) from common words/column names (e.g. Sex, Fare, Age). |
| **Chat Response Structuring** | [main.py](file:///d:/Data_Analyst_Agent/backend/main.py#L633) (`structure_chat_response`) | `analysis_model` (`llama3`) | `/api/generate` | Schema mapping prompt. Extracts raw markdown answers into a structured JSON response containing analysis, evidence columns, confidence score, visual suggestions, follow-up chips, and reasoning trace steps. |
| **RAG Document Ingestion & Search** | [KnowledgeBase](file:///d:/Data_Analyst_Agent/backend/knowledge.py) | `EMBEDDING_MODEL` (`nomic-embed-text`) | `/api/embeddings` | Vector generation function (`OllamaEmbeddings`). No system prompt required. |

---

## ⚙️ Environment Configurations

The behavior of the LLM Service and the local Ollama connection is controlled via the following environment variables (defined in your [.env](file:///d:/Data_Analyst_Agent/.env) file or parsed from defaults):

* **`OLLAMA_BASE_URL`**: Base host URL for the Ollama server. Defaults to `http://localhost:11434` if not specified.
* **`OLLAMA_TIMEOUT`**: Maximum seconds allowed for an LLM request before timing out. Defaults to `300` seconds.
* **`OLLAMA_NUM_GPU`**: Set GPU offloading parameter. A value of `0` forces CPU mode (useful when running on machines without dedicated GPU acceleration). Set to `-1` to let Ollama autodetect and use GPU acceleration.
* **`OLLAMA_NUM_PREDICT`**: Sets the token response limit. Defaults to `512`.

---

## 🔄 Dynamic Model Switching via Settings APIs

Users can dynamically view or update settings via the backend APIs:
* **GET `/settings/config`**: Retrieves the currently active `code_model`, `analysis_model`, and RAG search depth.
* **POST `/settings/model`**: Updates the active analysis model on-the-fly.
* **POST `/settings/temperature`**: Updates the generation temperature parameter (default is `0.1` for deterministic outputs).
* **POST `/settings/expert`**: Updates the base system prompt or Ollama URL.
* **POST `/settings/rag`**: Modifies the search depth (retrieved context chunks count `k`) for RAG from the knowledge base.
