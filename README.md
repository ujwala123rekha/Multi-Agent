# Production-Grade Multi-Agent AI System

A fully local, modular, and extensible multi-agent pipeline built with LangChain, LangGraph, and LangSmith.

## Architecture

- **Agent 1 (Planner):** Uses `llama3.2:3b` to decompose queries and route tasks.
- **Agent 2 (Researcher):** Conditional node. Performs web search (Tavily), fetches content, reranks results (Cross-Encoder), and synthesizes a report.
- **Agent 3 (Image Gen):** Conditional node. Optimizes image prompts and calls HuggingFace Inference API (FLUX.1-schnell).
- **Synthesizer:** Merges all outputs into a final coherent response.

## Tech Stack

- **Orchestration:** LangGraph (StateGraph)
- **Local LLMs:** Ollama (`llama3.2:3b`, `llama3.2:1b`)
- **Tracing:** LangSmith
- **Search:** Tavily API
- **Embeddings/Reranking:** Sentence-Transformers (Local)
- **Image Gen:** HuggingFace API

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Update the `.env` file with your API keys:
   - `LANGCHAIN_API_KEY`: From [LangSmith](https://smith.langchain.com/)
   - `TAVILY_API_KEY`: From [Tavily](https://tavily.com/)
   - `HF_TOKEN`: From [HuggingFace](https://huggingface.co/settings/tokens)

3. **Run Ollama:**
   Ensure Ollama is running locally with the required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull llama3.2:1b
   ```

## Usage

Run the pipeline from the command line:
```bash
python main.py "Explain quantum computing and generate a visual for it"
```

## Features

- **FAANG-Level Engineering:** Typed schemas, async-first architecture, retry logic.
- **Observability:** End-to-end tracing in LangSmith.
- **Local Inference:** Zero cost for LLM usage via Ollama.
- **Graceful Degradation:** Handles search failures or tool errors without crashing the pipeline.
