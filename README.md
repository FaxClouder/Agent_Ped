# Ped-Agent

Ped-Agent is an agentic toolkit for pedestrian-flow literature QA, experiment-plan
evaluation, structured scenario analysis, and optional video-to-trajectory extraction.

This repository currently contains the Phase 1 scaffold:

- Python package layout under `src/ped_agent`
- YAML configuration system with environment-variable interpolation
- Lightweight LLM factory and LangGraph-compatible routing graph
- Pydantic data models for literature, scenario, and trajectory data
- Module boundaries for RAG, analysis, experiment evaluation, vision plugins, and evals
- Pytest smoke tests for the scaffold

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest
ped-agent "How should I evaluate a pedestrian evacuation experiment?"
```

Optional extras:

```bash
pip install -e ".[rag]"
pip install -e ".[vision]"
```

Copy `config/.env.example` to `.env` and fill API keys when enabling real model,
LangSmith, RAG, or vision backends.
