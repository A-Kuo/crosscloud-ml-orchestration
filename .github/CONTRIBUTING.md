# Contributing

Thank you for your interest in CrossCloud Router. This document covers setup,
conventions, and workflow for contributors.

## Development Setup

```bash
# Clone and install
git clone https://github.com/austinkuo/crosscloud-orchestration.git
cd crosscloud-orchestration
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running Tests

```bash
# Fast tests (no model download, ~40s)
pytest

# Full suite including DistilBERT model tests (~5 min, downloads ~250 MB on first run)
pytest --runslow

# Single module
pytest tests/test_entropy.py -v

# With coverage
pytest --cov=router --cov-report=html
```

## Code Style

- **Python 3.11+** with type hints on all public APIs.
- **Dataclasses** for data containers; all expose `.to_dict()`.
- **Structured logging** via `structlog` (server) and `logging` (library).
- **Docstrings** on all public classes and functions. No narrating comments.

## Project Layout

Each source directory contains an `AGENTS.md` file that documents module
responsibilities, invariants, and modification guidance. Read these before
changing a module.

| Directory | Purpose |
|-----------|---------|
| `router/` | Core routing: entropy probe, threshold calibration, routing decisions |
| `inference/` | FastAPI server and HuggingFace model wrappers |
| `airflow/dags/` | Nightly orchestration DAG (mocked cloud calls by default) |
| `telemetry/` | BigQuery schema and synthetic data generation |
| `tests/` | pytest suite (fast + slow tiers) |

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Ensure all fast tests pass (`pytest`).
3. If your change touches `RoutingDecision`, update both `to_dict()` and
   `telemetry/bigquery_schema.json`.
4. Open a pull request against `main` using the PR template.

## Mathematical Invariants

These properties are tested and must be preserved:

- **Entropy bounds:** Uniform attention produces `log(T)` (maximum); deterministic attention produces ~0.
- **Routing boundary:** `H >= tau` always routes to AWS SageMaker; `H < tau` always to GCP Cloud Run.
- **Isotonic monotonicity:** `predict_error_prob()` is monotone non-decreasing.
- **Numerical stability:** Probabilities clipped to `[1e-12, 1.0]` before `log()`.
