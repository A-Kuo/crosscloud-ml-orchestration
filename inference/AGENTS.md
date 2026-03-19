# inference/ — Agent Context

## Purpose

Deployable FastAPI inference server and model wrappers. This directory is what gets packaged into the Docker image.

## Module Responsibilities

### server.py
- FastAPI app with three endpoints: `POST /infer`, `POST /entropy`, `GET /health`.
- Controlled by `SERVE_TARGET` env var:
  - `router` — full pipeline (probe + route + model). Used locally and in integration tests.
  - `absa` — only the ABSA model. Deployed on GCP Cloud Run.
  - `hallucination` — only the hallucination scorer. Deployed on AWS SageMaker.
- Lifespan handler loads models once at startup. The `_router` global is set only in router mode.
- Request/response schemas are Pydantic `BaseModel` subclasses.

### models.py
- `load_model(target)` — `@lru_cache`'d HuggingFace pipeline loader. Two targets: `"absa"` and `"hallucination"`.
- `run_absa(text)` / `run_hallucination_scorer(text, hypothesis)` — thin wrappers returning normalised dicts.
- Models are portfolio stand-ins. `nlptown/bert-base-multilingual-uncased-sentiment` for ABSA, `vectara/hallucination_evaluation_model` for hallucination detection.

### Dockerfile
- Base: `python:3.11-slim`.
- Copies `router/` and `inference/` into the image.
- Downloads model weights at build time (`RUN python -c "from transformers import ..."`) to avoid cold-start latency.
- Build args: `PROBE_MODEL`, `SERVE_TARGET`.
- Exposes port 8080. Health check hits `/health`.

## When Modifying

- New endpoints: add to `server.py`, add Pydantic schemas, add to the `tags` for OpenAPI grouping.
- New model targets: add to `ModelTarget` literal in `models.py`, add `load_model` branch, add `run_*` wrapper, update `server.py` routing logic.
- Keep the server stateless between requests (all state is in the `_router` global set at startup).
- The `sys.path.insert` hack in `lifespan` makes the `router` package importable from inside the Docker container. If you restructure the project, update this path.
