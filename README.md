# Cross-Cloud ML Inference Orchestration

**GCP Vertex AI + AWS SageMaker with Entropy-Based Routing**  
Austin Kuo · aus.kuo03@gmail.com · Portfolio Project

---

## Overview

A production-grade, multi-cloud ML inference pipeline that routes requests across GCP and AWS based on **Shannon entropy of transformer attention distributions** — a mathematically-grounded uncertainty signal derived from model internals.

| Entropy | Decision | Cloud |
|---------|----------|-------|
| H < τ (confident) | Fast path | GCP Cloud Run → BERT ABSA |
| H ≥ τ (uncertain) | Escalate | AWS SageMaker → Hallucination scorer |

---

## Project Structure

```
CrossCloud Orchestration/
├── router/
│   ├── entropy.py          # Shannon entropy from attention distributions
│   ├── threshold.py        # Isotonic regression threshold calibration
│   └── router.py           # Routing logic + RoutingDecision telemetry record
├── inference/
│   ├── models.py           # HuggingFace pipeline wrappers (ABSA + hallucination)
│   ├── server.py           # FastAPI inference server
│   └── Dockerfile          # Deployable to Cloud Run or SageMaker
├── airflow/
│   └── dags/
│       └── crosscloud_orchestration.py  # Nightly orchestration DAG
├── telemetry/
│   ├── bigquery_schema.json             # routing_events table schema
│   └── sample_telemetry.py             # Synthetic telemetry generator
├── tests/
│   ├── test_entropy.py
│   ├── test_threshold.py
│   └── test_router.py
├── requirements.txt
└── README.md
```

---

## Math

**Attention entropy (per head):**
```
H(α_h_i) = −Σⱼ α_h_i_j · log(α_h_i_j)
```

**Aggregated routing signal:**
```
H_route = (1 / L·H) · Σ_{l,h,i} H(α_h_i)
```

**Threshold calibration (isotonic regression):**  
Fit monotone function `f: ℝ → [0,1]` where `f(H) ≈ P(error | H_route)`.  
τ is the minimum entropy where `f(H) ≥ ε` (risk tolerance).

**Drift detection (KL divergence):**
```
D_KL(P ∥ Q) = Σ P(x) · log[P(x)/Q(x)]
```
- `D_KL > 0.1 nats` → recalibrate τ  
- `D_KL > 0.3 nats` → trigger full retraining

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run unit tests

```bash
pytest                      # fast tests (no model download)
pytest --runslow            # includes DistilBERT model tests
```

### Generate sample telemetry

```bash
python telemetry/sample_telemetry.py --rows 2000 --out telemetry/sample_rows.jsonl
```

### Run inference server locally (router mode)

```bash
cd inference
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```

Then:
```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was absolutely outstanding."}'
```

### Build and run Docker

```bash
# Local router mode
docker build -f inference/Dockerfile -t crosscloud-router .
docker run -p 8080:8080 -e SERVE_TARGET=router crosscloud-router

# GCP Cloud Run mode (ABSA fast path)
docker build -f inference/Dockerfile -t crosscloud-absa \
  --build-arg SERVE_TARGET=absa .

# AWS SageMaker mode (hallucination scorer)
docker build -f inference/Dockerfile -t crosscloud-hallucination \
  --build-arg SERVE_TARGET=hallucination .
```

### Run Airflow DAG locally

```bash
pip install apache-airflow
export AIRFLOW_HOME=$(pwd)/airflow
export MOCK_CLOUD=true
airflow db init
airflow dags test crosscloud_ml_orchestration $(date +%Y-%m-%d)
```

---

## Airflow DAG Tasks

| Task | Trigger | Action |
|------|---------|--------|
| `entropy_audit` | Always | Query BigQuery 24h entropy; compute KL vs 30-day baseline |
| `threshold_recalibrate` | KL > 0.1 | Refit isotonic regression; push τ to AWS SSM |
| `retrain_trigger` | KL > 0.3 | Submit Vertex AI Pipeline retraining job |
| `health_check` | Always | Ping both endpoints; alert if p99 > 150ms SLA |

---

## Scope & Honest Framing

**In scope (demonstrable):**
- Entropy computation module with unit tests and AUROC benchmarks
- Dockerized inference server deployable to Cloud Run or SageMaker
- Airflow DAG with mocked cross-cloud hooks, runnable locally
- BigQuery schema and synthetic telemetry data

**Out of scope:**
- Full GCP/AWS account setup and IAM configuration
- Real financial data or proprietary model weights
- End-to-end load testing beyond local Docker simulation

The value of this project is in **architectural reasoning and mathematical rigor**, not in claiming production readiness that does not exist.
