# CrossCloud: Entropy-Based Multi-Cloud ML Inference Router

[![CI](https://github.com/austinkuo/crosscloud-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/austinkuo/crosscloud-orchestration/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A multi-cloud ML inference router that uses **Shannon entropy of transformer
attention distributions** as a routing signal. Low-entropy (confident) requests
go to a fast GCP Cloud Run endpoint; high-entropy (uncertain) requests are
escalated to a heavier AWS SageMaker model. The routing threshold is
recalibrated nightly via an Airflow DAG that detects distributional drift
with KL divergence.

---

## Architecture

```
                          ┌──────────────────────┐
                          │     Input text        │
                          └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ AttentionEntropyProbe │
                          │    (DistilBERT)       │
                          └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ H_route = mean entropy│
                          │  across L layers,     │
                          │  H heads, T tokens    │
                          └──────────┬───────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                 H < τ (confident)            H ≥ τ (uncertain)
                      │                             │
                      ▼                             ▼
           ┌──────────────────┐          ┌──────────────────┐
           │  GCP Cloud Run   │          │  AWS SageMaker   │
           │  BERT ABSA       │          │  Hallucination   │
           │  (fast path)     │          │  scorer (heavy)  │
           └──────────────────┘          └──────────────────┘
```

**Nightly orchestration (Airflow):**

```
entropy_audit ──┬──► threshold_recalibrate ──┐
                │                             ├──► health_check
                └──► retrain_trigger ────────┘
```

## Mathematical Foundation

| Concept | Formula | Role |
|---------|---------|------|
| Attention entropy | H(α) = −Σ p(a_i) log p(a_i) | Per-head uncertainty signal |
| Routing signal | H_route = mean(H) over L layers × H heads | Scalar input to routing decision |
| Threshold calibration | Isotonic regression: f(H) ≈ P(error \| H_route) | Adaptive τ from labeled data |
| Drift detection | D_KL(P_current \|\| Q_baseline) on entropy histograms | Triggers recalibration or retraining |

Threshold τ is not a fixed constant. It is the minimum H where
P(error | H) ≥ ε (risk tolerance, default 0.15), recalibrated nightly from
(entropy, error) pairs stored in BigQuery.

## Project Structure

```
crosscloud-orchestration/
├── router/
│   ├── entropy.py           # AttentionEntropyProbe: attention → entropy → H_route
│   ├── threshold.py         # ThresholdCalibrator: isotonic regression, AUROC, tau
│   └── router.py            # InferenceRouter: probe + calibrator → RoutingDecision
├── inference/
│   ├── server.py            # FastAPI: /infer, /entropy, /health
│   ├── models.py            # HuggingFace pipeline wrappers (ABSA, hallucination)
│   └── Dockerfile           # Multi-target: router | absa | hallucination
├── airflow/dags/
│   └── crosscloud_orchestration.py   # 4-task nightly DAG
├── telemetry/
│   ├── bigquery_schema.json          # routing_events table definition
│   └── sample_telemetry.py           # Synthetic JSONL generator
├── tests/                            # 45 fast + 4 slow tests
├── .github/workflows/ci.yml          # GitHub Actions CI
├── pyproject.toml
├── requirements.txt
└── DEVELOPMENT.md                    # Architecture deep-dive for contributors
```

## Quick Start

### Install

```bash
git clone https://github.com/austinkuo/crosscloud-orchestration.git
cd crosscloud-orchestration
pip install -r requirements.txt
```

### Run Tests

```bash
pytest                   # fast tests — no model download, ~40s
pytest --runslow         # full suite — downloads DistilBERT (~250 MB)
```

### Start the Inference Server

```bash
cd inference
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```

```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was absolutely outstanding."}'
```

### Docker

```bash
# Router mode (local development)
docker build -f inference/Dockerfile -t crosscloud-router .
docker run -p 8080:8080 -e SERVE_TARGET=router crosscloud-router

# GCP Cloud Run target
docker build -f inference/Dockerfile -t crosscloud-absa --build-arg SERVE_TARGET=absa .

# AWS SageMaker target
docker build -f inference/Dockerfile -t crosscloud-hallucination --build-arg SERVE_TARGET=hallucination .
```

### Generate Telemetry Data

```bash
python telemetry/sample_telemetry.py --rows 2000 --out telemetry/sample_rows.jsonl
```

### Airflow (Local, Mocked Cloud)

```bash
pip install apache-airflow
export AIRFLOW_HOME=$(pwd)/airflow MOCK_CLOUD=true
airflow db init
airflow dags test crosscloud_ml_orchestration $(date +%Y-%m-%d)
```

## Airflow DAG

| Task | Trigger | Action |
|------|---------|--------|
| `entropy_audit` | Always | Query 24h entropy from BigQuery; compute KL divergence vs 30-day baseline |
| `threshold_recalibrate` | D_KL > 0.1 | Refit isotonic regression; push new τ to AWS SSM / GCP Secret Manager |
| `retrain_trigger` | D_KL > 0.3 | Submit Vertex AI Pipeline retraining job |
| `health_check` | Always | Ping both cloud endpoints; alert if p99 latency exceeds 150 ms SLA |

## Configuration

All thresholds and endpoints are configurable via environment variables. See
[DEVELOPMENT.md](DEVELOPMENT.md) for the full reference.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVE_TARGET` | `router` | Server mode: `router`, `absa`, or `hallucination` |
| `TAU` | 2.0 | Override routing threshold |
| `PROBE_MODEL` | `distilbert-base-uncased` | HuggingFace model ID for the entropy probe |
| `MOCK_CLOUD` | `true` | Gate real GCP/AWS SDK calls in Airflow |

## Scope and Limitations

**Demonstrated:**
- Entropy computation module with full test coverage and AUROC benchmarks
- Dockerised FastAPI server deployable to Cloud Run or SageMaker
- Airflow DAG with mocked cross-cloud hooks, runnable locally
- BigQuery telemetry schema with synthetic data generator

**Out of scope (not claimed):**
- Production IAM, VPC, and networking configuration
- Real financial data or proprietary model weights
- Load testing beyond local Docker simulation

The value of this project is in the **information-theoretic routing design
and cross-cloud orchestration architecture**, not in claiming production
readiness that does not exist.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for setup instructions,
coding conventions, and the pull request workflow.

## Citation

```bibtex
@software{crosscloud2026,
  author = {Kuo, Austin},
  title  = {{CrossCloud}: Entropy-Based Multi-Cloud {ML} Inference Routing},
  year   = {2026},
  url    = {https://github.com/austinkuo/crosscloud-orchestration},
}
```

## License

[Apache 2.0](LICENSE)
