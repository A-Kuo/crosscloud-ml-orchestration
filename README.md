# Cross-Cloud ML Orchestration

**Entropy-based multi-cloud inference routing with calibrated uncertainty**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.7+-green.svg)](https://airflow.apache.org)
[![GCP](https://img.shields.io/badge/GCP%20Vertex%20AI-supported-blue.svg)](https://cloud.google.com/vertex-ai)
[![AWS](https://img.shields.io/badge/AWS%20SageMaker-supported-orange.svg)](https://aws.amazon.com/sagemaker)
[![Status](https://img.shields.io/badge/Status-Beta-green.svg)]()

> *"Most multi-cloud ML implementations route by latency or cost. They ignore the most important signal: which model actually knows what it's talking about for this specific input."*

---

## The Problem

You have the same model deployed on GCP Vertex and AWS SageMaker. Which one do you call?

**Wrong answers:**
- Always the cheaper one (saves money, ignores accuracy)
- Round-robin (ignores model confidence entirely)
- Always the faster one (latency != correctness)

**Right answer:**
- Route to the model instance with **highest confidence calibrated to the specific input** — and have a principled way to measure that confidence.

---

## The Solution: Entropy-Based Routing

This system implements a novel routing strategy: **use predictive entropy as a routing signal.**

For each incoming inference request:
1. Send to both GCP and AWS endpoints (or sample based on cost constraints)
2. Measure the entropy of each model's output distribution
3. Route subsequent similar inputs to the provider with lower average entropy
4. Continuously recalibrate using isotonic regression on held-out validation

**Why entropy?**
- It's a mathematically grounded measure of uncertainty
- It works across model architectures (no architecture-specific assumptions)
- It captures "knowing what you don't know" — high entropy means the model is genuinely uncertain

**Why isotonic regression for calibration?**
- Non-parametric calibration that respects monotonicity
- Handles the non-linear relationship between raw model confidence and true accuracy
- Standard technique in probabilistic forecasting, applied here to routing decisions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE REQUEST                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ENTROPY ROUTING DECISION                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  GCP Vertex AI  │    │  AWS SageMaker  │    │  Entropy-Based  │  │
│  │   (Candidate)   │    │   (Candidate)   │    │    Aggregator   │  │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────┘  │
│           │                      │                                   │
│           ▼                      ▼                                   │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │  Entropy Score  │    │  Entropy Score  │                        │
│  │    H(p_gcp)     │    │    H(p_aws)     │                        │
│  └────────┬────────┘    └────────┬────────┘                        │
│           │                      │                                   │
│           └──────────┬───────────┘                                   │
│                      ▼                                              │
│           ┌──────────────────────┐                                  │
│           │   ROUTING DECISION   │  ← Lower entropy = higher trust │
│           └──────────┬───────────┘                                  │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │    INFERENCE   │
              │   RESPONSE     │
              └────────────────┘
```

---

## Telemetry & Feedback Loop

Every routing decision is logged to BigQuery:
- Input features (hashed for privacy)
- Routing destination
- Entropy scores from both providers
- Final output
- Ground truth (if available)

This enables:
- **Retrospective analysis**: Was the routing decision correct?
- **Calibration monitoring**: Is isotonic regression still well-calibrated?
- **Cost optimization**: When can we safely route to cheaper, higher-entropy provider?

---

## Components

### 1. Airflow DAGs

Orchestration for:
- Model deployment synchronization (same model version on both clouds)
- Calibration data collection runs
- Retraining isotonic regression calibrators
- Cost and latency metric aggregation

### 2. Routing Service

FastAPI service that:
- Accepts inference requests
- Queries entropy cache or computes on-the-fly
- Routes to appropriate cloud provider
- Logs telemetry

### 3. Calibration Pipeline

Isotonic regression training:
- Collects (predicted probability, actual outcome) pairs
- Fits isotonic calibrator
- Evaluates calibration on held-out set
- Deploys updated calibrator if improved

---

## Key Innovation

Most multi-cloud ML papers focus on:
- Cost optimization (bin packing, spot instances)
- Latency reduction (geographic routing)
- Availability (failover)

This system adds:
- **Accuracy-aware routing** — route to the model that actually performs better for this input type
- **Uncertainty-calibrated aggregation** — when you need both responses, combine them weighted by inverse entropy

The entropy signal is particularly valuable for:
- Heterogeneous model versions (GCP running v2.1, AWS running v2.0)
- Different hardware backends (GPU vs TPU inference with subtle numeric differences)
- Model drift detection (entropy spike = potential distribution shift)

---

## Configuration Example

```yaml
# config/production.yaml
providers:
  gcp:
    project: my-project
    endpoint: projects/my-project/locations/us-central1/endpoints/my-endpoint
    cost_per_1k_requests: 0.50
    latency_p99_ms: 120
  
  aws:
    endpoint: https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invocations
    cost_per_1k_requests: 0.45
    latency_p99_ms: 150

routing:
  strategy: entropy_based
  fallback: gcp  # if entropy scores are equal
  calibration_window_hours: 168  # 1 week

telemetry:
  bigquery_dataset: ml_routing
  bigquery_table: routing_decisions
```

---

## Installation

```bash
git clone https://github.com/A-Kuo/crosscloud-ml-orchestration.git
cd crosscloud-ml-orchestration
pip install -r requirements.txt
```

## Usage

```python
from crosscloud_ml import EntropyRouter

router = EntropyRouter.from_config("config/production.yaml")

# Route based on entropy
response = router.predict(
    features=my_input_features,
    return_metadata=True  # includes entropy scores, routing decision
)

# Or get both and aggregate
responses = router.predict_all(
    features=my_input_features,
    aggregation="entropy_weighted"  # weighted by inverse entropy
)
```

---

## Research Context

This implementation extends prior work on [entropy-based hallucination detection](https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence) from the LLM domain to the multi-cloud orchestration domain. The core insight — that entropy is a reliable signal of model uncertainty — transfers across problem types.

See also:
- [Multi-Source-Clinical-Data-Engineering-Platform](https://github.com/A-Kuo/Multi-Source-Clinical-Data-Engineering-Platform) — Uses similar entropy signals for anomaly detection

---

## Status

**Production Ready (April 2026)**

- ✅ Entropy-based routing engine
- ✅ Airflow orchestration DAGs
- ✅ BigQuery telemetry pipeline
- ✅ Isotonic regression calibration
- ✅ GCP + AWS provider implementations
- 🔄 Azure support (in development)
- 🔄 Automatic drift detection (experimentation)

---

## Citation

```bibtex
@software{crosscloud_ml_2026,
  author = {A-Kuo},
  title = {Cross-Cloud ML Orchestration: Entropy-Based Multi-Cloud Inference Routing},
  url = {https://github.com/A-Kuo/crosscloud-ml-orchestration},
  year = {2026}
}
```

---

*Route to confidence, not just convenience. April 2026.*
