# Skill: Deployment & Infrastructure Guide

## When to Use

Use this skill when working on Docker builds, cloud deployment configuration, environment variable setup, or the Airflow orchestration layer.

## Deployment Modes

The same Docker image supports three modes via `SERVE_TARGET`:

| Mode | Target Cloud | What Runs | Endpoint |
|------|-------------|-----------|----------|
| `router` | Local / dev | Full pipeline: probe → route → model | All endpoints |
| `absa` | GCP Cloud Run | BERT sentiment model only | `/infer`, `/health` |
| `hallucination` | AWS SageMaker | Hallucination scorer only | `/infer`, `/health` |

## Docker Builds

```bash
# Local development (router mode, includes all models)
docker build -f inference/Dockerfile -t crosscloud-router .
docker run -p 8080:8080 -e SERVE_TARGET=router crosscloud-router

# GCP Cloud Run image (only ABSA model baked in)
docker build -f inference/Dockerfile -t crosscloud-absa \
  --build-arg SERVE_TARGET=absa .

# AWS SageMaker image (only hallucination model)
docker build -f inference/Dockerfile -t crosscloud-hallucination \
  --build-arg SERVE_TARGET=hallucination .
```

### Build Args

- `PROBE_MODEL` — HuggingFace model ID baked into image at build time (default: `distilbert-base-uncased`)
- `SERVE_TARGET` — sets the default `SERVE_TARGET` env var in the image

Model weights are downloaded during `docker build` (the `RUN python -c "..."` step) so the container starts instantly without network access.

## GCP Cloud Run Deployment

```bash
# Tag and push to GCR
docker tag crosscloud-absa gcr.io/$PROJECT_ID/crosscloud-absa
docker push gcr.io/$PROJECT_ID/crosscloud-absa

# Deploy
gcloud run deploy crosscloud-absa \
  --image gcr.io/$PROJECT_ID/crosscloud-absa \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars SERVE_TARGET=absa \
  --region us-central1
```

Health check is automatic — Cloud Run probes `GET /health`.

## AWS SageMaker Deployment

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker tag crosscloud-hallucination $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/crosscloud-hallucination
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/crosscloud-hallucination

# Create SageMaker endpoint (via boto3 or AWS CLI)
# The image must serve on port 8080 and respond to /health
```

## Airflow Setup (Local)

```bash
pip install apache-airflow apache-airflow-providers-google apache-airflow-providers-amazon

export AIRFLOW_HOME=$(pwd)/airflow
export MOCK_CLOUD=true

airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@example.com

# Test a single DAG run
airflow dags test crosscloud_ml_orchestration 2026-03-18

# Run the scheduler + webserver
airflow scheduler &
airflow webserver --port 8082
```

## Airflow DAG Task Graph

```
entropy_audit ──┬──► threshold_recalibrate ──┐
                │                             ├──► health_check
                └──► retrain_trigger ────────┘
```

- `entropy_audit` always runs. Computes KL divergence.
- `threshold_recalibrate` runs isotonic regression only if KL > 0.1.
- `retrain_trigger` submits Vertex AI Pipeline only if KL > 0.3.
- `health_check` always runs last. Pings both endpoints.

## Environment Variable Reference (Full)

### Inference Server
| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVE_TARGET` | `router` | Operating mode |
| `TAU` | (calibrator default) | Override routing threshold |
| `PROBE_MODEL` | `distilbert-base-uncased` | Entropy probe model |

### Airflow DAG
| Variable | Default | Purpose |
|----------|---------|---------|
| `MOCK_CLOUD` | `true` | Mock all cloud SDK calls |
| `BQ_PROJECT` | `crosscloud-demo` | BigQuery project |
| `BQ_DATASET` | `ml_telemetry` | BigQuery dataset |
| `KL_RECALIBRATE_BOUND` | `0.1` | Nats: trigger recalibration |
| `KL_RETRAIN_BOUND` | `0.3` | Nats: trigger full retraining |
| `GCP_CLOUD_RUN_URL` | `http://localhost:8080` | Health check target |
| `AWS_SAGEMAKER_URL` | `http://localhost:8081` | Health check target |
| `LATENCY_SLA_MS` | `150.0` | p99 latency SLA |
| `RISK_TOLERANCE` | `0.15` | Epsilon for calibration |
| `SSM_TAU_PARAMETER` | `/crosscloud/tau` | AWS SSM parameter name |

## Portfolio Scope Reminder

This is a portfolio project. When making deployment changes:
- Document what *would* happen in production, but keep `MOCK_CLOUD=true` as the default.
- Never commit real credentials, IAM configs, or account IDs.
- Dockerfile should remain buildable without cloud credentials.
