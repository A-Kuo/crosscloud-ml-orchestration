"""
CrossCloud ML Orchestration DAG.

Four core tasks run nightly:
  1. entropy_audit        — query BigQuery for the past 24h of entropy telemetry;
                            compute KL divergence against the rolling 30-day baseline.
  2. threshold_recalibrate — if drift detected, refit isotonic regression and push
                             new tau to AWS SSM / GCP Secret Manager.
  3. retrain_trigger      — if KL divergence exceeds the retrain bound, kick off a
                             Vertex AI Pipeline retraining job.
  4. health_check         — ping both Cloud Run and SageMaker endpoints; alert via
                             SNS Lambda if p99 latency exceeds SLA.

Cross-cloud hooks are mocked (via environment variables) so the DAG can run
in a local Airflow environment without real cloud credentials.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via Airflow Variables or environment variables
# ---------------------------------------------------------------------------
BIGQUERY_PROJECT = os.getenv("BQ_PROJECT", "crosscloud-demo")
BIGQUERY_DATASET = os.getenv("BQ_DATASET", "ml_telemetry")
TELEMETRY_TABLE = f"{BIGQUERY_PROJECT}.{BIGQUERY_DATASET}.routing_events"

KL_RECALIBRATE_BOUND = float(os.getenv("KL_RECALIBRATE_BOUND", "0.1"))  # nats
KL_RETRAIN_BOUND = float(os.getenv("KL_RETRAIN_BOUND", "0.3"))           # nats

GCP_CLOUD_RUN_URL = os.getenv("GCP_CLOUD_RUN_URL", "http://localhost:8080")
AWS_SAGEMAKER_URL = os.getenv("AWS_SAGEMAKER_URL", "http://localhost:8081")
LATENCY_SLA_MS = float(os.getenv("LATENCY_SLA_MS", "150.0"))

SSM_TAU_PARAMETER = os.getenv("SSM_TAU_PARAMETER", "/crosscloud/tau")
MOCK_CLOUD = os.getenv("MOCK_CLOUD", "true").lower() == "true"

RISK_TOLERANCE = float(os.getenv("RISK_TOLERANCE", "0.15"))

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
default_args = {
    "owner": "austin.kuo",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="crosscloud_ml_orchestration",
    description="Nightly entropy audit, threshold recalibration, drift-triggered retrain, and health check",
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "routing", "cross-cloud"],
) as dag:

    # -----------------------------------------------------------------------
    # Task 1: entropy_audit
    # -----------------------------------------------------------------------
    def entropy_audit(**context) -> dict:
        """
        Query BigQuery (or mock) for entropy scores from the past 24h.
        Compute KL divergence against the 30-day rolling baseline.
        Push results to XCom for downstream tasks.
        """
        logger.info("entropy_audit: fetching last-24h entropy distribution")

        if MOCK_CLOUD:
            current_entropies, baseline_entropies = _mock_entropy_distributions()
        else:
            current_entropies = _query_bigquery_entropies(hours=24)
            baseline_entropies = _query_bigquery_entropies(hours=24 * 30)

        kl = _kl_divergence(current_entropies, baseline_entropies)

        logger.info(
            "entropy_audit: KL(current||baseline)=%.4f  "
            "recalibrate_bound=%.2f  retrain_bound=%.2f",
            kl, KL_RECALIBRATE_BOUND, KL_RETRAIN_BOUND,
        )

        result = {
            "kl_divergence": kl,
            "should_recalibrate": kl > KL_RECALIBRATE_BOUND,
            "should_retrain": kl > KL_RETRAIN_BOUND,
            "n_current_samples": len(current_entropies),
            "n_baseline_samples": len(baseline_entropies),
            "current_mean_entropy": float(np.mean(current_entropies)),
            "baseline_mean_entropy": float(np.mean(baseline_entropies)),
            # Pass the raw arrays as lists for downstream calibration
            "current_entropies": current_entropies.tolist(),
        }
        context["ti"].xcom_push(key="audit_result", value=result)
        return result

    t_audit = PythonOperator(
        task_id="entropy_audit",
        python_callable=entropy_audit,
        provide_context=True,
    )

    # -----------------------------------------------------------------------
    # Task 2: threshold_recalibrate
    # -----------------------------------------------------------------------
    def threshold_recalibrate(**context) -> dict:
        """
        Pull audit result from XCom. If drift was detected, refit isotonic
        regression and push new tau to the parameter store.
        """
        audit: dict = context["ti"].xcom_pull(task_ids="entropy_audit", key="audit_result")
        if not audit["should_recalibrate"]:
            logger.info("threshold_recalibrate: no recalibration needed (KL=%.4f)", audit["kl_divergence"])
            return {"skipped": True, "reason": "KL below recalibrate bound"}

        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
        from router.threshold import ThresholdCalibrator
        from router.artifact import CalibrationArtifact, CalibrationArtifactManager

        entropies = np.array(audit["current_entropies"])

        # Generate synthetic error labels correlated with entropy
        # In production these come from ground-truth labels logged in BigQuery.
        if MOCK_CLOUD:
            error_labels = _mock_error_labels(entropies)
        else:
            error_labels = _query_error_labels_from_bigquery(entropies)

        # Load previous tau history from SSM / Secret Manager
        tau_history = _load_tau_history()

        calibrator = ThresholdCalibrator(risk_tolerance=RISK_TOLERANCE)
        result = calibrator.fit(entropies, error_labels, previous_tau_history=tau_history)

        # Safety gates: reject updates when calibration quality degrades.
        min_samples = int(os.getenv("CAL_MIN_SAMPLES", "50"))
        min_auroc = float(os.getenv("CAL_MIN_AUROC", "0.55"))
        min_auroc_gain = float(os.getenv("CAL_MIN_AUROC_GAIN", "0.02"))
        hist_arr = np.array(tau_history, dtype=float) if tau_history else np.array([result.tau], dtype=float)
        q1, q3 = float(np.quantile(hist_arr, 0.25)), float(np.quantile(hist_arr, 0.75))
        prev_auroc = float(audit.get("previous_auroc", min_auroc))
        if result.n_samples < min_samples:
            logger.warning("threshold_recalibrate: rejected update, samples below minimum")
            return {"skipped": True, "reason": "insufficient samples", "n_samples": result.n_samples}
        if result.auroc < min_auroc:
            logger.warning("threshold_recalibrate: rejected update, AUROC below minimum")
            return {"skipped": True, "reason": "auroc below minimum", "auroc": result.auroc}
        if (result.auroc - prev_auroc) < min_auroc_gain:
            logger.warning("threshold_recalibrate: rejected update, AUROC gain below threshold")
            return {"skipped": True, "reason": "auroc gain below threshold", "auroc": result.auroc}
        if not (q1 <= result.tau <= q3 or len(hist_arr) < 4):
            logger.warning("threshold_recalibrate: rejected update, tau outside historical interquartile range")
            return {"skipped": True, "reason": "tau outside historical IQR", "new_tau": result.tau}

        logger.info(
            "threshold_recalibrate: new tau=%.4f  AUROC=%.4f",
            result.tau, result.auroc,
        )

        _push_tau_to_parameter_store(result.tau)
        _save_tau_history(result.tau_history)
        manager = CalibrationArtifactManager()
        previous_tau = float(tau_history[-1]) if tau_history else None
        manager.save(
            CalibrationArtifact(
                tau=result.tau,
                auroc=result.auroc,
                fit_date=float(datetime.utcnow().timestamp()),
                n_samples=result.n_samples,
                prev_tau=previous_tau,
                risk_tolerance=RISK_TOLERANCE,
                notes="Auto-generated by Airflow threshold_recalibrate",
            )
        )

        output = {
            "new_tau": result.tau,
            "auroc": result.auroc,
            "n_samples": result.n_samples,
            "skipped": False,
        }
        context["ti"].xcom_push(key="calibration_result", value=output)
        return output

    t_recalibrate = PythonOperator(
        task_id="threshold_recalibrate",
        python_callable=threshold_recalibrate,
        provide_context=True,
    )

    # -----------------------------------------------------------------------
    # Task 3: retrain_trigger
    # -----------------------------------------------------------------------
    def retrain_trigger(**context) -> dict:
        """
        If KL divergence exceeded the retrain bound, invoke a Vertex AI
        Pipeline retraining job (or mock it locally).
        """
        audit: dict = context["ti"].xcom_pull(task_ids="entropy_audit", key="audit_result")
        if not audit["should_retrain"]:
            logger.info("retrain_trigger: retraining not needed (KL=%.4f)", audit["kl_divergence"])
            return {"triggered": False, "reason": "KL below retrain bound"}

        logger.info(
            "retrain_trigger: KL=%.4f > %.2f — triggering Vertex AI Pipeline",
            audit["kl_divergence"], KL_RETRAIN_BOUND,
        )

        if MOCK_CLOUD:
            job_id = _mock_vertex_pipeline_trigger()
        else:
            job_id = _trigger_vertex_ai_pipeline()

        return {"triggered": True, "job_id": job_id}

    t_retrain = PythonOperator(
        task_id="retrain_trigger",
        python_callable=retrain_trigger,
        provide_context=True,
    )

    # -----------------------------------------------------------------------
    # Task 4: health_check
    # -----------------------------------------------------------------------
    def health_check(**context) -> dict:
        """
        Ping both Cloud Run and SageMaker /health endpoints.
        Alert if p99 latency exceeds SLA or if an endpoint is down.

        Note: Uses 5 probes per endpoint; with 5 samples, np.percentile(..., 99)
        equals the max latency, not a true p99. For production, increase probe count.
        """
        import time
        import urllib.request
        import urllib.error

        results = {}
        for name, url in [
            ("gcp_cloud_run", GCP_CLOUD_RUN_URL),
            ("aws_sagemaker", AWS_SAGEMAKER_URL),
        ]:
            if MOCK_CLOUD:
                results[name] = {"status": "ok", "latency_ms": 42.0, "mocked": True}
                logger.info("health_check [mock]: %s ok", name)
                continue

            latencies = []
            endpoint_ok = True
            # 5 probes: np.percentile(..., 99) is the max; true p99 would need more samples.
            for _ in range(5):
                t0 = time.perf_counter()
                try:
                    urllib.request.urlopen(f"{url}/health", timeout=5)
                    latencies.append((time.perf_counter() - t0) * 1000)
                except Exception as exc:
                    logger.error("health_check: %s unreachable — %s", name, exc)
                    endpoint_ok = False
                    break

            if not endpoint_ok:
                results[name] = {"status": "down"}
                _send_alert(f"{name} endpoint unreachable")
                continue

            p99 = float(np.percentile(latencies, 99))
            status = "ok" if p99 < LATENCY_SLA_MS else "degraded"
            if status == "degraded":
                _send_alert(f"{name} p99 latency {p99:.0f}ms exceeds SLA {LATENCY_SLA_MS:.0f}ms")
            results[name] = {"status": status, "p99_ms": p99}
            logger.info("health_check: %s status=%s p99=%.1fms", name, status, p99)

        return results

    t_health = PythonOperator(
        task_id="health_check",
        python_callable=health_check,
        provide_context=True,
    )

    # -----------------------------------------------------------------------
    # Task dependencies
    # -----------------------------------------------------------------------
    t_audit >> [t_recalibrate, t_retrain]
    t_recalibrate >> t_health
    t_retrain >> t_health


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _mock_entropy_distributions() -> tuple[np.ndarray, np.ndarray]:
    """Simulate 24h current vs 30-day baseline with slight distributional shift."""
    rng = np.random.default_rng(seed=42)
    baseline = rng.normal(loc=1.8, scale=0.4, size=5000)
    # Slight shift toward higher entropy (simulating harder inputs)
    current = rng.normal(loc=2.0, scale=0.5, size=500)
    return np.clip(current, 0, None), np.clip(baseline, 0, None)


def _mock_error_labels(entropies: np.ndarray) -> np.ndarray:
    """
    Simulate ground-truth error labels: higher entropy -> higher error probability.
    Uses a sigmoid on the centred entropy.
    """
    rng = np.random.default_rng(seed=7)
    prob_error = 1 / (1 + np.exp(-(entropies - np.median(entropies)) * 2))
    return rng.binomial(1, prob_error).astype(float)


def _kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 50) -> float:
    """
    Estimate KL(P||Q) from samples using histogram density estimation.
    D_KL(P||Q) = sum P(x) * log(P(x)/Q(x))
    """
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    _eps = 1e-10

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)
    p_hist = np.clip(p_hist, _eps, None)
    q_hist = np.clip(q_hist, _eps, None)
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def _query_bigquery_entropies(hours: int) -> np.ndarray:
    """
    Query BigQuery for entropy scores from the past `hours` window.
    Returns a numpy array of h_route values.
    """
    from google.cloud import bigquery
    client = bigquery.Client(project=BIGQUERY_PROJECT)
    query = f"""
        SELECT h_route
        FROM `{TELEMETRY_TABLE}`
        WHERE timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    """
    rows = list(client.query(query).result())
    return np.array([r["h_route"] for r in rows], dtype=np.float64)


def _query_error_labels_from_bigquery(entropies: np.ndarray) -> np.ndarray:
    """
    In production, fetch (entropy, is_error) pairs from BigQuery.
    Stub that returns zeros until real labels are available.
    """
    logger.warning("_query_error_labels_from_bigquery: returning zero labels (stub)")
    return np.zeros(len(entropies), dtype=np.float64)


def _load_tau_history() -> list:
    """Load tau history from local file (stand-in for SSM / Secret Manager)."""
    path = "/tmp/tau_history.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_tau_history(history: list) -> None:
    with open("/tmp/tau_history.json", "w") as f:
        json.dump(history, f)


def _push_tau_to_parameter_store(tau: float) -> None:
    """Push updated tau to AWS SSM or GCP Secret Manager."""
    if MOCK_CLOUD:
        logger.info("_push_tau_to_parameter_store [mock]: tau=%.4f -> %s", tau, SSM_TAU_PARAMETER)
        with open("/tmp/current_tau.json", "w") as f:
            json.dump({"tau": tau}, f)
        return
    import boto3
    ssm = boto3.client("ssm")
    ssm.put_parameter(
        Name=SSM_TAU_PARAMETER,
        Value=str(tau),
        Type="String",
        Overwrite=True,
    )
    logger.info("Pushed tau=%.4f to SSM parameter %s", tau, SSM_TAU_PARAMETER)


def _trigger_vertex_ai_pipeline() -> str:
    """Invoke Vertex AI Pipeline retraining job via Google Cloud SDK."""
    from google.cloud import aiplatform
    aiplatform.init(project=BIGQUERY_PROJECT, location="us-central1")
    job = aiplatform.PipelineJob(
        display_name="crosscloud-retrain",
        template_path=os.getenv("VERTEX_PIPELINE_TEMPLATE", "gs://crosscloud-demo/pipelines/retrain.json"),
        parameter_values={
            "training_data_uri": os.getenv("TRAINING_DATA_URI", "gs://crosscloud-demo/data/"),
        },
    )
    job.submit()
    logger.info("Submitted Vertex AI Pipeline job: %s", job.resource_name)
    return job.resource_name


def _mock_vertex_pipeline_trigger() -> str:
    job_id = "mock-vertex-job-001"
    logger.info("_mock_vertex_pipeline_trigger: job_id=%s", job_id)
    return job_id


def _send_alert(message: str) -> None:
    """Send alert via SNS (production) or log (mock)."""
    if MOCK_CLOUD:
        logger.warning("ALERT [mock]: %s", message)
        return
    import boto3
    sns = boto3.client("sns")
    sns.publish(
        TopicArn=os.getenv("SNS_ALERT_TOPIC", "arn:aws:sns:us-east-1:000000000000:crosscloud-alerts"),
        Message=message,
        Subject="CrossCloud ML Alert",
    )
