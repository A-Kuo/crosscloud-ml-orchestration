"""
Sample telemetry generator.

Produces a JSONL file of realistic routing_events rows that can be:
  - loaded into BigQuery for drift detection demonstration
  - used in unit / integration tests

Run:
    python telemetry/sample_telemetry.py --rows 1000 --out telemetry/sample_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def generate_rows(
    n_rows: int = 1000,
    tau: float = 2.0,
    base_time: datetime | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Generate `n_rows` synthetic routing event rows.

    Entropy is drawn from a mixture of two Gaussians to simulate a realistic
    bimodal distribution: one cluster of confident (low-entropy) inputs and
    one cluster of uncertain (high-entropy) inputs.
    """
    rng = np.random.default_rng(seed)

    if base_time is None:
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)

    rows = []
    # AR(1) entropy with burst-sensitive drift component.
    entropy_state = 1.7
    current_time = base_time
    for i in range(n_rows):
        # Poisson inter-arrival with occasional bursty traffic.
        arrival_rate = 1.8 if rng.random() > 0.08 else 0.2
        delta_seconds = max(1, int(rng.exponential(60.0 / arrival_rate)))
        current_time += timedelta(seconds=delta_seconds)

        # AR(1): entropy_t = c + phi * entropy_{t-1} + noise
        phi = 0.86
        noise = rng.normal(loc=0.0, scale=0.18)
        drift = 0.25 if rng.random() < 0.05 else 0.0
        entropy_state = 0.22 + phi * entropy_state + noise + drift
        h = max(0.0, float(entropy_state))

        destination = "gcp_cloud_run" if h < tau else "aws_sagemaker"
        error_prob = float(1 / (1 + np.exp(-(h - tau) * 3)))
        is_error = bool(rng.binomial(1, error_prob))

        ts = current_time
        probe_latency = float(rng.normal(loc=12.0, scale=2.5))

        if destination == "gcp_cloud_run":
            model_label = rng.choice(["1 star", "2 stars", "3 stars", "4 stars", "5 stars"])
            model_score = float(rng.uniform(0.6, 0.99))
            total_latency = probe_latency + float(rng.normal(loc=45.0, scale=8.0))
        else:
            model_label = rng.choice(["consistent", "hallucinated"])
            model_score = float(rng.uniform(0.5, 0.95))
            total_latency = probe_latency + float(rng.normal(loc=120.0, scale=20.0))

        label_delay_hours = int(rng.integers(24, 73))

        rows.append({
            "request_id": str(uuid.uuid4()),
            "timestamp_utc": ts.isoformat(),
            "label_available_utc": (ts + timedelta(hours=label_delay_hours)).isoformat(),
            "destination": destination,
            "h_route": round(h, 6),
            "tau": tau,
            "error_probability": round(error_prob, 6),
            "input_tokens": int(rng.integers(10, 128)),
            "probe_latency_ms": round(probe_latency, 2),
            "total_latency_ms": round(total_latency, 2),
            "probe_model": "distilbert-base-uncased",
            "is_error": is_error,
            "model_label": model_label,
            "model_score": round(model_score, 4),
            "metadata": {},
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample routing telemetry")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--tau", type=float, default=2.0, help="Routing threshold used in simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="telemetry/sample_rows.jsonl")
    args = parser.parse_args()

    rows = generate_rows(n_rows=args.rows, tau=args.tau, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    gcp_count = sum(1 for r in rows if r["destination"] == "gcp_cloud_run")
    aws_count = args.rows - gcp_count
    print(f"Wrote {args.rows} rows to {out_path}")
    print(f"  GCP Cloud Run: {gcp_count} ({100*gcp_count/args.rows:.1f}%)")
    print(f"  AWS SageMaker: {aws_count} ({100*aws_count/args.rows:.1f}%)")


if __name__ == "__main__":
    main()
