"""
Compute routing regret from historical telemetry rows.

Regret is defined as the extra expected error cost incurred by the chosen route
compared with a counterfactual alternative route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_regret(rows: list[dict]) -> dict:
    regrets = []
    for row in rows:
        err = 1.0 if row.get("is_error") else 0.0
        # If routed to GCP and error occurred, counterfactual AWS could be safer.
        # If routed to AWS and no error occurred, cheaper GCP could be preferable.
        destination = row.get("destination")
        if destination == "gcp_cloud_run":
            counterfactual_cost = 0.6 * err
            observed_cost = 1.0 * err
        else:
            counterfactual_cost = 1.0 * err
            observed_cost = 0.6 * err
        regrets.append(observed_cost - counterfactual_cost)
    arr = np.asarray(regrets, dtype=float)
    return {
        "n": int(arr.size),
        "mean_regret": float(arr.mean()) if arr.size else 0.0,
        "p95_regret": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "total_regret": float(arr.sum()),
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute routing regret")
    parser.add_argument("--input", required=True, help="Path to telemetry JSONL")
    args = parser.parse_args()
    rows = _read_jsonl(Path(args.input))
    result = compute_regret(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
