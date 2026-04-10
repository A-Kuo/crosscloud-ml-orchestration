"""
Cross-cloud configuration loading utilities.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    tau: Optional[float]
    source: str


def load_runtime_config() -> RuntimeConfig:
    """
    Load routing config using a simple precedence:
      1) explicit TAU env var
      2) local artifact path (for parity between cloud targets)
      3) Firestore (optional)
      4) None (use calibrator default)
    """
    if os.getenv("TAU"):
        return RuntimeConfig(tau=float(os.getenv("TAU")), source="env")

    artifact_path = os.getenv("TAU_ARTIFACT_PATH", "/tmp/current_tau.json")
    try:
        with open(artifact_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "tau" in payload:
            return RuntimeConfig(tau=float(payload["tau"]), source=f"file:{artifact_path}")
    except FileNotFoundError:
        pass

    firestore_doc = _load_tau_from_firestore()
    if firestore_doc is not None:
        return RuntimeConfig(tau=firestore_doc, source="firestore")

    return RuntimeConfig(tau=None, source="default")


def _load_tau_from_firestore() -> Optional[float]:
    if os.getenv("USE_FIRESTORE_CONFIG", "false").lower() != "true":
        return None
    try:
        from google.cloud import firestore
    except Exception:
        return None
    project = os.getenv("BQ_PROJECT", "crosscloud-demo")
    collection = os.getenv("CONFIG_COLLECTION", "crosscloud_config")
    document = os.getenv("CONFIG_DOCUMENT", "routing")
    client = firestore.Client(project=project)
    snap = client.collection(collection).document(document).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    tau = data.get("tau")
    return float(tau) if tau is not None else None
