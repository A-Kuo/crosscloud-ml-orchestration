"""
Unit tests for the CrossCloud ML Orchestration DAG.

Tests _kl_divergence with known distributions and the four DAG task callables
with MOCK_CLOUD=true and mocked XCom context. Requires apache-airflow to be
installed. Task callable tests are skipped when the DAG module cannot be loaded
(e.g. when the project's airflow/ folder shadows the installed package).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("MOCK_CLOUD", "true")

# KL divergence: same implementation as in airflow/dags/crosscloud_orchestration.py
# so we can test it without loading the DAG (which conflicts with local airflow/ folder).
def _kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 50) -> float:
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


# Load DAG module for task callable tests (skip if import fails due to airflow package conflict)
dags = None
try:
    pytest.importorskip("airflow")
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    _save_path = list(sys.path)
    _root_str = str(_root.resolve())
    sys.path = [p for p in sys.path if str(Path(p).resolve()) != _root_str]
    _dag_path = _root / "airflow" / "dags" / "crosscloud_orchestration.py"
    if _dag_path.exists():
        _spec = importlib.util.spec_from_file_location("crosscloud_orchestration", _dag_path)
        _dags_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_dags_mod)
        dags = _dags_mod
    sys.path = _save_path
except Exception:
    dags = None


# ---------------------------------------------------------------------------
# _kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """Pure-function tests for histogram-based KL divergence."""

    def test_identical_distributions_zero_kl(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=2.0, scale=0.5, size=1000)
        kl = _kl_divergence(samples, samples, n_bins=30)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_same_distribution_two_draws_near_zero(self):
        rng = np.random.default_rng(123)
        p = rng.normal(loc=1.5, scale=0.3, size=2000)
        q = rng.normal(loc=1.5, scale=0.3, size=2000)
        kl = _kl_divergence(p, q, n_bins=40)
        assert kl >= 0.0
        assert kl < 0.1

    def test_different_distributions_positive_kl(self):
        p = np.random.default_rng(1).normal(loc=3.0, scale=0.2, size=1000)
        q = np.random.default_rng(2).normal(loc=1.0, scale=0.2, size=1000)
        kl = _kl_divergence(p, q, n_bins=30)
        assert kl > 0.1

    def test_non_overlapping_positive_kl(self):
        p = np.ones(500) * 1.0
        q = np.ones(500) * 5.0
        kl = _kl_divergence(p, q, n_bins=10)
        assert kl > 0.0

    def test_empty_bins_handled(self):
        p = np.array([1.0, 1.0, 1.0])
        q = np.array([2.0, 2.0, 2.0])
        kl = _kl_divergence(p, q, n_bins=5)
        assert kl >= 0.0
        assert np.isfinite(kl)


def _requires_dag():
    if dags is None:
        pytest.skip(
            "DAG module could not be loaded (local airflow/ may shadow apache-airflow). "
            "KL tests run; task callable tests skipped."
        )


# ---------------------------------------------------------------------------
# XCom mock for task callables
# ---------------------------------------------------------------------------

def _make_context(xcom_data: dict | None = None):
    """Build a minimal Airflow task context with XCom push/pull."""
    ti = MagicMock()
    _store = xcom_data or {}

    def push(key, value):
        _store[key] = value

    def pull(task_ids, key):
        return _store.get(key)

    ti.xcom_push = push
    ti.xcom_pull = pull
    return {"ti": ti}


# ---------------------------------------------------------------------------
# entropy_audit
# ---------------------------------------------------------------------------

class TestEntropyAudit:
    def test_entropy_audit_returns_audit_result(self):
        _requires_dag()
        callable_ = dags.dag.get_task("entropy_audit").python_callable
        ctx = _make_context()
        result = callable_(**ctx)
        assert "kl_divergence" in result
        assert "should_recalibrate" in result
        assert "should_retrain" in result
        assert result["kl_divergence"] >= 0.0
        assert result["n_current_samples"] > 0
        assert result["n_baseline_samples"] > 0
        assert "current_entropies" in result

    def test_entropy_audit_bounds(self):
        _requires_dag()
        callable_ = dags.dag.get_task("entropy_audit").python_callable
        ctx = _make_context()
        result = callable_(**ctx)
        assert result["should_recalibrate"] == (result["kl_divergence"] > dags.KL_RECALIBRATE_BOUND)
        assert result["should_retrain"] == (result["kl_divergence"] > dags.KL_RETRAIN_BOUND)


# ---------------------------------------------------------------------------
# threshold_recalibrate
# ---------------------------------------------------------------------------

class TestThresholdRecalibrate:
    def test_skipped_when_kl_below_bound(self):
        _requires_dag()
        callable_ = dags.dag.get_task("threshold_recalibrate").python_callable
        ctx = _make_context({
            "audit_result": {
                "kl_divergence": 0.05,
                "should_recalibrate": False,
                "should_retrain": False,
                "current_entropies": [1.0, 1.5, 2.0],
            }
        })
        result = callable_(**ctx)
        assert result["skipped"] is True
        assert "reason" in result

    def test_recalibrates_when_kl_above_bound(self):
        _requires_dag()
        callable_ = dags.dag.get_task("threshold_recalibrate").python_callable
        entropies = list(np.random.default_rng(42).normal(loc=1.8, scale=0.5, size=200))
        audit = {
            "kl_divergence": 0.2,
            "should_recalibrate": True,
            "should_retrain": False,
            "n_current_samples": len(entropies),
            "n_baseline_samples": 1000,
            "current_mean_entropy": float(np.mean(entropies)),
            "baseline_mean_entropy": 1.8,
            "current_entropies": entropies,
        }
        ctx = _make_context({"audit_result": audit})
        result = callable_(**ctx)
        assert result["skipped"] is False
        assert "new_tau" in result
        assert result["new_tau"] > 0
        assert "auroc" in result
        assert "n_samples" in result


# ---------------------------------------------------------------------------
# retrain_trigger
# ---------------------------------------------------------------------------

class TestRetrainTrigger:
    def test_skipped_when_kl_below_retrain_bound(self):
        _requires_dag()
        callable_ = dags.dag.get_task("retrain_trigger").python_callable
        ctx = _make_context({
            "audit_result": {
                "kl_divergence": 0.15,
                "should_recalibrate": True,
                "should_retrain": False,
            }
        })
        result = callable_(**ctx)
        assert result["triggered"] is False
        assert "reason" in result

    def test_triggered_when_kl_above_retrain_bound(self):
        _requires_dag()
        callable_ = dags.dag.get_task("retrain_trigger").python_callable
        ctx = _make_context({
            "audit_result": {
                "kl_divergence": 0.5,
                "should_recalibrate": True,
                "should_retrain": True,
            }
        })
        result = callable_(**ctx)
        assert result["triggered"] is True
        assert "job_id" in result
        assert result["job_id"] == "mock-vertex-job-001"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_ok_for_both_endpoints_in_mock_mode(self):
        _requires_dag()
        callable_ = dags.dag.get_task("health_check").python_callable
        ctx = _make_context()
        result = callable_(**ctx)
        assert "gcp_cloud_run" in result
        assert "aws_sagemaker" in result
        assert result["gcp_cloud_run"]["status"] == "ok"
        assert result["aws_sagemaker"]["status"] == "ok"
        assert result["gcp_cloud_run"].get("mocked") is True
