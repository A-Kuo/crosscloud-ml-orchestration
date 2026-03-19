"""
Unit tests for router/threshold.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from router.threshold import ThresholdCalibrator, CalibrationResult, _safe_auroc


class TestThresholdCalibrator:
    @pytest.fixture
    def calibrator(self):
        return ThresholdCalibrator(risk_tolerance=0.15)

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(seed=0)
        n = 500
        entropies = rng.normal(loc=2.0, scale=0.5, size=n)
        # Error probability increases with entropy (sigmoid centred at 2.0)
        prob_error = 1 / (1 + np.exp(-(entropies - 2.0) * 3))
        labels = rng.binomial(1, prob_error).astype(float)
        return entropies, labels

    def test_fit_returns_calibration_result(self, calibrator, synthetic_data):
        entropies, labels = synthetic_data
        result = calibrator.fit(entropies, labels)
        assert isinstance(result, CalibrationResult)

    def test_tau_is_positive(self, calibrator, synthetic_data):
        entropies, labels = synthetic_data
        result = calibrator.fit(entropies, labels)
        assert result.tau > 0.0

    def test_tau_within_observed_range(self, calibrator, synthetic_data):
        entropies, labels = synthetic_data
        result = calibrator.fit(entropies, labels)
        assert entropies.min() <= result.tau <= entropies.max()

    def test_auroc_above_chance(self, calibrator, synthetic_data):
        """Isotonic regression on correlated data should outperform random (>0.55)."""
        entropies, labels = synthetic_data
        result = calibrator.fit(entropies, labels)
        assert result.auroc > 0.55, f"Expected AUROC > 0.55, got {result.auroc:.4f}"

    def test_predict_error_prob_monotone(self, calibrator, synthetic_data):
        """Isotonic regression guarantees monotone predictions."""
        entropies, labels = synthetic_data
        calibrator.fit(entropies, labels)
        test_entropies = np.linspace(entropies.min(), entropies.max(), 100)
        probs = [calibrator.predict_error_prob(e) for e in test_entropies]
        # Each prob should be >= the previous (monotone non-decreasing)
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1] - 1e-9, (
                f"Monotonicity violated at index {i}: {probs[i-1]:.4f} -> {probs[i]:.4f}"
            )

    def test_unfitted_predict_returns_one(self, calibrator):
        prob = calibrator.predict_error_prob(3.0)
        assert prob == 1.0

    def test_update_tau(self, calibrator):
        calibrator.update_tau(1.23)
        assert calibrator.tau == pytest.approx(1.23)

    def test_tau_history_accumulates(self, calibrator, synthetic_data):
        entropies, labels = synthetic_data
        r1 = calibrator.fit(entropies, labels, previous_tau_history=[1.5, 1.8])
        assert len(r1.tau_history) == 3
        assert r1.tau_history[:2] == [1.5, 1.8]
        assert r1.tau_history[2] == pytest.approx(r1.tau)

    def test_save_and_load_round_trip(self, calibrator, synthetic_data):
        entropies, labels = synthetic_data
        result = calibrator.fit(entropies, labels)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            result.save(path)
            loaded = CalibrationResult.load(path)
        assert loaded.tau == pytest.approx(result.tau)
        assert loaded.auroc == pytest.approx(result.auroc)
        assert loaded.n_samples == result.n_samples

    def test_small_sample_warning(self, calibrator, caplog):
        """< 10 samples should trigger a warning."""
        import logging
        entropies = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        with caplog.at_level(logging.WARNING, logger="router.threshold"):
            calibrator.fit(entropies, labels)
        assert any("unreliable" in r.message.lower() for r in caplog.records)

    def test_mismatched_lengths_raises(self, calibrator):
        with pytest.raises(ValueError, match="length"):
            calibrator.fit(np.array([1.0, 2.0]), np.array([0.0]))


class TestSafeAuroc:
    def test_single_class_returns_half(self):
        labels = np.zeros(10)
        scores = np.random.rand(10)
        assert _safe_auroc(labels, scores) == pytest.approx(0.5)

    def test_perfect_classifier(self):
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _safe_auroc(labels, scores) == pytest.approx(1.0)
