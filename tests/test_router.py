"""
Unit tests for router/router.py.

The InferenceRouter is tested with a mock AttentionEntropyProbe to avoid
loading transformer weights during the test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from router.types import EntropyResult
from router.router import InferenceRouter, RoutingDecision, RoutingDestination
from router.threshold import ThresholdCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_probe(h_route: float) -> MagicMock:
    """Return a mock AttentionEntropyProbe that always returns h_route."""
    probe = MagicMock()
    probe.compute.return_value = EntropyResult(
        h_route=h_route,
        per_head_entropies=np.array([[h_route]]),
        input_tokens=12,
        model_name="mock-probe",
    )
    return probe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInferenceRouter:
    @pytest.fixture
    def calibrator(self):
        cal = ThresholdCalibrator(risk_tolerance=0.15)
        cal.update_tau(2.0)
        return cal

    def test_low_entropy_routes_to_gcp(self, calibrator):
        probe = make_mock_probe(h_route=1.0)  # below tau=2.0
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Some clear input")
        assert decision.destination == RoutingDestination.GCP_CLOUD_RUN

    def test_high_entropy_routes_to_aws(self, calibrator):
        probe = make_mock_probe(h_route=3.5)  # above tau=2.0
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Ambiguous input")
        assert decision.destination == RoutingDestination.AWS_SAGEMAKER

    def test_exactly_at_tau_routes_to_aws(self, calibrator):
        """H >= tau should go to AWS (boundary condition)."""
        probe = make_mock_probe(h_route=2.0)  # exactly equal to tau
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Boundary input")
        assert decision.destination == RoutingDestination.AWS_SAGEMAKER

    def test_decision_captures_h_route(self, calibrator):
        probe = make_mock_probe(h_route=1.7)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test input")
        assert decision.h_route == pytest.approx(1.7)

    def test_decision_captures_tau(self, calibrator):
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test input")
        assert decision.tau == pytest.approx(2.0)

    def test_auto_generated_request_id(self, calibrator):
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        d1 = router.route("First input")
        d2 = router.route("Second input")
        assert d1.request_id != d2.request_id

    def test_caller_provided_request_id(self, calibrator):
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test", request_id="my-id-123")
        assert decision.request_id == "my-id-123"

    def test_metadata_attached(self, calibrator):
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test", metadata={"source": "api"})
        assert decision.metadata["source"] == "api"

    def test_probe_latency_ms_positive(self, calibrator):
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test input")
        assert decision.probe_latency_ms >= 0.0
        assert decision.total_latency_ms is None

    def test_update_tau_changes_routing(self, calibrator):
        probe = make_mock_probe(h_route=1.5)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        # Initially h=1.5 < tau=2.0 -> GCP
        assert router.route("Input").destination == RoutingDestination.GCP_CLOUD_RUN
        # Lower tau to 1.0 -> h=1.5 >= tau=1.0 -> AWS
        router.update_tau(1.0)
        assert router.route("Input").destination == RoutingDestination.AWS_SAGEMAKER

    def test_to_dict_serialisable(self, calibrator):
        import json
        probe = make_mock_probe(h_route=1.0)
        router = InferenceRouter(probe=probe, calibrator=calibrator)
        decision = router.route("Test")
        d = decision.to_dict()
        assert "probe_latency_ms" in d
        assert "total_latency_ms" in d
        assert d["total_latency_ms"] is None
        # Should be JSON-serialisable for BigQuery streaming
        serialised = json.dumps(d)
        assert "gcp_cloud_run" in serialised

    def test_is_escalated_property(self, calibrator):
        probe_low = make_mock_probe(h_route=0.5)
        probe_high = make_mock_probe(h_route=3.0)
        router_low = InferenceRouter(probe=probe_low, calibrator=calibrator)
        router_high = InferenceRouter(probe=probe_high, calibrator=calibrator)
        assert not router_low.route("Clear").is_escalated
        assert router_high.route("Unclear").is_escalated
