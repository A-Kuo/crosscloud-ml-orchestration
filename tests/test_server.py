"""
Tests for the FastAPI inference server.

Uses httpx.AsyncClient with ASGITransport. Router mode is tested by injecting
a pre-built InferenceRouter with a mock probe so the lifespan never loads
transformer weights. Single-model modes (absa, hallucination) are tested with
mocked model runners.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

# Lifespan skips AttentionEntropyProbe when not "router"; tests patch router mode per case.
os.environ.setdefault("SERVE_TARGET", "absa")

from inference.server import app
from router.router import InferenceRouter
from router.types import EntropyResult
from router.threshold import ThresholdCalibrator

import numpy as np


def make_mock_probe(h_route: float) -> MagicMock:
    """Return a mock probe that always returns the given h_route."""
    probe = MagicMock()

    def _one_result():
        return EntropyResult(
            h_route=h_route,
            per_head_entropies=np.array([[h_route]]),
            input_tokens=12,
            model_name="mock-probe",
        )

    probe.compute.return_value = _one_result()
    probe.compute_batch.side_effect = lambda texts: [_one_result() for _ in texts]
    return probe


def make_router(h_route: float, tau: float = 2.0) -> InferenceRouter:
    """Build an InferenceRouter with a mock probe (no model load)."""
    calibrator = ThresholdCalibrator()
    calibrator.update_tau(tau)
    return InferenceRouter(probe=make_mock_probe(h_route), calibrator=calibrator)


@pytest.fixture
def client():
    """ASGI client for the inference app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _router_mode_patch(h_route: float, tau: float = 2.0):
    """Context manager that sets server to router mode with an injected router."""
    router = make_router(h_route, tau=tau)
    return patch("inference.server._serve_target", "router"), patch(
        "inference.server._router", router
    )


class TestHealth:
    async def test_health_returns_200_and_ok(self, client):
        p1, p2 = _router_mode_patch(1.5)
        with p1, p2:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["serve_target"] == "router"
        assert "tau" in data

    async def test_health_absa_mode(self, client):
        with patch("inference.server._serve_target", "absa"), patch(
            "inference.server._router", None
        ):
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["serve_target"] == "absa"
        assert resp.json()["tau"] is None

    async def test_health_hallucination_mode(self, client):
        with patch("inference.server._serve_target", "hallucination"), patch(
            "inference.server._router", None
        ):
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["serve_target"] == "hallucination"


class TestEntropy:
    async def test_entropy_returns_h_route_in_router_mode(self, client):
        p1, p2 = _router_mode_patch(2.3)
        with p1, p2:
            resp = await client.post(
                "/entropy",
                json={"text": "Some input text"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["h_route"] == pytest.approx(2.3)
        assert data["input_tokens"] == 12
        assert data["model_name"] == "mock-probe"

    async def test_entropy_unavailable_in_absa_mode(self, client):
        with patch("inference.server._serve_target", "absa"), patch(
            "inference.server._router", None
        ):
            resp = await client.post(
                "/entropy",
                json={"text": "Some input"},
            )
        assert resp.status_code == 503
        assert "router mode" in resp.json()["detail"].lower()


class TestInfer:
    async def test_infer_router_mode_low_entropy_routes_to_gcp(self, client):
        p1, p2 = _router_mode_patch(1.0)
        with p1, p2, patch(
            "inference.models.run_absa",
            return_value={"label": "positive", "score": 0.9},
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.1},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Clear input"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["destination"] == "gcp_cloud_run"
        assert data["h_route"] == pytest.approx(1.0)
        assert data["result"] == {"label": "positive", "score": 0.9}
        assert data["latency_ms"] >= 0
        assert "request_id" in data

    async def test_infer_router_mode_high_entropy_routes_to_aws(self, client):
        p1, p2 = _router_mode_patch(3.0)
        with p1, p2, patch(
            "inference.models.run_absa",
            return_value={"label": "neutral", "score": 0.5},
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.8},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Ambiguous input"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["destination"] == "aws_sagemaker"
        assert data["h_route"] == pytest.approx(3.0)
        assert data["result"] == {"score": 0.8}
        assert data["latency_ms"] >= 0

    async def test_infer_absa_mode(self, client):
        with patch("inference.server._serve_target", "absa"), patch(
            "inference.server._router", None
        ), patch(
            "inference.models.run_absa",
            return_value={"label": "positive", "score": 0.95},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Product is great"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["destination"] == "absa"
        assert data["h_route"] == -1.0
        assert data["tau"] == -1.0
        assert data["result"] == {"label": "positive", "score": 0.95}
        assert data["latency_ms"] >= 0

    async def test_infer_hallucination_mode(self, client):
        with patch("inference.server._serve_target", "hallucination"), patch(
            "inference.server._router", None
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.7},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Claim.", "hypothesis": "Optional hypothesis"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["destination"] == "hallucination"
        assert data["h_route"] == -1.0
        assert data["result"] == {"score": 0.7}
        assert data["latency_ms"] >= 0

    async def test_infer_preserves_request_id(self, client):
        p1, p2 = _router_mode_patch(1.0)
        with p1, p2, patch(
            "inference.models.run_absa",
            return_value={"label": "positive", "score": 0.9},
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.1},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Input", "request_id": "my-id-123"},
            )
        assert resp.status_code == 200
        assert resp.json()["request_id"] == "my-id-123"

    async def test_infer_rejects_empty_text(self, client):
        p1, p2 = _router_mode_patch(1.0)
        with p1, p2:
            resp = await client.post(
                "/infer",
                json={"text": ""},
            )
        assert resp.status_code == 422

    async def test_infer_returns_trace_headers(self, client):
        p1, p2 = _router_mode_patch(1.0)
        with p1, p2, patch(
            "inference.models.run_absa",
            return_value={"label": "positive", "score": 0.9},
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.1},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Hello"},
            )
        assert resp.status_code == 200
        assert "X-Trace-ID" in resp.headers
        assert "X-Span-ID" in resp.headers

    async def test_infer_preserves_incoming_trace_id(self, client):
        p1, p2 = _router_mode_patch(1.0)
        with p1, p2, patch(
            "inference.models.run_absa",
            return_value={"label": "positive", "score": 0.9},
        ), patch(
            "inference.models.run_hallucination_scorer",
            return_value={"score": 0.1},
        ):
            resp = await client.post(
                "/infer",
                json={"text": "Hello"},
                headers={"X-Trace-ID": "client-trace-abc"},
            )
        assert resp.status_code == 200
        assert resp.headers["X-Trace-ID"] == "client-trace-abc"
