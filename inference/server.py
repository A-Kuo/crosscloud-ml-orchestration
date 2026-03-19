"""
FastAPI inference server.

Exposes two endpoints:
  POST /infer           - full pipeline: compute entropy, route, run model
  POST /entropy         - entropy-only (debugging / telemetry)
  GET  /health          - liveness probe used by Cloud Run and SageMaker

Environment variables
---------------------
SERVE_TARGET   : "absa" | "hallucination" | "router"
                  "router" mode runs the full cross-cloud routing logic locally
                  (used for local development and integration testing).
                  In production, Cloud Run sets SERVE_TARGET=absa and
                  SageMaker sets SERVE_TARGET=hallucination.
TAU            : float  - override routing threshold (router mode only)
PROBE_MODEL    : HuggingFace model ID for the entropy probe (router mode)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import structlog
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy globals — loaded once at startup via lifespan
# ---------------------------------------------------------------------------
_router = None  # InferenceRouter  (router mode)
_serve_target: str = os.getenv("SERVE_TARGET", "router").lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router
    import sys
    sys.path.insert(0, str(__file__ + "/../.."))  # make parent package importable
    from router import InferenceRouter, AttentionEntropyProbe, ThresholdCalibrator

    if _serve_target == "router":
        tau_override = os.getenv("TAU")
        probe_model = os.getenv("PROBE_MODEL", "distilbert-base-uncased")
        probe = AttentionEntropyProbe(model_name=probe_model)
        calibrator = ThresholdCalibrator()
        if tau_override:
            calibrator.update_tau(float(tau_override))
        _router = InferenceRouter(probe=probe, calibrator=calibrator)
        logger.info("Router mode initialised", tau=calibrator.tau)
    else:
        logger.info("Single-model mode", target=_serve_target)
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="CrossCloud ML Inference Server",
    description=(
        "Entropy-based cross-cloud inference router. "
        "Routes requests to GCP Cloud Run (low entropy) or AWS SageMaker (high entropy)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Input text to classify")
    hypothesis: Optional[str] = Field(None, description="Optional hypothesis for hallucination scorer")
    request_id: Optional[str] = Field(None, description="Caller-provided request ID")


class EntropyResponse(BaseModel):
    h_route: float
    input_tokens: int
    model_name: str


class InferResponse(BaseModel):
    request_id: str
    destination: str
    h_route: float
    tau: float
    error_probability: float
    result: dict          # model output
    latency_ms: float
    timestamp_utc: float


class HealthResponse(BaseModel):
    status: str
    serve_target: str
    tau: Optional[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        serve_target=_serve_target,
        tau=_router.tau if _router else None,
    )


@app.post("/entropy", response_model=EntropyResponse, tags=["debug"])
async def compute_entropy(req: InferRequest) -> EntropyResponse:
    """Return the entropy score without running downstream inference."""
    if _router is None:
        raise HTTPException(
            status_code=503,
            detail="Server is not in router mode; /entropy is unavailable",
        )
    result = _router.probe.compute(req.text)
    return EntropyResponse(
        h_route=result.h_route,
        input_tokens=result.input_tokens,
        model_name=result.model_name,
    )


@app.post("/infer", response_model=InferResponse, tags=["inference"])
async def infer(req: InferRequest) -> InferResponse:
    """
    Full inference pipeline:
    1. Compute attention entropy via the probe model.
    2. Route: GCP fast path (H < tau) or AWS heavy path (H >= tau).
    3. Run the appropriate model and return results.
    """
    t0 = time.perf_counter()

    if _serve_target == "router" and _router is not None:
        return await _handle_router_mode(req, t0)

    # Single-model mode (deployed on the actual cloud endpoint)
    return await _handle_single_model_mode(req, t0)


async def _handle_router_mode(req: InferRequest, t0: float) -> InferResponse:
    from inference.models import run_absa, run_hallucination_scorer

    decision = _router.route(
        text=req.text,
        request_id=req.request_id,
        metadata={"has_hypothesis": req.hypothesis is not None},
    )

    if decision.destination.value == "gcp_cloud_run":
        model_result = run_absa(req.text)
    else:
        model_result = run_hallucination_scorer(req.text, req.hypothesis)

    total_latency = (time.perf_counter() - t0) * 1000.0

    return InferResponse(
        request_id=decision.request_id,
        destination=decision.destination.value,
        h_route=decision.h_route,
        tau=decision.tau,
        error_probability=decision.error_probability,
        result=model_result,
        latency_ms=total_latency,
        timestamp_utc=decision.timestamp_utc,
    )


async def _handle_single_model_mode(req: InferRequest, t0: float) -> InferResponse:
    from inference.models import run_absa, run_hallucination_scorer

    request_id = req.request_id or str(uuid.uuid4())
    if _serve_target == "absa":
        model_result = run_absa(req.text)
    elif _serve_target == "hallucination":
        model_result = run_hallucination_scorer(req.text, req.hypothesis)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown SERVE_TARGET: {_serve_target}")

    total_latency = (time.perf_counter() - t0) * 1000.0
    return InferResponse(
        request_id=request_id,
        destination=_serve_target,
        h_route=-1.0,   # not computed in single-model mode
        tau=-1.0,
        error_probability=-1.0,
        result=model_result,
        latency_ms=total_latency,
        timestamp_utc=time.time(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
