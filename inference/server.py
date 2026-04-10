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

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy globals — loaded once at startup via lifespan
# ---------------------------------------------------------------------------
_router = None  # InferenceRouter  (router mode)
_serve_target: str = os.getenv("SERVE_TARGET", "router").lower()
_telemetry_writer = None
_batch_queue: asyncio.Queue = asyncio.Queue()
_batch_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _telemetry_writer, _batch_task
    import sys

    sys.path.insert(0, str(__file__ + "/../.."))  # make parent package importable
    from inference.config import load_runtime_config
    from telemetry.writer import from_env

    _telemetry_writer = from_env()
    _telemetry_writer.start()
    if _serve_target == "router":
        from router import (
            AttentionEntropyProbe,
            CalibrationArtifactManager,
            InferenceRouter,
            TemporalDriftController,
            ThresholdCalibrator,
        )
        runtime = load_runtime_config()
        tau_override = runtime.tau
        probe_model = os.getenv("PROBE_MODEL", "distilbert-base-uncased")
        probe = AttentionEntropyProbe(model_name=probe_model)
        calibrator = ThresholdCalibrator()

        # Prefer runtime config (env/file/firestore), fallback to latest artifact.
        if tau_override:
            calibrator.update_tau(float(tau_override))
        else:
            artifact = CalibrationArtifactManager().load_latest()
            if artifact is not None and artifact.is_valid():
                calibrator.update_tau(artifact.tau)

        temporal = TemporalDriftController(
            window_size=int(os.getenv("TEMPORAL_WINDOW_SIZE", "200")),
            z_threshold=float(os.getenv("TEMPORAL_Z_THRESHOLD", "2.0")),
            tau_offset=float(os.getenv("TEMPORAL_TAU_OFFSET", "-0.2")),
        )
        _router = InferenceRouter(probe=probe, calibrator=calibrator, temporal_controller=temporal)
        _batch_task = asyncio.create_task(_batch_worker(), name="router-batch-worker")
        logger.info("Router mode initialised", tau=calibrator.tau, config_source=runtime.source)
    else:
        logger.info("Single-model mode", target=_serve_target)
    yield
    if _batch_task is not None:
        _batch_task.cancel()
        try:
            await _batch_task
        except asyncio.CancelledError:
            pass
    if _telemetry_writer is not None:
        await _telemetry_writer.stop()
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
async def infer(req: InferRequest, request: Request, response: Response) -> InferResponse:
    """
    Full inference pipeline:
    1. Compute attention entropy via the probe model.
    2. Route: GCP fast path (H < tau) or AWS heavy path (H >= tau).
    3. Run the appropriate model and return results.
    """
    t0 = time.perf_counter()
    from telemetry.tracing import new_trace, trace_headers
    trace_ctx = new_trace(trace_id=request.headers.get("X-Trace-ID"))
    response.headers.update(trace_headers(trace_ctx))

    if _serve_target == "router" and _router is not None:
        result = await _handle_router_mode(req, t0, trace_ctx.trace_id)
        return result

    # Single-model mode (deployed on the actual cloud endpoint)
    result = await _handle_single_model_mode(req, t0)
    return result


async def _handle_router_mode(req: InferRequest, t0: float, trace_id: str) -> InferResponse:
    from inference.models import run_absa, run_hallucination_scorer

    decision = await _compute_entropy_routed_decision(req, trace_id)

    if decision.destination.value == "gcp_cloud_run":
        model_result = run_absa(req.text)
    else:
        model_result = run_hallucination_scorer(req.text, req.hypothesis)

    total_latency = (time.perf_counter() - t0) * 1000.0
    decision.total_latency_ms = total_latency
    if _telemetry_writer is not None:
        await _telemetry_writer.write(decision.to_dict())

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


async def _compute_entropy_routed_decision(req: InferRequest, trace_id: str):
    if _batch_task is None:
        return _router.route(
            text=req.text,
            request_id=req.request_id,
            metadata={"has_hypothesis": req.hypothesis is not None, "trace_id": trace_id},
        )
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await _batch_queue.put((req, trace_id, future))
    return await future


async def _batch_worker() -> None:
    """
    Collect incoming requests and compute entropy in true batches.
    """
    while True:
        req, trace_id, fut = await _batch_queue.get()
        batch = [(req, trace_id, fut)]
        deadline = asyncio.get_running_loop().time() + float(os.getenv("BATCH_WINDOW_MS", "8")) / 1000.0
        max_batch = int(os.getenv("MAX_BATCH_SIZE", "16"))
        while len(batch) < max_batch:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(_batch_queue.get(), timeout=timeout))
            except asyncio.TimeoutError:
                break
        texts = [item[0].text for item in batch]
        batch_t0 = time.perf_counter()
        results = await asyncio.to_thread(_router.probe.compute_batch, texts)
        per_item_probe_latency = ((time.perf_counter() - batch_t0) * 1000.0) / max(1, len(results))
        for (req_i, trace_id_i, fut_i), entropy_result in zip(batch, results):
            try:
                h = entropy_result.h_route
                tau = _router.tau
                if getattr(_router, "_temporal", None) is not None:
                    _router._temporal.update(h)
                    tau = _router._temporal.adjusted_tau(tau)
                error_prob = _router._calibrator.predict_error_prob(h)
                from router.router import RoutingDecision, RoutingDestination
                decision = RoutingDecision(
                    request_id=req_i.request_id or str(uuid.uuid4()),
                    destination=RoutingDestination.AWS_SAGEMAKER if h >= tau else RoutingDestination.GCP_CLOUD_RUN,
                    h_route=h,
                    tau=tau,
                    error_probability=error_prob,
                    input_tokens=entropy_result.input_tokens,
                    probe_latency_ms=per_item_probe_latency,
                    total_latency_ms=None,
                    timestamp_utc=time.time(),
                    model_name=entropy_result.model_name,
                    metadata={"has_hypothesis": req_i.hypothesis is not None, "trace_id": trace_id_i},
                )
                fut_i.set_result(decision)
            except Exception as exc:  # pragma: no cover - defensive
                fut_i.set_exception(exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
