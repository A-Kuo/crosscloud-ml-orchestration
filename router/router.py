"""
Inference router.

Combines AttentionEntropyProbe and ThresholdCalibrator to make routing
decisions for incoming inference requests.

Routing logic:
    H_route < tau  ->  GCP Cloud Run  (fast path, low-uncertainty)
    H_route >= tau ->  AWS SageMaker  (heavy model, high-uncertainty)

Every decision is logged as a RoutingDecision dataclass so callers can emit
telemetry to BigQuery without depending on this module's internals.

Cost model (2025 us-east-1 / us-central1 prices):
    GCP Cloud Run: $0.0000024/request + $0.0000000024/ms
    AWS SageMaker ml.g5.2xlarge: $1.515/hr → billed per-ms of actual request duration
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .temporal import TemporalDriftController
from .threshold import ThresholdCalibrator
from .types import EntropyResult

if TYPE_CHECKING:
    from .entropy import AttentionEntropyProbe

# BigQuery schema uses probe_latency_ms and total_latency_ms; see telemetry/bigquery_schema.json

logger = logging.getLogger(__name__)


class RoutingDestination(str, Enum):
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AWS_SAGEMAKER = "aws_sagemaker"


# ---------------------------------------------------------------------------
# Per-request cost estimation
# ---------------------------------------------------------------------------

# GCP Cloud Run pricing (us-central1): per-request + per-ms fee
_GCP_COST_PER_REQUEST_USD: float = 0.0000024
_GCP_COST_PER_MS_USD: float = 0.0000000024

# AWS SageMaker ml.g5.2xlarge on-demand (us-east-1): $1.515/hr
_SAGEMAKER_INSTANCE_COST_PER_MS_USD: float = 1.515 / 3_600_000


def estimate_cost(
    destination: "RoutingDestination",
    total_latency_ms: Optional[float],
) -> tuple[float, str]:
    """
    Estimate the USD cost for a single inference request.

    For GCP Cloud Run we use a per-request flat fee plus a per-ms fee.
    For SageMaker we prorate the hourly instance cost by actual request duration.
    If total_latency_ms is unknown we fall back to destination-specific defaults.

    Returns (cost_usd, cost_model_label).
    """
    if destination == RoutingDestination.GCP_CLOUD_RUN:
        duration_ms = total_latency_ms if total_latency_ms is not None else 150.0
        cost = _GCP_COST_PER_REQUEST_USD + duration_ms * _GCP_COST_PER_MS_USD
        return round(cost, 10), "gcp_cloud_run_per_request"
    else:
        duration_ms = total_latency_ms if total_latency_ms is not None else 500.0
        cost = duration_ms * _SAGEMAKER_INSTANCE_COST_PER_MS_USD
        return round(cost, 10), "aws_sagemaker_per_hour"


@dataclass
class RoutingDecision:
    """
    Immutable record of a single routing decision.

    Emitted by InferenceRouter.route() and intended to be forwarded to
    BigQuery for audit, drift detection, and post-hoc analysis.

    Fields
    ------
    cost_per_request_usd
        Estimated USD cost for this request (GCP per-request model or SageMaker
        prorated hourly rate). Populated immediately using probe_latency_ms as a
        lower-bound; callers should update via set_total_latency() once the
        downstream model call completes.
    cost_model
        String label identifying which pricing model was used.
    """
    request_id: str
    destination: RoutingDestination
    h_route: float
    tau: float
    error_probability: float           # P(error | H_route) from calibrator
    input_tokens: int
    probe_latency_ms: float            # time to compute entropy + decide
    total_latency_ms: Optional[float]   # end-to-end including model call; set by caller if known
    timestamp_utc: float               # Unix timestamp
    model_name: str
    cost_per_request_usd: float = 0.0
    cost_model: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_escalated(self) -> bool:
        return self.destination == RoutingDestination.AWS_SAGEMAKER

    def set_total_latency(self, total_latency_ms: float) -> None:
        """Update end-to-end latency and recompute cost once the model call finishes."""
        self.total_latency_ms = total_latency_ms
        self.cost_per_request_usd, self.cost_model = estimate_cost(
            self.destination, total_latency_ms
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["destination"] = self.destination.value
        return d


class InferenceRouter:
    """
    Primary router class.

    Parameters
    ----------
    probe:
        AttentionEntropyProbe instance. If None, a default DistilBERT probe
        is instantiated on first use (lazy init).
    calibrator:
        ThresholdCalibrator instance. If None, a default calibrator with
        tau=2.0 (uncalibrated fallback) is used.
    """

    def __init__(
        self,
        probe: Optional["AttentionEntropyProbe"] = None,
        calibrator: Optional[ThresholdCalibrator] = None,
        temporal_controller: Optional[TemporalDriftController] = None,
    ) -> None:
        self._probe = probe
        self._calibrator = calibrator if calibrator is not None else ThresholdCalibrator()
        self._temporal = temporal_controller

    @property
    def probe(self) -> "AttentionEntropyProbe":
        if self._probe is None:
            from .entropy import AttentionEntropyProbe

            logger.info("Lazy-initialising default AttentionEntropyProbe (DistilBERT)")
            self._probe = AttentionEntropyProbe()
        return self._probe

    @property
    def tau(self) -> float:
        return self._calibrator.tau

    def route(
        self,
        text: str,
        request_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Compute attention entropy for `text` and return a RoutingDecision.

        Parameters
        ----------
        text:
            Raw input text to route.
        request_id:
            Optional caller-provided ID. Auto-generated UUID4 if not given.
        metadata:
            Arbitrary dict attached to the decision record for downstream use.
        """
        t0 = time.perf_counter()
        result: EntropyResult = self.probe.compute(text)
        probe_latency_ms = (time.perf_counter() - t0) * 1000.0

        h = result.h_route
        tau = self._calibrator.tau
        if self._temporal is not None:
            self._temporal.update(h)
            tau = self._temporal.adjusted_tau(tau)
        error_prob = self._calibrator.predict_error_prob(h)

        destination = (
            RoutingDestination.AWS_SAGEMAKER
            if h >= tau
            else RoutingDestination.GCP_CLOUD_RUN
        )

        # Estimate cost using probe latency as a lower-bound proxy until
        # the caller reports total_latency_ms via decision.set_total_latency().
        initial_cost, cost_model = estimate_cost(destination, probe_latency_ms)

        decision = RoutingDecision(
            request_id=request_id or str(uuid.uuid4()),
            destination=destination,
            h_route=h,
            tau=tau,
            error_probability=error_prob,
            input_tokens=result.input_tokens,
            probe_latency_ms=probe_latency_ms,
            total_latency_ms=None,
            timestamp_utc=time.time(),
            model_name=result.model_name,
            cost_per_request_usd=initial_cost,
            cost_model=cost_model,
            metadata=metadata or {},
        )

        logger.info(
            "route | id=%s h=%.4f tau=%.4f dest=%s probe_latency=%.1fms",
            decision.request_id,
            h, tau,
            destination.value,
            probe_latency_ms,
        )
        return decision

    def update_tau(self, new_tau: float) -> None:
        """Push an externally recalibrated tau into the calibrator."""
        self._calibrator.update_tau(new_tau)
