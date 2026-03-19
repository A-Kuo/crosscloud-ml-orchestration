"""
Inference router.

Combines AttentionEntropyProbe and ThresholdCalibrator to make routing
decisions for incoming inference requests.

Routing logic:
    H_route < tau  ->  GCP Cloud Run  (fast path, low-uncertainty)
    H_route >= tau ->  AWS SageMaker  (heavy model, high-uncertainty)

Every decision is logged as a RoutingDecision dataclass so callers can emit
telemetry to BigQuery without depending on this module's internals.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from .entropy import AttentionEntropyProbe, EntropyResult
from .threshold import ThresholdCalibrator

logger = logging.getLogger(__name__)


class RoutingDestination(str, Enum):
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AWS_SAGEMAKER = "aws_sagemaker"


@dataclass
class RoutingDecision:
    """
    Immutable record of a single routing decision.

    Emitted by InferenceRouter.route() and intended to be forwarded to
    BigQuery for audit, drift detection, and post-hoc analysis.
    """
    request_id: str
    destination: RoutingDestination
    h_route: float
    tau: float
    error_probability: float           # P(error | H_route) from calibrator
    input_tokens: int
    latency_ms: float                  # time to compute entropy + decide
    timestamp_utc: float               # Unix timestamp
    model_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_escalated(self) -> bool:
        return self.destination == RoutingDestination.AWS_SAGEMAKER

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
        probe: Optional[AttentionEntropyProbe] = None,
        calibrator: Optional[ThresholdCalibrator] = None,
    ) -> None:
        self._probe = probe
        self._calibrator = calibrator if calibrator is not None else ThresholdCalibrator()

    @property
    def probe(self) -> AttentionEntropyProbe:
        if self._probe is None:
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
        latency_ms = (time.perf_counter() - t0) * 1000.0

        h = result.h_route
        tau = self._calibrator.tau
        error_prob = self._calibrator.predict_error_prob(h)

        destination = (
            RoutingDestination.AWS_SAGEMAKER
            if h >= tau
            else RoutingDestination.GCP_CLOUD_RUN
        )

        decision = RoutingDecision(
            request_id=request_id or str(uuid.uuid4()),
            destination=destination,
            h_route=h,
            tau=tau,
            error_probability=error_prob,
            input_tokens=result.input_tokens,
            latency_ms=latency_ms,
            timestamp_utc=time.time(),
            model_name=result.model_name,
            metadata=metadata or {},
        )

        logger.info(
            "route | id=%s h=%.4f tau=%.4f dest=%s latency=%.1fms",
            decision.request_id,
            h, tau,
            destination.value,
            latency_ms,
        )
        return decision

    def update_tau(self, new_tau: float) -> None:
        """Push an externally recalibrated tau into the calibrator."""
        self._calibrator.update_tau(new_tau)
