"""Cross-cloud routing package."""

from __future__ import annotations

import importlib
from typing import Any

from .artifact import CalibrationArtifact, CalibrationArtifactManager
from .distillation import DistillationMetrics, distill_entropy_probe
from .learned_router import LearnedRouterResult, compare_learned_vs_isotonic
from .router import InferenceRouter, RoutingDecision, RoutingDestination
from .temporal import TemporalDriftController
from .threshold import ThresholdCalibrator
from .types import EntropyResult

__all__ = [
    "AttentionEntropyProbe",
    "CalibrationArtifact",
    "CalibrationArtifactManager",
    "DistillationMetrics",
    "distill_entropy_probe",
    "EntropyResult",
    "InferenceRouter",
    "LearnedRouterResult",
    "compare_learned_vs_isotonic",
    "RoutingDecision",
    "RoutingDestination",
    "TemporalDriftController",
    "ThresholdCalibrator",
]


def __getattr__(name: str) -> Any:
    if name == "AttentionEntropyProbe":
        mod = importlib.import_module(".entropy", __name__)
        return mod.AttentionEntropyProbe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
