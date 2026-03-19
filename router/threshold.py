"""
Threshold calibration module.

Implements isotonic regression over (entropy_score, error_label) pairs to
produce a monotone function f: R -> [0,1] approximating P(error | H_route).
The routing threshold tau is the entropy value where f(H) crosses a
user-specified risk tolerance epsilon.

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
           adapted to attention-space features.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

_DEFAULT_RISK_TOLERANCE = 0.15  # epsilon: acceptable P(error) before escalating
_DEFAULT_TAU = 2.0              # fallback tau if calibration data is unavailable


@dataclass
class CalibrationResult:
    """Output of a calibration run."""
    tau: float               # routing threshold
    risk_tolerance: float    # epsilon used to derive tau
    auroc: float             # AUROC of isotonic model on calibration set
    n_samples: int           # number of (entropy, error) pairs used
    tau_history: list        # previous tau values for trend tracking

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved calibration result to %s (tau=%.4f)", path, self.tau)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        data = json.loads(Path(path).read_text())
        return cls(**data)


class ThresholdCalibrator:
    """
    Fits a monotone isotonic regression model on (entropy, error) pairs and
    exposes a calibrated routing threshold tau.

    Usage
    -----
        calibrator = ThresholdCalibrator(risk_tolerance=0.15)
        result = calibrator.fit(entropy_scores, error_labels)
        tau = result.tau

    Parameters
    ----------
    risk_tolerance:
        Target P(error) epsilon. Routing threshold is set to the minimum
        entropy where f(H) >= epsilon, i.e., the first entropy value where
        the model predicts error probability at or above the tolerance.
    """

    def __init__(self, risk_tolerance: float = _DEFAULT_RISK_TOLERANCE) -> None:
        self.risk_tolerance = risk_tolerance
        self._iso: Optional[IsotonicRegression] = None
        self._entropy_grid: Optional[np.ndarray] = None
        self._tau: float = _DEFAULT_TAU

    @property
    def tau(self) -> float:
        """Current routing threshold. Falls back to default if not fitted."""
        return self._tau

    def fit(
        self,
        entropy_scores: np.ndarray,
        error_labels: np.ndarray,
        previous_tau_history: Optional[list] = None,
    ) -> CalibrationResult:
        """
        Fit isotonic regression on entropy-error pairs.

        Parameters
        ----------
        entropy_scores:
            1-D array of H_route values computed by AttentionEntropyProbe.
        error_labels:
            1-D binary array; 1 = model made an error on this input.
        previous_tau_history:
            List of previous tau values to append to for trend tracking.

        Returns
        -------
        CalibrationResult with the new tau and AUROC.
        """
        entropy_scores = np.asarray(entropy_scores, dtype=np.float64)
        error_labels = np.asarray(error_labels, dtype=np.float64)

        if len(entropy_scores) != len(error_labels):
            raise ValueError(
                f"entropy_scores length {len(entropy_scores)} != "
                f"error_labels length {len(error_labels)}"
            )
        if len(entropy_scores) < 10:
            logger.warning(
                "Only %d samples for calibration — tau may be unreliable",
                len(entropy_scores),
            )

        self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._iso.fit(entropy_scores, error_labels)

        # Build a fine evaluation grid over the observed entropy range
        lo, hi = entropy_scores.min(), entropy_scores.max()
        self._entropy_grid = np.linspace(lo, hi, num=1000)
        predicted_probs = self._iso.predict(self._entropy_grid)

        # tau = first grid point where predicted error prob >= epsilon
        tau = self._find_threshold(predicted_probs)
        self._tau = tau

        # AUROC
        preds_on_data = self._iso.predict(entropy_scores)
        auroc = _safe_auroc(error_labels, preds_on_data)

        history = list(previous_tau_history or [])
        history.append(tau)

        result = CalibrationResult(
            tau=tau,
            risk_tolerance=self.risk_tolerance,
            auroc=auroc,
            n_samples=len(entropy_scores),
            tau_history=history,
        )
        logger.info(
            "Calibration complete: tau=%.4f  AUROC=%.4f  n=%d",
            tau, auroc, len(entropy_scores),
        )
        return result

    def predict_error_prob(self, entropy: float) -> float:
        """
        Return P(error | entropy) using the fitted isotonic model.
        Returns 1.0 (always escalate) if model is not fitted.
        """
        if self._iso is None:
            logger.warning("Calibrator not fitted; returning P(error)=1.0")
            return 1.0
        return float(self._iso.predict([entropy])[0])

    def _find_threshold(self, predicted_probs: np.ndarray) -> float:
        """
        Return the entropy grid value where predicted P(error) first reaches
        the risk tolerance. Falls back to the maximum observed entropy if the
        error model never reaches epsilon (i.e., always route to fast path).
        """
        above = np.where(predicted_probs >= self.risk_tolerance)[0]
        if len(above) == 0:
            tau = float(self._entropy_grid[-1])
            logger.warning(
                "P(error) never reached epsilon=%.3f in entropy range; "
                "setting tau to max observed entropy %.4f",
                self.risk_tolerance, tau,
            )
            return tau
        return float(self._entropy_grid[above[0]])

    def update_tau(self, new_tau: float) -> None:
        """Manually override tau (e.g., loaded from a parameter store)."""
        logger.info("Updating tau: %.4f -> %.4f", self._tau, new_tau)
        self._tau = new_tau


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC; returns 0.5 if only one class is present."""
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class in calibration labels; AUROC undefined, returning 0.5")
        return 0.5
    return float(roc_auc_score(labels, scores))
