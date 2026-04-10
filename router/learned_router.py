"""
Learned router model (MLP) and comparison utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from .threshold import ThresholdCalibrator


def features_from_entropy(entropies: np.ndarray) -> np.ndarray:
    e = np.asarray(entropies, dtype=float).reshape(-1, 1)
    return np.hstack([e, e ** 2, np.log1p(np.maximum(e, 0.0))])


@dataclass
class LearnedRouterResult:
    mlp_auroc: float
    isotonic_auroc: float
    calibration_gap: float


def compare_learned_vs_isotonic(entropies: np.ndarray, labels: np.ndarray) -> LearnedRouterResult:
    X = features_from_entropy(entropies)
    y = np.asarray(labels, dtype=int)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), random_state=42, max_iter=600)
    mlp.fit(X, y)
    mlp_scores = mlp.predict_proba(X)[:, 1]
    mlp_auroc = float(roc_auc_score(y, mlp_scores))

    cal = ThresholdCalibrator(risk_tolerance=0.15)
    cal.fit(entropies, labels)
    iso_scores = np.array([cal.predict_error_prob(float(e)) for e in entropies])
    isotonic_auroc = float(roc_auc_score(y, iso_scores))

    frac_pos, mean_pred = calibration_curve(y, mlp_scores, n_bins=10)
    calibration_gap = float(np.mean(np.abs(frac_pos - mean_pred)))

    return LearnedRouterResult(
        mlp_auroc=mlp_auroc,
        isotonic_auroc=isotonic_auroc,
        calibration_gap=calibration_gap,
    )
