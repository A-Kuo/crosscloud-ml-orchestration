"""
Probe distillation scaffolding for entropy regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


@dataclass
class DistillationMetrics:
    mae_entropy: float
    teacher_auroc: float
    student_auroc: float
    auroc_drop: float


def distill_entropy_probe(
    teacher_entropy: Sequence[float],
    token_features: np.ndarray,
    labels: Sequence[int],
) -> DistillationMetrics:
    """
    Lightweight stand-in for a full neural distillation process.
    Uses tree regression to mimic teacher entropy from cheap token features.
    """
    teacher = np.asarray(teacher_entropy, dtype=float)
    y = np.asarray(labels, dtype=int)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(token_features, teacher)
    student_entropy = model.predict(token_features)

    mae = float(mean_absolute_error(teacher, student_entropy))
    teacher_auroc = float(roc_auc_score(y, teacher))
    student_auroc = float(roc_auc_score(y, student_entropy))
    return DistillationMetrics(
        mae_entropy=mae,
        teacher_auroc=teacher_auroc,
        student_auroc=student_auroc,
        auroc_drop=teacher_auroc - student_auroc,
    )
