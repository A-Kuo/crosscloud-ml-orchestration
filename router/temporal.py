"""
Temporal routing helper with sliding-window drift signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalDriftController:
    window_size: int = 200
    z_threshold: float = 2.0
    tau_offset: float = -0.2

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.window_size)
        self._baseline_mean = 0.0
        self._baseline_std = 1.0

    def set_baseline(self, baseline_entropies: list[float]) -> None:
        arr = np.asarray(baseline_entropies, dtype=float)
        if arr.size == 0:
            return
        self._baseline_mean = float(arr.mean())
        self._baseline_std = float(arr.std() + 1e-9)

    def update(self, entropy: float) -> None:
        self._window.append(float(entropy))

    def adjusted_tau(self, tau: float) -> float:
        if len(self._window) < max(10, self.window_size // 4):
            return tau
        recent = float(np.mean(self._window))
        z = (recent - self._baseline_mean) / self._baseline_std
        if z > self.z_threshold:
            # Lowering tau escalates more requests to the heavy path.
            return tau + self.tau_offset
        return tau
