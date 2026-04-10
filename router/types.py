"""
Lightweight types shared across the router stack (no torch/transformers).

Keeping EntropyResult here avoids importing heavy ML deps when only data
structures are needed (e.g. ASGI tests with mocks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EntropyResult:
    """Detailed result from a single entropy computation pass."""

    h_route: float
    per_head_entropies: np.ndarray
    input_tokens: int
    model_name: str

    def to_dict(self) -> dict:
        return {
            "h_route": self.h_route,
            "per_head_entropies": self.per_head_entropies.tolist(),
            "input_tokens": self.input_tokens,
            "model_name": self.model_name,
        }
