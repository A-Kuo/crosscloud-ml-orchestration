"""
Calibration artifact storage and rollback helpers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationArtifact:
    tau: float
    auroc: float
    fit_date: float
    n_samples: int
    prev_tau: Optional[float]
    risk_tolerance: float
    notes: str = ""

    def is_valid(self, max_age_days: int = 30) -> bool:
        age_seconds = time.time() - self.fit_date
        return age_seconds < max_age_days * 86400

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationArtifact":
        return cls(**payload)


class CalibrationArtifactManager:
    def __init__(self, base_path: str | Path = "router/artifacts") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.base_path / "latest.json"
        self.history_path = self.base_path / "history.json"

    def save(self, artifact: CalibrationArtifact) -> None:
        self.latest_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
        history = self.load_history()
        history.append(artifact.to_dict())
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def load_latest(self) -> Optional[CalibrationArtifact]:
        if not self.latest_path.exists():
            return None
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        return CalibrationArtifact.from_dict(payload)

    def load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def rollback(self) -> Optional[CalibrationArtifact]:
        history = self.load_history()
        if len(history) < 2:
            return None
        history.pop()
        previous = CalibrationArtifact.from_dict(history[-1])
        self.latest_path.write_text(json.dumps(previous.to_dict(), indent=2), encoding="utf-8")
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return previous
