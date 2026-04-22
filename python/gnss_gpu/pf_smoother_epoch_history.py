"""Forward-epoch history for PF smoother evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class ForwardEpochHistory:
    prev_tow: float | None = None
    prev_measurements: list[Any] | None = None
    prev_estimate: np.ndarray | None = None
    prev_pf_estimate: np.ndarray | None = None
    prev_pf_state: np.ndarray | None = None
    epochs_done: int = 0

    def reached_limit(self, max_epochs: int, skip_valid_epochs: int) -> bool:
        return bool(max_epochs) and self.epochs_done >= int(skip_valid_epochs) + int(max_epochs)

    def dt_for(self, tow: float, default_dt: float = 0.1) -> float:
        if self.prev_tow is None:
            return float(default_dt)
        return float(tow) - float(self.prev_tow)

    def has_previous_motion(self, dt: float) -> bool:
        return self.prev_tow is not None and float(dt) > 0.0

    def advance(
        self,
        *,
        tow: float,
        measurements: Iterable[Any],
        pf_estimate_now: np.ndarray,
        pf_state: np.ndarray,
    ) -> None:
        pf_estimate = np.asarray(pf_estimate_now, dtype=np.float64).copy()
        self.prev_tow = float(tow)
        self.prev_measurements = list(measurements)
        self.prev_estimate = pf_estimate.copy()
        self.prev_pf_estimate = pf_estimate.copy()
        self.prev_pf_state = np.asarray(pf_state, dtype=np.float64).copy()
        self.epochs_done += 1
