"""Small explicit cache for sharing parsed RINEX observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnss_gpu.io.rinex import read_rinex_obs


class RinexObservationCache:
    """Cache parsed observation files by resolved path for one experiment."""

    def __init__(self) -> None:
        self._observations: dict[Path, Any] = {}

    def load(self, path: str | Path) -> Any:
        key = Path(path).resolve()
        observation = self._observations.get(key)
        if observation is None:
            observation = read_rinex_obs(key)
            self._observations[key] = observation
        return observation

    def __len__(self) -> int:
        return len(self._observations)
