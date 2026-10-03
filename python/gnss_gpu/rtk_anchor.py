"""RTK fixed solutions used to anchor the PF smoother cloud."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# libgnsspp SolutionStatus value for an ambiguity-fixed RTK solution.
RTK_FIXED_STATUS = 4


def load_rtk_fix_lookup(
    pos_path: str | Path,
    *,
    load_solution_func: Callable[[str], Any] | None = None,
) -> dict[float, np.ndarray]:
    """Return ``{round(tow, 1): ecef}`` for the FIXED epochs of a .pos file.

    Reads gnssplusplus and RTKLIB solution files through
    ``libgnsspp.load_solution``.
    """

    path = Path(pos_path)
    if not path.exists():
        raise FileNotFoundError(f"RTK anchor solution not found: {path}")
    if load_solution_func is None:
        from libgnsspp import load_solution

        solution = load_solution(str(path))
    else:
        solution = load_solution_func(str(path))
    lookup: dict[float, np.ndarray] = {}
    for record in solution.records():
        if int(record.status) != RTK_FIXED_STATUS:
            continue
        ecef = np.asarray(record.position_ecef_m, dtype=np.float64).ravel()[:3]
        if ecef.size == 3 and np.all(np.isfinite(ecef)):
            lookup[round(float(record.time.tow), 1)] = ecef
    return lookup
