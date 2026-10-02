"""Small shared helpers for PF smoother experiment support modules."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        # Unsupported values raise TypeError/ValueError, handled below.
        out = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out
