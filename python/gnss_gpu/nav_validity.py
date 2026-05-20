"""Pure broadcast-record validity predicates.

Centralising these guards keeps the orbit / ephemeris orchestration code free
of constellation-specific ``if nav.system == ...`` branches: callers ask the
predicate ``is_broadcast_usable(nav)`` and route through the right propagator
once.  Each predicate is a pure function over the NavMessage-shaped record so
the predicates can be unit-tested with light-weight dataclasses or stubs.
"""

from __future__ import annotations

import math
from typing import Any


# Legacy BDS-2 + BDS-3 GEO PRN assignments commonly seen in mixed RINEX.
# Excluded from the broadcast-usable set because GEO-specific broadcast-frame
# handling is not yet implemented in this codebase — feel free to widen the
# coverage by adding a dedicated GEO orbit module and dropping these PRNs.
BEIDOU_GEO_PRNS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 59, 60, 61, 62, 63})


def is_kepler_broadcast_valid(nav: Any | None) -> bool:
    """True iff the record has finite, in-range Kepler elements.

    Accepts any object exposing ``system``, ``sqrt_a``, ``e`` — typically a
    ``NavMessage``.  Rejects GLONASS (``"R"``) and SBAS (``"S"``) since they do
    not carry Kepler broadcasts at all.
    """

    if nav is None:
        return False
    if getattr(nav, "system", None) in {"R", "S"}:
        return False
    sqrt_a = float(getattr(nav, "sqrt_a", 0.0))
    if not math.isfinite(sqrt_a) or sqrt_a <= 0.0:
        return False
    eccentricity = float(getattr(nav, "e", 0.0))
    if not math.isfinite(eccentricity) or not (0.0 <= eccentricity < 1.0):
        return False
    return True


def is_glonass_broadcast_valid(nav: Any | None) -> bool:
    """True iff the GLONASS state vector is plausible (orbit radius bounded)."""

    if nav is None or getattr(nav, "system", None) != "R":
        return False
    px = float(getattr(nav, "glo_px_m", 0.0))
    py = float(getattr(nav, "glo_py_m", 0.0))
    pz = float(getattr(nav, "glo_pz_m", 0.0))
    radius = math.sqrt(px * px + py * py + pz * pz)
    if not math.isfinite(radius):
        return False
    # GLONASS MEOs sit near r ≈ 25,500 km — be generous either side.
    return 1.5e7 <= radius <= 5.0e7


def is_beidou_geo(nav: Any | None) -> bool:
    """True iff the record is a BeiDou GEO PRN in the exclusion list."""

    if nav is None or getattr(nav, "system", None) != "C":
        return False
    try:
        prn = int(getattr(nav, "prn"))
    except (TypeError, ValueError):
        return False
    return prn in BEIDOU_GEO_PRNS


def is_broadcast_usable(nav: Any | None) -> bool:
    """Composite predicate used by the ephemeris orchestrator.

    A record is usable if it is either a valid GLONASS state vector or a valid
    Kepler broadcast, and is not a (currently excluded) BeiDou GEO satellite.
    """

    if is_beidou_geo(nav):
        return False
    return is_glonass_broadcast_valid(nav) or is_kepler_broadcast_valid(nav)


__all__ = [
    "BEIDOU_GEO_PRNS",
    "is_kepler_broadcast_valid",
    "is_glonass_broadcast_valid",
    "is_beidou_geo",
    "is_broadcast_usable",
]
