"""Truth-free commit gate for integer-basin RTK FIX decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrustedFixGateDecision:
    passed: bool
    float_consistent: bool
    ddpr_consistent: bool
    ddpr_supported: bool
    ddpr_fresh: bool


def trusted_fix_gate(
    *,
    map_float_separation_m: float,
    map_ddpr_separation_m: float,
    last_ddpr_pairs: int,
    ddpr_age_epochs: int,
    max_float_separation_m: float,
    max_ddpr_separation_m: float,
    min_ddpr_pairs: int,
    max_ddpr_age_epochs: int,
) -> TrustedFixGateDecision:
    """Require agreement with independent float and DDPR-only navigation arms."""

    float_ok = bool(
        math.isfinite(map_float_separation_m)
        and map_float_separation_m <= max_float_separation_m
    )
    ddpr_ok = bool(
        math.isfinite(map_ddpr_separation_m)
        and map_ddpr_separation_m <= max_ddpr_separation_m
    )
    support_ok = int(last_ddpr_pairs) >= int(min_ddpr_pairs)
    fresh_ok = 0 <= int(ddpr_age_epochs) <= int(max_ddpr_age_epochs)
    return TrustedFixGateDecision(
        passed=bool(float_ok and ddpr_ok and support_ok and fresh_ok),
        float_consistent=float_ok,
        ddpr_consistent=ddpr_ok,
        ddpr_supported=support_ok,
        ddpr_fresh=fresh_ok,
    )
