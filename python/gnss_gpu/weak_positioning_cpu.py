"""Assisted C/A code-period resolution and fresh-state pseudorange positioning.

Requires known code epoch alignment and independent coarse range/clock bounds.
This does not decode GNSS time or resolve unassisted millisecond ambiguities.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float
from gnss_gpu.positioning_cpu import PositionSolution, solve_pseudorange_position
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class CaRangeResolution:
    status: str
    pseudorange_m: Optional[float]
    period_index: Optional[int]
    candidate_count: int


def resolve_ca_range(delay_samples, sampling_freq, *, prior_range_m, prior_bound_m):
    """Return a range only if the supplied closed prior interval contains one candidate.

    Positive delay means additional propagation time. The supplied lag must be
    referenced to a known common transmit/receive code epoch. The coarse prior
    must include geometric, receiver-clock and unapplied correction uncertainty.
    A wrong prior can select a wrong period; this is not an integrity guarantee.
    A 1 micrometre numeric padding is included at interval boundaries.
    """
    fs = positive_float("sampling_freq", sampling_freq)
    spc = round(fs / CHIP_RATE)
    if spc < 2 or not np.isclose(fs, spc * CHIP_RATE, rtol=0.0, atol=1e-6):
        raise ValueError("require an integer number of samples per chip, at least two")
    delay = finite_float("delay_samples", delay_samples)
    prior = positive_float("prior_range_m", prior_range_m)
    bound = positive_float("prior_bound_m", prior_bound_m)
    period_samples = 1023 * spc
    period_m = period_samples * C_LIGHT / fs
    base = (delay % period_samples) * C_LIGHT / fs
    lower, upper = max(0.0, prior - bound), prior + bound
    if not np.isfinite(upper):
        raise ValueError("range interval must be finite")
    first = max(0, math.ceil((lower - base - 1e-6) / period_m))
    last = math.floor((upper - base + 1e-6) / period_m)
    if base == 0.0:
        first = max(1, first)
    count = max(0, last - first + 1)
    if count != 1:
        return CaRangeResolution(
            "ambiguous" if count > 1 else "outside_prior", None, None, count
        )
    return CaRangeResolution("resolved", float(base + first * period_m), first, 1)


@dataclass(frozen=True)
class CaWeakPositionResult:
    status: str
    solution: Optional[PositionSolution]
    start_sample: Optional[int]
    used_prns: tuple
    pseudoranges_m: tuple
    rejected: tuple


def position_from_weak_tracks(
    states_by_prn,
    satellite_positions_m,
    range_priors_m,
    range_bounds_m,
    sampling_freq,
    *,
    initial_position_m=(0.0, 0.0, 0.0),
):
    """Use synchronized, fresh states and explicit per-PRN range assistance only.

    Coasting and pending states are excluded. Satellite coordinates/corrections
    must already use a common frame and epoch. Equal-weight position/clock LS;
    no truth gating, outlier rejection, ephemerides or atmospheric estimation.
    The result's timestamp refers to the window start, not processing availability.
    """
    fs = positive_float("sampling_freq", sampling_freq)
    period_samples = round(fs * 0.001)
    used, ranges, satellites, starts, rejected = [], [], [], set(), []
    for prn, state in sorted(states_by_prn.items()):
        key = _sample_index("prn", prn)
        if not 1 <= key <= 32:
            raise ValueError("PRN must be in 1..32")
        if (
            not state.measurement_used
            or state.delay_samples is None
            or state.status not in ("acquired", "tracking")
        ):
            rejected.append((key, "not_fresh"))
            continue
        if (
            key not in satellite_positions_m
            or key not in range_priors_m
            or key not in range_bounds_m
        ):
            rejected.append((key, "missing_assistance"))
            continue
        satellite = np.asarray(satellite_positions_m[key], dtype=float)
        if satellite.shape != (3,) or not np.all(np.isfinite(satellite)):
            raise ValueError("satellite coordinates must be finite (3,) arrays")
        start = _sample_index("start_sample", state.start_sample)
        # Tracker lag is relative to row zero. Restore the known receiver epoch
        # phase before resolving the globally aligned transmit code period.
        resolution = resolve_ca_range(
            state.delay_samples + start % period_samples,
            fs,
            prior_range_m=range_priors_m[key],
            prior_bound_m=range_bounds_m[key],
        )
        if resolution.status != "resolved":
            rejected.append((key, resolution.status))
            continue
        starts.add(start)
        used.append(key)
        ranges.append(resolution.pseudorange_m)
        satellites.append(satellite)
    if len(starts) > 1:
        raise ValueError("fresh ranges must have the same window-start timestamp")
    solution = (
        solve_pseudorange_position(
            satellites, ranges, initial_position_m=initial_position_m
        )
        if len(used) >= 4
        else None
    )
    status = (
        "solved"
        if solution is not None
        else (
            "insufficient_ranges"
            if len(used) < 4
            else "geometry_or_convergence_failure"
        )
    )
    return CaWeakPositionResult(
        status,
        solution,
        next(iter(starts), None),
        tuple(used),
        tuple(ranges),
        tuple(rejected),
    )
