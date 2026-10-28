"""Descriptive search-support diagnostics, separate from calibrated detection.

These labels describe where an estimate sits in the supplied search interval.
They are not an accuracy, integrity or out-of-grid probability estimate.
"""

from dataclasses import dataclass
from typing import Optional

from gnss_gpu.acquisition_cpu import CaAcquisition, CaSearch
from gnss_gpu.input_validation import finite_float, nonnegative_float


@dataclass(frozen=True)
class CaCarrierSupport:
    status: str
    carrier_hz: Optional[float]
    distance_to_boundary_hz: Optional[float]
    boundary_side: Optional[str]
    boundary_margin_hz: float


def assess_ca_carrier_support(candidate: Optional[CaAcquisition], search: CaSearch, *,
                              boundary_margin_hz=.1):
    """Label a current refined estimate without changing any receiver decision.

    Pass result.refined_candidate, not the smoothed/predicted track frequency.
    None gives ``not_estimated``; a single-bin grid gives ``fixed_frequency``
    because it provides no frequency resolution. Otherwise ``boundary`` means
    within the supplied inclusive margin of either endpoint; ``interior`` only
    means neither endpoint was reached. Both true edge signals and out-of-grid
    signals can produce boundary estimates. Interior estimates can be wrong.

    The default margin matches the default refinement tolerance (0.1 Hz).
    Callers using another tolerance should choose their margin explicitly.
    Applying this to a coarse candidate is conservative: an endpoint grid bin
    can refine to an interior frequency. No truth or IQ is inspected here.
    """
    margin = nonnegative_float("boundary_margin_hz", boundary_margin_hz)
    if candidate is None:
        return CaCarrierSupport("not_estimated", None, None, None, margin)
    if candidate.delay.sampling_freq != search.sampling_freq:
        raise ValueError("candidate sampling frequency must match search")
    frequency = finite_float("candidate carrier_hz", candidate.carrier_hz)
    lower, upper = search.carrier_grid_hz[0], search.carrier_grid_hz[-1]
    if not lower <= frequency <= upper:
        raise ValueError("candidate carrier_hz must lie inside search interval")
    if len(search.carrier_grid_hz) == 1:
        return CaCarrierSupport("fixed_frequency", frequency, 0., None, margin)
    low, high = frequency-lower, upper-frequency
    side = "both" if low <= margin and high <= margin else (
        "lower" if low <= margin else "upper" if high <= margin else None)
    return CaCarrierSupport("boundary" if side else "interior", frequency,
                            min(low, high), side, margin)
