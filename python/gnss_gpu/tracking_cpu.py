"""CPU frequency refinement and gated, blockwise carrier/code-state following.

This is a noncoherent one-period estimator with temporal smoothing and code
aid, not a phase-locked loop, navigation decoder or ranging-integrity monitor.
Detection always uses the original, fixed, calibrated coarse search.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.acquisition_cpu import (
    CaAcquisition, CaSearch, CaSearchThreshold, detect_ca_search,
)
from gnss_gpu.code_delay_cpu import _prepare_ca_observation, _estimate_prepared_delay
from gnss_gpu.code_detection_cpu import ca_detection_statistic
from gnss_gpu.input_validation import finite_float, positive_float, positive_int
from gnss_gpu.signal_sim_cpu import L1_HZ, _sample_index


def refine_ca_carrier(iq, search: CaSearch, candidate: CaAcquisition, *,
                      tolerance_hz=.1, max_iterations=32):
    """Locally maximize the delay-profile score between neighboring grid bins.

    Supply a candidate obtained from this IQ and search. Golden-section search
    assumes a locally unimodal score near the coarse peak; it is not a global
    frequency solution. Refit delay at each evaluation, retaining the original
    candidate if no improvement is found. The bracket stays inside the original
    grid. A single-bin search cannot be refined. The refined score must NOT be
    tested against a threshold calibrated for the coarse grid: apply that gate
    first and use refinement only for parameter estimation.
    """
    tolerance = positive_float("tolerance_hz", tolerance_hz)
    iterations = positive_int("max_iterations", max_iterations)
    if (candidate.carrier_hz not in search.carrier_grid_hz
            or candidate.delay.sampling_freq != search.sampling_freq):
        raise ValueError("candidate must match the supplied coarse search")
    data = np.array(iq, copy=True)
    prepared = _prepare_ca_observation(data, search.sampling_freq, prn=search.prn)
    # Validate IQ even when the grid contains only one bin.
    fitted = _estimate_prepared_delay(prepared, carrier_hz=candidate.carrier_hz)
    if fitted is None:
        return None
    best = CaAcquisition(candidate.carrier_hz, fitted, ca_detection_statistic(fitted))

    def evaluate(frequency):
        nonlocal best
        delay = _estimate_prepared_delay(prepared, carrier_hz=frequency)
        if delay is None:
            raise RuntimeError("nonzero IQ lost its delay candidate during refinement")
        result = CaAcquisition(float(frequency), delay, ca_detection_statistic(delay))
        if result.statistic > best.statistic:
            best = result
        return result.statistic

    index = search.carrier_grid_hz.index(candidate.carrier_hz)
    left = search.carrier_grid_hz[max(index-1, 0)]
    right = search.carrier_grid_hz[min(index+1, len(search.carrier_grid_hz)-1)]
    if right-left <= tolerance:
        return best
    ratio = (np.sqrt(5.)-1)/2
    x = right-ratio*(right-left)
    y = left+ratio*(right-left)
    fx, fy = evaluate(x), evaluate(y)
    evaluate(left)
    evaluate(right)
    for _ in range(iterations):
        if right-left <= tolerance:
            break
        if fx >= fy:
            right, y, fy = y, x, fx
            x = right-ratio*(right-left)
            fx = evaluate(x)
        else:
            left, x, fx = x, y, fy
            y = left+ratio*(right-left)
            fy = evaluate(y)
    return best


@dataclass(frozen=True)
class CaTrackResult:
    start_sample: int
    status: str
    detection_accepted: bool
    measurement_used: bool
    detection_statistic: float
    coarse_candidate: Optional[CaAcquisition]
    refined_candidate: Optional[CaAcquisition]
    carrier_hz: Optional[float]
    delay_samples: Optional[float]
    missed_blocks: int
    confirmation_count: int = 0


class CpuCaTracker:
    """Fixed-search gated following of contiguous 1 ms observations.

    Each block gets the SAME whole-grid calibrated gate, even while following
    a path. Accepted blocks are frequency-refined. An exponential update uses
    carrier and wrapped-delay innovations; frequency aids code-delay prediction
    via d(delay_samples)/dt = -fs*(carrier-IF)/f_L1. The carrier prediction is
    constant, so sustained acceleration incurs smoothing lag. Bounds prevent
    abrupt parameter jumps but do not guarantee a correct path or valid lock.

    After coast_blocks consecutive unused measurements the state is discarded.
    Acquisition/reacquisition requires confirmation_blocks consecutive accepted,
    mutually consistent candidates (default two). Pending candidates are exposed
    as confirming, without publishing carrier/delay state. A rejection clears
    confirmation; an inconsistent accepted candidate starts a new chain.
    confirmation_blocks=1 retains the earlier single-block behavior.
    Coasting outputs are predictions, not measurements. A supplied timestamp
    must be contiguous; missing samples require reset() and fresh acquisition.
    Constant nav polarity is required within each block, not between blocks.
    No carrier phase unwrapping, PLL/FLL/DLL or navigation bit synchronization
    is implemented. Per-block gate calibration is not a track false-alarm bound.
    """

    def __init__(self, threshold: CaSearchThreshold, *, intermediate_freq=0.,
                 gain=.25, max_frequency_step_hz=500., max_delay_step_samples=1.,
                 coast_blocks=3, workers=1, confirmation_blocks=2):
        self.threshold = threshold
        self.intermediate_freq = finite_float("intermediate_freq", intermediate_freq)
        self.gain = positive_float("gain", gain)
        if self.gain > 1:
            raise ValueError("gain must be at most one")
        self.max_frequency_step_hz = positive_float("max_frequency_step_hz", max_frequency_step_hz)
        self.max_delay_step_samples = positive_float("max_delay_step_samples", max_delay_step_samples)
        self.coast_blocks = positive_int("coast_blocks", coast_blocks)
        self.confirmation_blocks = _sample_index("confirmation_blocks", confirmation_blocks)
        if self.confirmation_blocks < 1:
            raise ValueError("confirmation_blocks must be a positive integer")
        self.workers = positive_int("workers", workers)
        self.count = round(threshold.search.sampling_freq*.001)
        self.reset()

    def reset(self):
        self._next_start = None
        self._carrier = None
        self._delay = None
        self._missed = 0
        self._pending_carrier = None
        self._pending_delay = None
        self._confirmation_count = 0

    def update(self, iq, *, start_sample):
        start = _sample_index("start_sample", start_sample)
        if self._next_start is not None and start != self._next_start:
            raise ValueError("IQ blocks must be contiguous; reset before a timestamp gap")
        if start+self.count >= 2**53:
            raise ValueError("block end must remain below 2**53 samples")
        detection = detect_ca_search(iq, self.threshold, workers=self.workers)
        refined = None
        if detection.accepted and detection.candidate is not None:
            refined = refine_ca_carrier(iq, self.threshold.search, detection.candidate)
        # Commit timestamps only after input validation and estimation succeed.
        self._next_start = start+self.count
        predicted_delay = None
        if self._carrier is not None and self._delay is not None:
            predicted_delay = self._wrap(self._delay-self.count*(self._carrier-self.intermediate_freq)/L1_HZ)
        used = False
        confirmation_count = 0
        status = "searching"
        if refined is not None:
            if self._carrier is None or predicted_delay is None:
                consistent = False
                if self._pending_carrier is not None and self._pending_delay is not None:
                    pending_prediction = self._wrap(
                        self._pending_delay-self.count*(self._pending_carrier-self.intermediate_freq)/L1_HZ)
                    consistent = (
                        abs(refined.carrier_hz-self._pending_carrier) <= self.max_frequency_step_hz
                        and abs(self._wrap(refined.delay.delay_samples-pending_prediction)) <= self.max_delay_step_samples)
                self._confirmation_count = self._confirmation_count+1 if consistent else 1
                self._pending_carrier = refined.carrier_hz
                self._pending_delay = refined.delay.delay_samples
                confirmation_count = self._confirmation_count
                status = "confirming"
                if confirmation_count >= self.confirmation_blocks:
                    self._carrier = refined.carrier_hz
                    self._delay = refined.delay.delay_samples
                    status, used = "acquired", True
                    self._clear_confirmation()
            else:
                frequency_error = refined.carrier_hz-self._carrier
                delay_error = self._wrap(refined.delay.delay_samples-predicted_delay)
                if (abs(frequency_error) <= self.max_frequency_step_hz
                        and abs(delay_error) <= self.max_delay_step_samples):
                    self._carrier += self.gain*frequency_error
                    self._delay = self._wrap(predicted_delay+self.gain*delay_error)
                    status, used = "tracking", True
        elif self._carrier is None:
            self._clear_confirmation()
        if used:
            self._missed = 0
        elif self._carrier is not None:
            self._missed += 1
            self._delay = predicted_delay
            status = "coasting"
            if self._missed >= self.coast_blocks:
                self._carrier = self._delay = None
                self._clear_confirmation()
                status = "lost"
        else:
            self._missed += 1
        return CaTrackResult(start, status, detection.accepted, used, detection.statistic,
                             detection.candidate, refined, self._carrier, self._delay, self._missed,
                             confirmation_count)

    def _clear_confirmation(self):
        self._pending_carrier = None
        self._pending_delay = None
        self._confirmation_count = 0

    def _wrap(self, delay):
        return float((delay+self.count/2) % self.count-self.count/2)
