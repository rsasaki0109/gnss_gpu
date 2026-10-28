"""Fixed-search, disjoint-window following using calibrated weak acquisition.

This follows coarse carrier/code states; it is not a PLL/FLL/DLL or navigation
decoder. A marginal acquisition gate supplies no whole-track false-alarm bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeCandidate
from gnss_gpu.input_validation import finite_float, positive_float
from gnss_gpu.signal_sim_cpu import L1_HZ, _sample_index
from gnss_gpu.weak_acquisition_cpu import CaWeakAcquisitionThreshold, detect_weak_ca


@dataclass(frozen=True)
class CaWeakTrackResult:
    start_sample: int
    status: str
    detection_accepted: bool
    measurement_used: bool
    detection_statistic: float
    candidate: Optional[CaBitEdgeCandidate]
    carrier_hz: Optional[float]
    delay_samples: Optional[float]
    missed_windows: int
    confirmation_count: int


class CpuWeakCaTracker:
    """Follow contiguous windows with the same complete calibrated search.

    No state-dependent search narrowing, gate change or truth input is used.
    Candidate delay refers to the first row of each window. Code prediction
    advances by the full window duration, using estimated carrier minus known
    IF as L1 Doppler. Coasting values are predictions, never measurements.
    Confirmation and smoothing are heuristics, not a lock/integrity guarantee.
    """

    def __init__(self, threshold: CaWeakAcquisitionThreshold, *, gain=.25,
                 max_frequency_step_hz=75., max_delay_step_samples=2.,
                 confirmation_windows=2, coast_windows=2, intermediate_freq=0.):
        self.threshold = threshold
        self.gain = positive_float('gain', gain)
        if self.gain > 1:
            raise ValueError('gain must be at most one')
        self.max_frequency_step_hz = positive_float('max_frequency_step_hz', max_frequency_step_hz)
        self.max_delay_step_samples = positive_float('max_delay_step_samples', max_delay_step_samples)
        self.intermediate_freq = finite_float('intermediate_freq', intermediate_freq)
        self.confirmation_windows = _sample_index('confirmation_windows', confirmation_windows)
        self.coast_windows = _sample_index('coast_windows', coast_windows)
        if min(self.confirmation_windows, self.coast_windows) < 1:
            raise ValueError('confirmation and coast windows must be positive integers')
        self.period_samples = round(threshold.detection.config.search.sampling_freq*.001)
        self.window_samples = self.period_samples*threshold.detection.config.periods
        self.reset()

    def reset(self):
        self._next_start = None
        self._carrier = self._delay = None
        self._pending = None
        self._confirmation = self._missed = 0

    def _wrap(self, delay):
        return float((delay+self.period_samples/2) % self.period_samples-self.period_samples/2)

    def _predict(self, carrier, delay):
        return self._wrap(delay-self.window_samples*(carrier-self.intermediate_freq)/L1_HZ)

    def _consistent(self, carrier, delay, other_carrier, other_delay):
        return (abs(carrier-other_carrier) <= self.max_frequency_step_hz
                and abs(self._wrap(delay-other_delay)) <= self.max_delay_step_samples)

    def update(self, iq, *, start_sample):
        start = _sample_index('start_sample', start_sample)
        if self._next_start is not None and start != self._next_start:
            raise ValueError('IQ windows must be contiguous; reset before a timestamp gap')
        if start+self.window_samples >= 2**53:
            raise ValueError('window end must remain below 2**53 samples')
        # Estimation/validation must succeed before committing any state.
        detection = detect_weak_ca(iq, self.threshold).detection
        self._next_start = start+self.window_samples
        candidate = detection.candidate if detection.accepted else None
        predicted = self._predict(self._carrier, self._delay) if self._carrier is not None else None
        used, confirmation, status = False, 0, 'searching'
        if candidate is not None:
            if self._carrier is None:
                consistent = self._pending is not None and self._consistent(
                    candidate.carrier_hz, candidate.delay_samples, self._pending[0],
                    self._predict(*self._pending))
                self._confirmation = self._confirmation+1 if consistent else 1
                self._pending = (candidate.carrier_hz, candidate.delay_samples)
                confirmation, status = self._confirmation, 'confirming'
                if confirmation >= self.confirmation_windows:
                    self._carrier, self._delay = self._pending
                    self._pending, self._confirmation = None, 0
                    used, status = True, 'acquired'
            elif self._consistent(candidate.carrier_hz, candidate.delay_samples, self._carrier, predicted):
                assert predicted is not None
                self._carrier += self.gain*(candidate.carrier_hz-self._carrier)
                self._delay = self._wrap(predicted+self.gain*self._wrap(candidate.delay_samples-predicted))
                used, status = True, 'tracking'
        elif self._carrier is None:
            self._pending, self._confirmation = None, 0
        if used:
            self._missed = 0
        elif self._carrier is not None:
            self._missed += 1
            self._delay = predicted
            status = 'coasting'
            if self._missed >= self.coast_windows:
                self._carrier = self._delay = None
                self._pending, self._confirmation = None, 0
                status = 'lost'
        else:
            self._missed += 1
        return CaWeakTrackResult(start, status, detection.accepted, used, detection.statistic,
                                 detection.candidate, self._carrier, self._delay, self._missed, confirmation)
