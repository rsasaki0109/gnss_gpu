from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, CaBitEdgeThreshold, CaBitEdgeCandidate
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig
from gnss_gpu.weak_acquisition_cpu import CaWeakAcquisitionThreshold
from gnss_gpu.weak_tracking_cpu import CpuWeakCaTracker
from tests.test_ca_cancellation_cpu import observation, FS, N


def gate():
    search = CaBitEdgeSearch(CaSearch(FS, (-250., 0., 250.)))
    return CaWeakAcquisitionThreshold(CaCancellationConfig(FS, nuisance_prns=()),
                                    CaBitEdgeThreshold(search, 20., (), .01, .05, 1))


def mock_detector(monkeypatch, candidates):
    iterator = iter(candidates)
    def detect(iq, threshold):
        candidate = next(iterator)
        if isinstance(candidate, Exception):
            raise candidate
        return SimpleNamespace(detection=SimpleNamespace(accepted=candidate is not None,
            statistic=candidate.statistic if candidate else 0., candidate=candidate))
    monkeypatch.setattr('gnss_gpu.weak_tracking_cpu.detect_weak_ca', detect)


def candidate(frequency=175., lag=37):
    return CaBitEdgeCandidate(frequency, lag, None, 250., 100.)


def test_confirmation_loss_and_reacquisition_distinguish_measurements_and_predictions(monkeypatch):
    mock_detector(monkeypatch, [candidate(), candidate(), candidate(), None, None, candidate(), candidate()])
    tracker = CpuWeakCaTracker(gate())
    rows = [tracker.update(None, start_sample=i*20*N) for i in range(7)]
    assert [r.status for r in rows] == ['confirming', 'acquired', 'tracking', 'coasting', 'lost', 'confirming', 'acquired']
    assert rows[0].carrier_hz is None and rows[5].carrier_hz is None
    assert rows[3].carrier_hz is not None and not rows[3].measurement_used
    assert rows[4].carrier_hz is None
    assert rows[6].confirmation_count == 2


def test_rejection_and_inconsistent_candidates_restart_confirmation(monkeypatch):
    mock_detector(monkeypatch, [candidate(), None, candidate(), candidate(lag=100), candidate(lag=100)])
    tracker = CpuWeakCaTracker(gate())
    rows = [tracker.update(None, start_sample=i*20*N) for i in range(5)]
    assert [r.confirmation_count for r in rows] == [1, 0, 1, 1, 2]
    assert rows[-1].status == 'acquired'


def test_code_prediction_uses_full_window_and_removes_known_if(monkeypatch):
    # 50 kHz IF would produce 1.3 samples of false motion in 20 ms.
    mock_detector(monkeypatch, [candidate(50000., 1022), candidate(50000., -1023)])
    tracker = CpuWeakCaTracker(gate(), intermediate_freq=50000., max_delay_step_samples=1.01)
    tracker.update(None, start_sample=0)
    assert tracker.update(None, start_sample=20*N).status == 'acquired'


def test_failed_estimation_and_timestamp_gap_preserve_pending_state(monkeypatch):
    mock_detector(monkeypatch, [candidate(), ValueError('bad IQ'), candidate(), candidate()])
    tracker = CpuWeakCaTracker(gate())
    tracker.update(None, start_sample=0)
    with pytest.raises(ValueError, match='contiguous'):
        tracker.update(None, start_sample=40*N)
    with pytest.raises(ValueError, match='bad IQ'):
        tracker.update(None, start_sample=20*N)
    assert tracker.update(None, start_sample=20*N).status == 'acquired'
    tracker.reset()
    assert tracker.update(None, start_sample=100*N).status == 'confirming'


def test_integrated_iq_uses_calibrated_detector_and_preserves_inputs():
    tracker = CpuWeakCaTracker(gate())
    iq = observation(1, 175., 201)
    saved = iq.copy()
    assert tracker.update(iq, start_sample=0).status == 'confirming'
    assert tracker.update(-iq, start_sample=20*N).status == 'acquired'
    np.testing.assert_array_equal(iq, saved)
    with pytest.raises(ValueError):
        tracker.update(iq[:19], start_sample=40*N)
    assert tracker.update(iq, start_sample=40*N).measurement_used


@pytest.mark.parametrize('changed', [dict(gain=2.), dict(confirmation_windows=0), dict(coast_windows=True),
                                    dict(max_frequency_step_hz=-1.), dict(intermediate_freq=np.nan)])
def test_invalid_configuration(changed):
    with pytest.raises(ValueError):
        CpuWeakCaTracker(gate(), **changed)
