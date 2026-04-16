"""Tests for E2E acquisition/pseudorange helpers."""

import numpy as np
import pytest


from gnss_gpu.e2e_helpers import (
    C_LIGHT,
    GPS_CA_PERIOD_M,
    acquisition_code_phase_to_pseudorange,
    acquisition_lag_to_code_phase_chips,
    code_phase_chips_to_acquisition_lag,
    compute_e2e_wls_weights,
    pseudorange_to_code_phase_chips,
    refine_acquisition_code_lag_dll,
)

signal_sim = pytest.importorskip(
    "gnss_gpu.signal_sim",
    reason="CUDA signal simulator bindings not available",
)
acquisition = pytest.importorskip(
    "gnss_gpu.acquisition",
    reason="CUDA acquisition bindings not available",
)

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]

SignalSimulator = signal_sim.SignalSimulator
Acquisition = acquisition.Acquisition


def _lag_from_pseudorange(pseudorange_m: float, sampling_freq: float) -> float:
    n_samples = int(round(float(sampling_freq) * 1e-3))
    delay_samples = (float(pseudorange_m) / C_LIGHT) * float(sampling_freq)
    return (n_samples - delay_samples) % n_samples


def test_acquisition_pseudorange_roundtrip_from_signal_sim():
    fs = 2.6e6
    sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-60, noise_seed=1)
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)
    sample_resolution_m = C_LIGHT / fs

    for raw_pr in [20_000_000.0, 20_123_456.0, 21_999_999.0]:
        iq = sim.generate_epoch([{
            "prn": 1,
            "code_phase": float(pseudorange_to_code_phase_chips(raw_pr)),
            "carrier_phase": 0.0,
            "doppler_hz": 0.0,
            "amplitude": 1.0,
            "nav_bit": 1,
        }])
        result = acq.acquire(iq[0::2].copy(), prn_list=[1])[0]

        reconstructed = acquisition_code_phase_to_pseudorange(
            result["code_phase"], fs, raw_pr)

        assert reconstructed == pytest.approx(raw_pr, abs=sample_resolution_m)


def test_ambiguity_resolution_uses_approximate_pseudorange():
    fs = 2.6e6
    base_pr = 20_000_000.0
    lag = _lag_from_pseudorange(base_pr, fs)

    resolved_base = acquisition_code_phase_to_pseudorange(lag, fs, base_pr)
    resolved_next_ms = acquisition_code_phase_to_pseudorange(
        lag, fs, base_pr + GPS_CA_PERIOD_M)

    assert resolved_base == pytest.approx(base_pr, abs=C_LIGHT / fs)
    assert resolved_next_ms == pytest.approx(base_pr + GPS_CA_PERIOD_M, abs=C_LIGHT / fs)


def test_compute_e2e_wls_weights_mean_normalized():
    w = compute_e2e_wls_weights([1e4, 2e4], [2.0, 8.0], [0.0, 0.5])
    assert w.shape == (2,)
    assert np.all(np.isfinite(w))
    assert np.mean(w) == pytest.approx(1.0)
    assert w[1] < w[0]


def test_lag_and_chips_roundtrip():
    fs = 2.6e6
    n_samples = int(round(fs * 1e-3))
    for lag in [0.0, 12.25, 100.5, float(n_samples) * 0.5]:
        chips = acquisition_lag_to_code_phase_chips(lag, fs)
        lag2 = code_phase_chips_to_acquisition_lag(chips, fs)
        circ = abs(lag2 - lag)
        circ = min(circ, abs(circ - n_samples))
        assert circ < 1e-9


def test_dll_refine_moves_toward_truth_on_synthetic():
    fs = 2.6e6
    pytest.importorskip(
        "gnss_gpu._gnss_gpu_tracking",
        reason="tracking CUDA bindings not available",
    )
    sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-55, noise_seed=2)
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)
    raw_pr = 20_000_100.0
    iq = sim.generate_epoch([{
        "prn": 1,
        "code_phase": float(pseudorange_to_code_phase_chips(raw_pr)),
        "carrier_phase": 0.0,
        "doppler_hz": 0.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }])
    signal_i = iq[0::2].copy()
    r = acq.acquire(signal_i, prn_list=[1])[0]
    assert r["acquired"]
    lag_ref = refine_acquisition_code_lag_dll(
        signal_i, 1, r["code_phase"], r["doppler_hz"], fs, intermediate_freq=0.0,
        n_iter=20, dll_gain=0.25)
    pr_coarse = acquisition_code_phase_to_pseudorange(
        r["code_phase"], fs, raw_pr)
    pr_ref = acquisition_code_phase_to_pseudorange(lag_ref, fs, raw_pr)
    err_coarse = abs(pr_coarse - raw_pr)
    err_ref = abs(pr_ref - raw_pr)
    assert err_ref <= err_coarse + C_LIGHT / fs


def test_acquisition_returns_fractional_code_phase_after_interpolation():
    fs = 2.6e6
    raw_pr = 20_000_030.0
    expected_lag = _lag_from_pseudorange(raw_pr, fs)

    sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-60, noise_seed=1)
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)

    iq = sim.generate_epoch([{
        "prn": 1,
        "code_phase": float(pseudorange_to_code_phase_chips(raw_pr)),
        "carrier_phase": 0.0,
        "doppler_hz": 0.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }])
    result = acq.acquire(iq[0::2].copy(), prn_list=[1])[0]

    assert abs(result["code_phase"] - round(result["code_phase"])) > 0.05
    assert result["code_phase"] == pytest.approx(expected_lag, abs=0.15)
