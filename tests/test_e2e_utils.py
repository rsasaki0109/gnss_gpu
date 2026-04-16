"""Tests for E2E acquisition/pseudorange helpers."""

import csv

import numpy as np
import pytest


from gnss_gpu.e2e_helpers import (
    C_LIGHT,
    GPS_CA_PERIOD_M,
    acquisition_code_phase_to_pseudorange,
    acquisition_lag_to_code_phase_chips,
    code_phase_chips_to_acquisition_lag,
    compute_e2e_wls_weights,
    dump_e2e_diagnostics_csv,
    pseudorange_to_code_phase_chips,
    refine_acquisition_code_lag_dll,
    refine_acquisition_code_lags_diagnostic_batch,
    refine_acquisition_code_lags_dll_batch,
)

try:
    from gnss_gpu.signal_sim import SignalSimulator
    from gnss_gpu.acquisition import Acquisition
    import gnss_gpu._gnss_gpu_acq  # noqa: F401
    import gnss_gpu._gnss_gpu_signal_sim  # noqa: F401
    _HAS_CUDA_SIGNAL = True
except ImportError:
    SignalSimulator = None
    Acquisition = None
    _HAS_CUDA_SIGNAL = False

requires_cuda_signal = pytest.mark.skipif(
    not _HAS_CUDA_SIGNAL,
    reason="CUDA signal simulator/acquisition bindings not available",
)


def _lag_from_pseudorange(pseudorange_m: float, sampling_freq: float) -> float:
    n_samples = int(round(float(sampling_freq) * 1e-3))
    delay_samples = (float(pseudorange_m) / C_LIGHT) * float(sampling_freq)
    return (n_samples - delay_samples) % n_samples


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
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


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
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


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
def test_gain_schedule_constant_matches_existing_behavior():
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

    explicit = refine_acquisition_code_lags_dll_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25,
        gain_schedule="constant")
    default = refine_acquisition_code_lags_dll_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25)

    assert np.array_equal(explicit, default)


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
def test_gain_schedule_cn0_weighted_converges_on_synthetic():
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

    constant = refine_acquisition_code_lags_dll_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25,
        gain_schedule="constant")
    weighted = refine_acquisition_code_lags_dll_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25,
        gain_schedule="cn0_weighted")

    tol = 2 * C_LIGHT / fs
    pr_constant = acquisition_code_phase_to_pseudorange(constant[0], fs, raw_pr)
    pr_weighted = acquisition_code_phase_to_pseudorange(weighted[0], fs, raw_pr)
    assert abs(pr_constant - raw_pr) < tol
    assert abs(pr_weighted - raw_pr) < tol


def test_gain_schedule_invalid_raises():
    with pytest.raises(ValueError, match="unknown gain_schedule"):
        refine_acquisition_code_lags_dll_batch(
            np.zeros(0, dtype=np.float32),
            [],
            [],
            [],
            2.6e6,
            gain_schedule="nope",
        )


def test_diagnostic_includes_gain_schedule_key():
    signal_i = np.zeros(0, dtype=np.float32)
    diag = refine_acquisition_code_lags_diagnostic_batch(
        signal_i, [], [], [], 2.6e6)
    assert diag["gain_schedule"] == "constant"

    diag = refine_acquisition_code_lags_diagnostic_batch(
        signal_i, [], [], [], 2.6e6, gain_schedule="cn0_weighted")
    assert diag["gain_schedule"] == "cn0_weighted"


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
def test_diagnostic_batch_shapes_and_keys():
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

    diag = refine_acquisition_code_lags_diagnostic_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25)
    lag_ref = refine_acquisition_code_lags_dll_batch(
        signal_i, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25)

    array_keys = [
        "lag_samples",
        "code_phase_chips",
        "carrier_phase_cycles",
        "code_freq_hz",
        "carrier_freq_hz",
        "E_I",
        "E_Q",
        "P_I",
        "P_Q",
        "L_I",
        "L_Q",
        "prompt_power",
        "dll_abs",
        "cn0_est_db",
        "prn",
    ]
    assert set(array_keys + ["n_iter_used", "gain_schedule"]) == set(diag)
    for key in array_keys:
        assert diag[key].shape == (1,)
    assert diag["prn"].dtype == np.int32
    assert diag["lag_samples"][0] == pytest.approx(lag_ref[0], abs=1e-9)
    assert np.isfinite(diag["cn0_est_db"][0])
    assert not np.isnan(diag["cn0_est_db"][0])
    assert diag["n_iter_used"] == 20
    assert diag["gain_schedule"] == "constant"


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
def test_refine_accepts_multi_ms_buffer():
    fs = 2.6e6
    pytest.importorskip(
        "gnss_gpu._gnss_gpu_tracking",
        reason="tracking CUDA bindings not available",
    )
    samples_per_ms = int(round(fs * 1e-3))
    sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-55, noise_seed=2)
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)
    raw_pr = 20_000_100.0
    iq_1ms = sim.generate_epoch([{
        "prn": 1,
        "code_phase": float(pseudorange_to_code_phase_chips(raw_pr)),
        "carrier_phase": 0.0,
        "doppler_hz": 0.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }])
    iq_5ms = np.tile(iq_1ms, 5)
    signal_i_5ms = iq_5ms[0::2].copy()
    r = acq.acquire(signal_i_5ms[:samples_per_ms], prn_list=[1])[0]
    assert r["acquired"]

    lag_ref = refine_acquisition_code_lags_dll_batch(
        signal_i_5ms, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25)
    pr_coarse = acquisition_code_phase_to_pseudorange(
        r["code_phase"], fs, raw_pr)
    pr_ref = acquisition_code_phase_to_pseudorange(lag_ref[0], fs, raw_pr)
    err_coarse = abs(pr_coarse - raw_pr)
    err_ref = abs(pr_ref - raw_pr)

    assert lag_ref.shape == (1,)
    assert np.isfinite(lag_ref[0])
    assert not np.isnan(lag_ref[0])
    assert err_ref <= err_coarse + C_LIGHT / fs


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
def test_diagnostic_accepts_multi_ms_buffer():
    fs = 2.6e6
    pytest.importorskip(
        "gnss_gpu._gnss_gpu_tracking",
        reason="tracking CUDA bindings not available",
    )
    samples_per_ms = int(round(fs * 1e-3))
    sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-55, noise_seed=2)
    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)
    raw_pr = 20_000_100.0
    iq_1ms = sim.generate_epoch([{
        "prn": 1,
        "code_phase": float(pseudorange_to_code_phase_chips(raw_pr)),
        "carrier_phase": 0.0,
        "doppler_hz": 0.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }])
    iq_5ms = np.tile(iq_1ms, 5)
    signal_i_5ms = iq_5ms[0::2].copy()
    r = acq.acquire(signal_i_5ms[:samples_per_ms], prn_list=[1])[0]
    assert r["acquired"]

    diag = refine_acquisition_code_lags_diagnostic_batch(
        signal_i_5ms, [1], [r["code_phase"]], [r["doppler_hz"]], fs,
        intermediate_freq=0.0, n_iter=20, dll_gain=0.25)
    expected_keys = {
        "lag_samples",
        "code_phase_chips",
        "carrier_phase_cycles",
        "code_freq_hz",
        "carrier_freq_hz",
        "E_I",
        "E_Q",
        "P_I",
        "P_Q",
        "L_I",
        "L_Q",
        "prompt_power",
        "dll_abs",
        "cn0_est_db",
        "prn",
        "n_iter_used",
        "gain_schedule",
    }

    assert set(diag) == expected_keys
    assert diag["lag_samples"].shape == (1,)
    assert np.isfinite(diag["cn0_est_db"][0])


@pytest.mark.gpu
@pytest.mark.cuda
@requires_cuda_signal
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


def test_dump_e2e_diagnostics_csv_roundtrip(tmp_path):
    columns = [
        "prn",
        "lag_samples",
        "code_phase_chips",
        "carrier_phase_cycles",
        "code_freq_hz",
        "carrier_freq_hz",
        "E_I",
        "E_Q",
        "P_I",
        "P_Q",
        "L_I",
        "L_Q",
        "prompt_power",
        "dll_abs",
        "cn0_est_db",
    ]
    diagnostics = {
        "prn": np.array([3, 22], dtype=np.int32),
        "lag_samples": np.array([10.25, 20.5], dtype=np.float64),
        "code_phase_chips": np.array([101.0, 202.5], dtype=np.float64),
        "carrier_phase_cycles": np.array([0.125, 0.875], dtype=np.float64),
        "code_freq_hz": np.array([1.023e6, 1.023001e6], dtype=np.float64),
        "carrier_freq_hz": np.array([100.0, -250.0], dtype=np.float64),
        "E_I": np.array([1.25, 2.5], dtype=np.float64),
        "E_Q": np.array([-0.5, 0.75], dtype=np.float64),
        "P_I": np.array([10.0, 20.0], dtype=np.float64),
        "P_Q": np.array([0.25, -0.125], dtype=np.float64),
        "L_I": np.array([1.0, 2.0], dtype=np.float64),
        "L_Q": np.array([0.5, -0.75], dtype=np.float64),
        "prompt_power": np.array([100.0625, 400.015625], dtype=np.float64),
        "dll_abs": np.array([0.1, 0.2], dtype=np.float64),
        "cn0_est_db": np.array([2.0412, 14.0824], dtype=np.float64),
        "n_iter_used": 7,
    }

    path = tmp_path / "diag.csv"
    dump_e2e_diagnostics_csv(path, diagnostics)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == columns
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert int(row["prn"]) == int(diagnostics["prn"][i])
        for col in columns[1:]:
            assert float(row[col]) == pytest.approx(
                float(diagnostics[col][i]), rel=1e-8, abs=1e-12)
