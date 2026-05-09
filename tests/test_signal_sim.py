"""Tests for GPU signal simulator."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "gnss_gpu._gnss_gpu_signal_sim",
    reason="CUDA signal simulator bindings not available",
)
pytest.importorskip(
    "gnss_gpu._gnss_gpu_acq",
    reason="CUDA acquisition bindings not available",
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


def test_single_channel_acquisition_roundtrip():
    """Generate a signal and verify it can be acquired."""
    sim = SignalSimulator(noise_seed=1)
    channels = [{
        "prn": 1,
        "code_phase": 0.0,
        "carrier_phase": 0.0,
        "doppler_hz": 1000.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }]
    iq = sim.generate_epoch(channels)

    # Acquisition expects real signal; use I channel
    signal = iq[0::2].copy()

    acq = Acquisition(sampling_freq=sim.sampling_freq,
                      intermediate_freq=sim.intermediate_freq)
    results = acq.acquire(signal, prn_list=[1])

    assert len(results) == 1
    r = results[0]
    assert r["acquired"]
    assert abs(r["doppler_hz"] - 1000.0) <= 500.0
    n_samples = int(sim.sampling_freq * 1e-3)
    circular_error = min(abs(r["code_phase"]), abs(n_samples - r["code_phase"]))
    assert circular_error <= 2.0  # code_phase=0 -> acq returns ~0 modulo 1 ms


def test_multi_satellite():
    """Generate 3 satellites and verify all are acquired."""
    sim = SignalSimulator(noise_floor_db=-40, noise_seed=1)
    channels = [
        {"prn": 1,  "code_phase": 0.0,  "carrier_phase": 0.0,
         "doppler_hz": -1200.0, "amplitude": 1.0, "nav_bit": 1},
        {"prn": 5,  "code_phase": 50.0, "carrier_phase": 0.0,
         "doppler_hz": 300.0,   "amplitude": 1.0, "nav_bit": 1},
        {"prn": 10, "code_phase": 75.0, "carrier_phase": 0.0,
         "doppler_hz": 1800.0,  "amplitude": 1.0, "nav_bit": 1},
    ]
    # 1ms (one C/A code period for FFT-based acquisition)
    n_samples = int(sim.sampling_freq * 1e-3)
    iq = sim.generate_epoch(channels, n_samples=n_samples)
    signal = iq[0::2].copy()

    acq = Acquisition(sampling_freq=sim.sampling_freq,
                      intermediate_freq=sim.intermediate_freq)
    results = acq.acquire(signal, prn_list=list(range(1, 33)))
    acquired_prns = {r["prn"] for r in results if r["acquired"]}

    assert {1, 5, 10}.issubset(acquired_prns)


def test_output_formats(tmp_path: Path):
    """Verify binary output file sizes."""
    sim = SignalSimulator(noise_seed=1)
    iq = sim.generate_test_signal(prn=3, code_phase=10, doppler=200,
                                  duration_s=2e-3)
    n_iq = len(iq)

    p = tmp_path / "sig_int8.bin"
    sim.write_bin(iq, p, fmt="int8")
    assert p.stat().st_size == n_iq * 1

    p = tmp_path / "sig_int16.bin"
    sim.write_bin(iq, p, fmt="int16")
    assert p.stat().st_size == n_iq * 2

    p = tmp_path / "sig_float32.bin"
    sim.write_bin(iq, p, fmt="float32")
    assert p.stat().st_size == n_iq * 4


def test_noise_floor():
    """Verify noise-only signal RMS matches expected level."""
    noise_floor_db = -20.0
    sim = SignalSimulator(noise_floor_db=noise_floor_db, noise_seed=1)
    iq = sim.generate_epoch([], n_samples=8192)

    rms = float(np.sqrt(np.mean(np.asarray(iq, dtype=np.float64) ** 2)))
    expected = 10 ** (noise_floor_db / 20.0)
    assert rms == pytest.approx(expected, rel=0.35)
