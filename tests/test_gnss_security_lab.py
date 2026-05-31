import importlib.util
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp_gnss_security_lab.py"
_SPEC = importlib.util.spec_from_file_location("exp_gnss_security_lab", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_LAB = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LAB
_SPEC.loader.exec_module(_LAB)

_cpu_acquire = _LAB._cpu_acquire
_cpu_interference_summary = _LAB._cpu_interference_summary
_generate_ca_code = _LAB._generate_ca_code
_inject_narrowband_jammer = _LAB._inject_narrowband_jammer
_synth_gps_ca_signal = _LAB._synth_gps_ca_signal


def test_ca_code_generation_shape_and_values():
    code = _generate_ca_code(1)

    assert code.shape == (1023,)
    assert set(np.unique(code)).issubset({-1.0, 1.0})
    assert abs(int(np.sum(code == 1.0)) - int(np.sum(code == -1.0))) <= 1


def test_cpu_acquisition_finds_known_clean_signal():
    fs = 1.023e6
    rng = np.random.default_rng(123)
    signal = _synth_gps_ca_signal(
        prn=7,
        code_phase_samples=37,
        doppler_hz=1000.0,
        sampling_freq=fs,
        n_samples=1023,
        snr_db=24.0,
        rng=rng,
    )

    hits = _cpu_acquire(
        signal,
        sampling_freq=fs,
        prn_list=(7, 8),
        doppler_range=2000.0,
        doppler_step=500.0,
        threshold=2.0,
    )

    target = next(hit for hit in hits if hit.prn == 7)
    other = next(hit for hit in hits if hit.prn == 8)
    assert target.acquired
    assert abs(target.code_phase - 37) <= 1
    assert min(abs(target.doppler_hz - 1000.0), abs(target.doppler_hz + 1000.0)) <= 500.0
    assert target.snr > other.snr


def test_cpu_interference_detector_flags_narrowband_tone():
    fs = 1.023e6
    rng = np.random.default_rng(456)
    clean = _synth_gps_ca_signal(
        prn=7,
        code_phase_samples=0,
        doppler_hz=0.0,
        sampling_freq=fs,
        n_samples=4096,
        snr_db=18.0,
        rng=rng,
    )
    jammed = _inject_narrowband_jammer(
        clean,
        sampling_freq=fs,
        center_hz=120_000.0,
        jsr_db=24.0,
        phase_rad=0.0,
    )

    summary = _cpu_interference_summary(
        jammed,
        sampling_freq=fs,
        fft_size=512,
        hop_size=128,
        threshold_db=12.0,
    )

    assert summary["detected"]
    assert summary["kind"] in {"narrowband", "wideband"}
    assert abs(summary["center_freq_hz"] - 120_000.0) < 10_000.0
