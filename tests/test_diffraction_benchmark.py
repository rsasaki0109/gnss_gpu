"""Tests for the diffraction-model quantitative benchmark."""

import numpy as np
import pytest

from gnss_gpu.validation.diffraction_benchmark import (
    CA_CHIP_M,
    DiffractionCandidate,
    benchmark_models,
    candidates_from_paths,
    code_multipath_bias_chips,
    predict_bias_samples_m,
    triangle_acf,
)


def test_triangle_acf():
    np.testing.assert_allclose(triangle_acf([0.0, 0.5, 1.0, 1.5, -0.25]),
                               [1.0, 0.5, 0.0, 0.0, 0.75])


def test_no_multipath_zero_bias():
    assert code_multipath_bias_chips(0.0, 0.3, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_in_phase_and_anti_phase_opposite_sign():
    ip = code_multipath_bias_chips(0.5, 0.3, 0.0)
    ap = code_multipath_bias_chips(0.5, 0.3, np.pi)
    assert ip > 0.0  # in-phase delayed replica -> positive code bias
    assert ap < 0.0  # anti-phase -> negative
    assert abs(ip) > 1e-3 and abs(ap) > 1e-3


def test_bias_grows_with_amplitude():
    biases = [code_multipath_bias_chips(a, 0.3, 0.0) for a in (0.2, 0.4, 0.6, 0.8)]
    assert all(b1 < b2 for b1, b2 in zip(biases, biases[1:]))


def test_zero_delay_zero_bias():
    # A replica with no excess delay cannot bias the code loop.
    assert code_multipath_bias_chips(0.7, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_predict_samples_shape_and_scale():
    cands = [DiffractionCandidate(0.5, 30.0), DiffractionCandidate(0.3, 80.0)]
    s = predict_bias_samples_m(cands, n_phase=16)
    assert s.size == 32
    # Biases are bounded by the multipath error envelope (tens of meters).
    assert np.all(np.abs(s) < 100.0)


def test_predict_skips_zero_amplitude():
    cands = [DiffractionCandidate(0.0, 30.0)]
    assert predict_bias_samples_m(cands, n_phase=8).size == 0


def test_benchmark_identical_wins_and_w1_equals_shift():
    cands = [DiffractionCandidate(0.5, 30.0), DiffractionCandidate(0.4, 50.0)]
    s = predict_bias_samples_m(cands, n_phase=24)
    res = benchmark_models(s, {"identical": s, "shifted": s + 5.0})
    assert res["best_wasserstein"] == "identical"
    assert res["best_ks"] == "identical"
    assert res["models"]["identical"]["wasserstein"] == pytest.approx(0.0, abs=1e-9)
    # Wasserstein-1 of a pure translation equals the shift magnitude.
    assert res["models"]["shifted"]["wasserstein"] == pytest.approx(5.0, abs=1e-9)


def test_benchmark_prefers_closer_distribution():
    cands = [DiffractionCandidate(0.6, 40.0)]
    real = predict_bias_samples_m(cands, n_phase=32)
    near = real + 0.5
    far = real + 10.0
    res = benchmark_models(real, {"near": near, "far": far})
    assert res["best_wasserstein"] == "near"
    assert res["models"]["near"]["wasserstein"] < res["models"]["far"]["wasserstein"]


def test_candidates_from_paths():
    class _P:
        def __init__(self, amp, delay):
            self.amplitude = amp
            self.excess_delay = delay

    paths = [_P(0.4, 25.0), _P(0.2, 60.0)]
    cands = candidates_from_paths(paths, direct_amplitude=1.0)
    assert len(cands) == 2
    assert cands[0].amplitude_ratio == pytest.approx(0.4)
    assert cands[1].excess_delay_m == pytest.approx(60.0)
