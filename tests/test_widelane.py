from __future__ import annotations

import numpy as np

from gnss_gpu.widelane import (
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_WL,
    WidelaneAmbiguityResolver,
    WidelaneCandidate,
    WidelaneDDPseudorangeComputer,
    WidelaneFix,
    WidelaneObservation,
    _RawWidelaneObs,
    _dd_wl_float,
    _dd_wl_phase_m,
    compute_n_wl_float,
    detect_wl_candidates,
    fix_wl_ambiguities,
    wide_lane_phase_m,
    wl_fixed_pseudorange,
)


def _make_obs(range_m: float, n1: int, n2: int, code_noise_m: float = 0.0) -> WidelaneObservation:
    return WidelaneObservation(
        sat_id="G01",
        l1_carrier_cycles=range_m / LAMBDA_1 + float(n1),
        l2_carrier_cycles=range_m / LAMBDA_2 + float(n2),
        p1_m=range_m + code_noise_m,
        p2_m=range_m - code_noise_m,
    )


def _raw(range_m: float, n1: int, n2: int) -> _RawWidelaneObs:
    return _RawWidelaneObs(
        l1_carrier_cycles=range_m / LAMBDA_1 + float(n1),
        l2_carrier_cycles=range_m / LAMBDA_2 + float(n2),
        p1_m=range_m,
        p2_m=range_m,
    )


def test_widelane_float_and_fixed_pseudorange_recover_range() -> None:
    obs = _make_obs(22_000_000.0, n1=104, n2=91)

    n_float = compute_n_wl_float(
        obs.l1_carrier_cycles,
        obs.l2_carrier_cycles,
        obs.p1_m,
        obs.p2_m,
    )

    assert abs(n_float - 13.0) < 1.0e-6
    assert abs(wide_lane_phase_m(obs.l1_carrier_cycles, obs.l2_carrier_cycles) - (obs.p1_m + 13 * LAMBDA_WL)) < 1.0e-6
    assert abs(wl_fixed_pseudorange(obs, 13) - obs.p1_m) < 1.0e-6


def test_detect_and_lambda_fix_widelane_candidates() -> None:
    observations = [
        _make_obs(21_000_000.0, n1=12, n2=5),
        WidelaneObservation("G02", 22_000_000.0 / LAMBDA_1 - 3, 22_000_000.0 / LAMBDA_2 - 10, 22_000_000.0, 22_000_000.0),
    ]

    candidates = detect_wl_candidates(observations)
    fixes = fix_wl_ambiguities(
        [
            WidelaneCandidate(c.key, c.float_ambiguity_cycles + 0.02, variance_cycles2=0.01)
            for c in candidates
        ],
        ratio_threshold=3.0,
    )

    assert fixes["G01"].integer == 7
    assert fixes["G02"].integer == 7


def test_resolver_rejects_until_stable_then_fixes() -> None:
    resolver = WidelaneAmbiguityResolver(min_epochs=5, max_std_cycles=0.25, ratio_threshold=3.0)
    rng = np.random.default_rng(42)
    fix = None

    for value in 4.03 + rng.normal(0.0, 0.05, size=8):
        fix = resolver.update(("G", "G03", "G01"), float(value))

    assert fix is not None
    assert fix.integer == 4
    assert fix.ratio >= 3.0


def test_resolver_resets_on_cycle_slip() -> None:
    resolver = WidelaneAmbiguityResolver(
        min_epochs=3,
        max_std_cycles=0.25,
        ratio_threshold=3.0,
        cycle_slip_threshold_cycles=3.0,
    )
    key = ("G", "G04", "G01")
    fix = None
    for value in (2.01, 2.02, 2.00):
        fix = resolver.update(key, value)
    assert fix is not None

    assert resolver.update(key, 14.0) is None
    assert resolver.get(key) is None


def test_dd_widelane_float_and_phase_recover_dd_range() -> None:
    rover_k = _raw(20_000_010.0, n1=30, n2=22)
    rover_ref = _raw(20_500_000.0, n1=40, n2=31)
    base_k = _raw(19_900_000.0, n1=25, n2=21)
    base_ref = _raw(20_400_020.0, n1=33, n2=30)

    expected_dd_range = (20_000_010.0 - 20_500_000.0) - (19_900_000.0 - 20_400_020.0)
    expected_dd_integer = (30 - 22) - (40 - 31) - (25 - 21) + (33 - 30)

    assert abs(_dd_wl_float(rover_k, rover_ref, base_k, base_ref) - expected_dd_integer) < 1.0e-6
    fixed_dd_pr = _dd_wl_phase_m(rover_k, rover_ref, base_k, base_ref) - expected_dd_integer * LAMBDA_WL

    assert abs(fixed_dd_pr - expected_dd_range) < 1.0e-6


def test_widelane_dd_computer_min_ratio_filters_fixed_pairs() -> None:
    class _FakeResolver:
        def __init__(self, ratio: float) -> None:
            self.ratio = float(ratio)

        def update(self, key, _dd_float):
            return WidelaneFix(
                key=key,
                integer=0,
                ratio=self.ratio,
                residual_cycles=0.0,
                n_epochs=5,
                std_cycles=0.1,
            )

    class _Measurement:
        def __init__(self, prn: int, ecef: tuple[float, float, float]) -> None:
            self.system_id = 0
            self.prn = prn
            self.satellite_ecef = np.asarray(ecef, dtype=np.float64)
            self.elevation = 0.7
            self.weight = 1.0
            self.snr = 45.0

    comp = WidelaneDDPseudorangeComputer.__new__(WidelaneDDPseudorangeComputer)
    comp._allowed_systems = ("G",)
    comp._interpolate_base_epochs = False
    comp._min_fix_rate = 0.0
    comp._base_pos = np.zeros(3, dtype=np.float64)
    comp._base_by_tow = {
        100.0: {
            "G01": _raw(19_900_000.0, 25, 21),
            "G02": _raw(20_400_020.0, 33, 30),
        }
    }
    comp._rover_by_tow = {
        100.0: {
            "G01": _raw(20_000_010.0, 30, 22),
            "G02": _raw(20_500_000.0, 40, 31),
        }
    }
    comp._base_tow_keys = np.array([100.0], dtype=np.float64)
    measurements = [
        _Measurement(1, (20_200_000.0, 100.0, 0.0)),
        _Measurement(2, (21_200_000.0, 200.0, 0.0)),
    ]

    comp._resolver = _FakeResolver(ratio=4.0)
    result, stats = comp.compute_dd(
        100.0,
        measurements,
        min_common_sats=2,
        min_fix_rate=0.0,
        min_ratio=5.0,
    )
    assert result is None
    assert stats.n_ratio_rejected == 1
    assert stats.n_fixed_pairs == 0

    comp._resolver = _FakeResolver(ratio=6.0)
    result, stats = comp.compute_dd(
        100.0,
        measurements,
        min_common_sats=2,
        min_fix_rate=0.0,
        min_ratio=5.0,
    )
    assert result is not None
    assert result.n_dd == 1
    assert stats.n_ratio_rejected == 0
    assert stats.n_fixed_pairs == 1
