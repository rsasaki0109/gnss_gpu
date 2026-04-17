import numpy as np
import pytest
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from experiments.exp_pf_smoother_eval import (
    CarrierBiasState,
    _apply_dd_carrier_undiff_fallback,
    _expand_cli_preset_argv,
    _namespace_to_run_kwargs,
    _attempt_carrier_anchor_pseudorange_update,
    _attempt_dd_carrier_undiff_fallback,
    _build_carrier_anchor_pseudorange_update,
    _collect_hybrid_tracked_undiff_carrier_afv_inputs,
    _collect_tracked_undiff_carrier_afv_inputs,
    _prepare_dd_carrier_undiff_fallback,
    _propagate_carrier_bias_tracker_tdcp,
    _effective_dd_carrier_epoch_median_gate,
    _should_skip_low_support_dd_carrier,
    _should_replace_weak_dd_with_fallback,
    build_arg_parser,
    load_pf_smoother_dataset,
    run_pf_with_optional_smoother,
)


WAVELENGTH_M = 299792458.0 / 1575.42e6
REFERENCE_DATA_ROOT = Path("/tmp/UrbanNav-Tokyo")
LIBGNSSPP_BUILD = _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python" / "libgnsspp" / "_libgnsspp.cpython-312-x86_64-linux-gnu.so"


class _DummyPF:
    def __init__(self):
        self.last_update = None
        self.last_carrier_afv = None

    def update(self, sat_ecef, pseudoranges, *, weights, sigma_pr):
        self.last_update = {
            "sat_ecef": np.asarray(sat_ecef, dtype=np.float64),
            "pseudoranges": np.asarray(pseudoranges, dtype=np.float64),
            "weights": np.asarray(weights, dtype=np.float64),
            "sigma_pr": float(sigma_pr),
        }

    def update_carrier_afv(self, sat_ecef, carrier_phase_cycles, *, weights, wavelength, sigma_cycles):
        self.last_carrier_afv = {
            "sat_ecef": np.asarray(sat_ecef, dtype=np.float64),
            "carrier_phase_cycles": np.asarray(carrier_phase_cycles, dtype=np.float64),
            "weights": np.asarray(weights, dtype=np.float64),
            "wavelength": float(wavelength),
            "sigma_cycles": float(sigma_cycles),
        }


class _DummyDDResult:
    def __init__(self, n_dd: int):
        self.n_dd = int(n_dd)


def _make_tracker_and_rows():
    tow_prev = 100.0
    dt = 0.1
    tow_cur = tow_prev + dt
    rx_prev = np.array([10.0, -5.0, 2.0], dtype=np.float64)
    rx_cur = np.array([10.35, -4.9, 2.08], dtype=np.float64)
    cb_prev = 120.0
    cb_cur = 120.18
    bias_cycles_base = 1000.0

    sat_prev_list = [
        np.array([20_200_000.0, 1_400_000.0, 300_000.0], dtype=np.float64),
        np.array([1_100_000.0, 21_100_000.0, 400_000.0], dtype=np.float64),
        np.array([-300_000.0, 900_000.0, 20_800_000.0], dtype=np.float64),
        np.array([-20_500_000.0, -800_000.0, 600_000.0], dtype=np.float64),
    ]
    sat_vel_list = [
        np.array([250.0, -120.0, 40.0], dtype=np.float64),
        np.array([-180.0, 260.0, 35.0], dtype=np.float64),
        np.array([90.0, -140.0, 220.0], dtype=np.float64),
        np.array([210.0, 160.0, -75.0], dtype=np.float64),
    ]

    tracker = {}
    carrier_rows = {}
    expected_pseudoranges_cur = {}
    receiver_state_cur = np.array([rx_cur[0], rx_cur[1], rx_cur[2], cb_cur], dtype=np.float64)

    for i, (sat_prev, sat_vel) in enumerate(zip(sat_prev_list, sat_vel_list), start=1):
        sat_cur = sat_prev + sat_vel * dt
        bias_cycles = bias_cycles_base + float(i)

        pseudo_prev = float(np.linalg.norm(sat_prev - rx_prev) + cb_prev)
        pseudo_cur = float(np.linalg.norm(sat_cur - rx_cur) + cb_cur)
        carrier_prev = pseudo_prev / WAVELENGTH_M + bias_cycles
        carrier_cur = pseudo_cur / WAVELENGTH_M + bias_cycles

        key = (0, i)
        tracker[key] = CarrierBiasState(
            bias_cycles=bias_cycles,
            last_tow=tow_prev,
            last_expected_cycles=pseudo_prev / WAVELENGTH_M,
            last_carrier_phase_cycles=carrier_prev,
            last_pseudorange_m=pseudo_prev,
            last_receiver_state=np.array([rx_prev[0], rx_prev[1], rx_prev[2], cb_prev], dtype=np.float64),
            last_sat_ecef=sat_prev.copy(),
            last_sat_velocity=sat_vel.copy(),
            last_clock_drift=0.0,
            stable_epochs=3,
        )
        carrier_rows[key] = {
            "system_id": 0,
            "prn": i,
            "sat_ecef": sat_cur.copy(),
            "sat_velocity": sat_vel.copy(),
            "clock_drift": 0.0,
            "carrier_phase_cycles": carrier_cur,
            "weight": 1.0,
            "wavelength_m": WAVELENGTH_M,
        }
        expected_pseudoranges_cur[key] = pseudo_cur

    return tracker, carrier_rows, receiver_state_cur, tow_cur, expected_pseudoranges_cur


def test_build_carrier_anchor_update_uses_tdcp_predicted_pseudorange():
    tracker, carrier_rows, receiver_state_cur, tow_cur, expected_pseudoranges_cur = _make_tracker_and_rows()

    update, stats, accepted_rows = _build_carrier_anchor_pseudorange_update(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        max_age_s=1.0,
        max_residual_m=1.5,
        max_continuity_residual_m=1.5,
        min_stable_epochs=1,
        min_sats=4,
    )

    assert update is not None
    assert stats["n_sat"] == 4
    assert len(accepted_rows) == 4
    expected = np.array([expected_pseudoranges_cur[key] for key in sorted(expected_pseudoranges_cur)], dtype=np.float64)
    np.testing.assert_allclose(update["pseudoranges"], expected, atol=0.2)


def test_propagate_carrier_bias_tracker_tdcp_advances_state():
    tracker, carrier_rows, receiver_state_cur, tow_cur, expected_pseudoranges_cur = _make_tracker_and_rows()

    n_propagated = _propagate_carrier_bias_tracker_tdcp(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        blend_alpha=0.5,
        reanchor_jump_cycles=4.0,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
    )

    assert n_propagated == 4
    for key in sorted(expected_pseudoranges_cur):
        state = tracker[key]
        assert state.last_tow == tow_cur
        assert state.stable_epochs == 4
        np.testing.assert_allclose(state.last_pseudorange_m, expected_pseudoranges_cur[key], atol=0.2)


def test_collect_tracked_undiff_carrier_afv_inputs_keeps_tracker_consistent_rows():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected_pseudoranges_cur = _make_tracker_and_rows()

    fallback, stats = _collect_tracked_undiff_carrier_afv_inputs(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
        min_stable_epochs=2,
        min_sats=4,
    )

    assert fallback is not None
    assert fallback["n_sat"] == 4
    assert stats["n_sat"] == 4
    assert stats["continuity_median_m"] is not None
    assert stats["stable_epochs_median"] == 3.0


def test_collect_tracked_undiff_carrier_afv_inputs_skips_nonfinite_rows():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected_pseudoranges_cur = _make_tracker_and_rows()
    carrier_rows[(0, 4)]["carrier_phase_cycles"] = float("nan")

    fallback, stats = _collect_tracked_undiff_carrier_afv_inputs(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
        min_stable_epochs=2,
        min_sats=3,
    )

    assert fallback is not None
    assert fallback["n_sat"] == 3
    assert stats["n_sat"] == 3
    assert np.isfinite(stats["continuity_median_m"])
    assert np.isfinite(np.asarray(fallback["weights"], dtype=np.float64)).all()


def test_collect_tracked_undiff_carrier_afv_inputs_reports_stats_below_min_sats():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected_pseudoranges_cur = _make_tracker_and_rows()

    fallback, stats = _collect_tracked_undiff_carrier_afv_inputs(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
        min_stable_epochs=2,
        min_sats=5,
    )

    assert fallback is None
    assert stats["n_sat"] == 4
    assert np.isfinite(stats["continuity_median_m"])
    assert np.isfinite(stats["continuity_max_m"])


def test_collect_hybrid_tracked_undiff_carrier_afv_inputs_keeps_untracked_rows():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected_pseudoranges_cur = _make_tracker_and_rows()
    carrier_rows[(9, 99)] = {
        "system_id": 9,
        "prn": 99,
        "sat_ecef": np.array([21_500_000.0, 2_500_000.0, -900_000.0], dtype=np.float64),
        "sat_velocity": np.array([50.0, -40.0, 20.0], dtype=np.float64),
        "clock_drift": 0.0,
        "carrier_phase_cycles": 1.25e8,
        "weight": 0.75,
        "wavelength_m": WAVELENGTH_M,
    }

    fallback, stats = _collect_hybrid_tracked_undiff_carrier_afv_inputs(
        tracker,
        carrier_rows,
        receiver_state_cur,
        tow_cur,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
        min_stable_epochs=2,
        min_sats=5,
    )

    assert fallback is not None
    assert fallback["n_sat"] == 5
    assert stats["n_sat"] == 5
    assert stats["n_tracked_consistent_sat"] == 4
    assert np.isfinite(stats["continuity_median_m"])


def test_attempt_carrier_anchor_pseudorange_update_applies_pf_update():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected = _make_tracker_and_rows()
    pf = _DummyPF()

    attempt = _attempt_carrier_anchor_pseudorange_update(
        pf,
        tracker,
        carrier_rows,
        receiver_state_cur,
        prev_pf_state=tracker[(0, 1)].last_receiver_state.copy(),
        velocity=np.array([0.2, 0.1, 0.05], dtype=np.float64),
        dt=0.1,
        tow=tow_cur,
        enabled=True,
        dd_carrier_result=None,
        seed_dd_min_pairs=3,
        sigma_m=0.25,
        max_age_s=1.0,
        max_residual_m=1.5,
        max_continuity_residual_m=1.5,
        min_stable_epochs=1,
        min_sats=4,
    )

    assert attempt.used is True
    assert attempt.update is not None
    assert pf.last_update is not None
    assert pf.last_update["sigma_pr"] == 0.25
    assert pf.last_update["pseudoranges"].shape[0] == 4


def test_attempt_dd_carrier_undiff_fallback_uses_tracked_hybrid_rows():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected = _make_tracker_and_rows()
    pf = _DummyPF()

    attempt = _attempt_dd_carrier_undiff_fallback(
        pf,
        measurements=[],
        sat_ecef=np.zeros((0, 3), dtype=np.float64),
        pseudoranges=np.zeros(0, dtype=np.float64),
        spp_pos_check=receiver_state_cur[:3],
        tracker=tracker,
        carrier_rows=carrier_rows,
        carrier_state=receiver_state_cur,
        tow=tow_cur,
        enabled=True,
        mupf_enabled=False,
        dd_carrier_result=None,
        used_carrier_anchor=False,
        snr_min=25.0,
        elev_min=0.15,
        fallback_sigma_cycles=0.10,
        fallback_min_sats=4,
        prefer_tracked=True,
        tracked_min_stable_epochs=2,
        tracked_min_sats=2,
        tracked_continuity_good_m=None,
        tracked_continuity_bad_m=None,
        tracked_sigma_min_scale=1.0,
        tracked_sigma_max_scale=1.0,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
    )

    assert attempt.attempted_tracked is True
    assert attempt.used is True
    assert attempt.used_tracked is True
    assert attempt.afv is not None
    assert pf.last_carrier_afv is not None
    assert pf.last_carrier_afv["sigma_cycles"] == 0.10
    assert pf.last_carrier_afv["carrier_phase_cycles"].shape[0] == 4


def test_should_replace_weak_dd_with_fallback_requires_matching_guards():
    dd_result = _DummyDDResult(4)
    dd_pr_result = _DummyDDResult(2)

    assert _should_replace_weak_dd_with_fallback(
        dd_result,
        dd_pr_result,
        raw_afv_median_cycles=0.22,
        ess_ratio=None,
        weak_dd_max_pairs=4,
        weak_dd_max_ess_ratio=None,
        weak_dd_min_raw_afv_median_cycles=0.15,
        weak_dd_require_no_dd_pr=True,
    ) is True
    assert _should_replace_weak_dd_with_fallback(
        dd_result,
        dd_pr_result,
        raw_afv_median_cycles=0.10,
        ess_ratio=None,
        weak_dd_max_pairs=4,
        weak_dd_max_ess_ratio=None,
        weak_dd_min_raw_afv_median_cycles=0.15,
        weak_dd_require_no_dd_pr=True,
    ) is False
    assert _should_replace_weak_dd_with_fallback(
        dd_result,
        _DummyDDResult(5),
        raw_afv_median_cycles=0.22,
        ess_ratio=None,
        weak_dd_max_pairs=4,
        weak_dd_max_ess_ratio=None,
        weak_dd_min_raw_afv_median_cycles=0.15,
        weak_dd_require_no_dd_pr=True,
    ) is False


def test_should_replace_weak_dd_with_fallback_can_gate_on_ess_ratio():
    dd_result = _DummyDDResult(11)

    assert _should_replace_weak_dd_with_fallback(
        dd_result,
        None,
        raw_afv_median_cycles=0.22,
        ess_ratio=0.001,
        weak_dd_max_pairs=None,
        weak_dd_max_ess_ratio=0.01,
        weak_dd_min_raw_afv_median_cycles=0.15,
        weak_dd_require_no_dd_pr=True,
    ) is True
    assert _should_replace_weak_dd_with_fallback(
        dd_result,
        None,
        raw_afv_median_cycles=0.22,
        ess_ratio=0.02,
        weak_dd_max_pairs=None,
        weak_dd_max_ess_ratio=0.01,
        weak_dd_min_raw_afv_median_cycles=0.15,
        weak_dd_require_no_dd_pr=True,
    ) is False


def test_should_skip_low_support_dd_carrier_can_gate_on_spread():
    dd_result = _DummyDDResult(11)

    assert _should_skip_low_support_dd_carrier(
        dd_result,
        None,
        ess_ratio=0.001,
        spread_m=1.8,
        raw_afv_median_cycles=0.22,
        low_support_ess_ratio=0.01,
        low_support_max_pairs=None,
        low_support_max_spread_m=2.0,
        low_support_min_raw_afv_median_cycles=0.20,
        low_support_require_no_dd_pr=True,
    ) is True
    assert _should_skip_low_support_dd_carrier(
        dd_result,
        None,
        ess_ratio=0.001,
        spread_m=2.3,
        raw_afv_median_cycles=0.22,
        low_support_ess_ratio=0.01,
        low_support_max_pairs=None,
        low_support_max_spread_m=2.0,
        low_support_min_raw_afv_median_cycles=0.20,
        low_support_require_no_dd_pr=True,
    ) is False
    assert _should_skip_low_support_dd_carrier(
        dd_result,
        _DummyDDResult(5),
        ess_ratio=0.001,
        spread_m=1.8,
        raw_afv_median_cycles=0.22,
        low_support_ess_ratio=0.01,
        low_support_max_pairs=None,
        low_support_max_spread_m=2.0,
        low_support_min_raw_afv_median_cycles=0.20,
        low_support_require_no_dd_pr=True,
    ) is False


def test_effective_dd_carrier_epoch_median_gate_tightens_contextually():
    assert _effective_dd_carrier_epoch_median_gate(
        None,
        base_epoch_median_cycles=None,
        ess_ratio=0.001,
        spread_m=1.8,
        low_ess_epoch_median_cycles=0.18,
        low_ess_max_ratio=0.01,
        low_ess_max_spread_m=2.0,
        low_ess_require_no_dd_pr=True,
    ) == 0.18
    assert _effective_dd_carrier_epoch_median_gate(
        None,
        base_epoch_median_cycles=0.20,
        ess_ratio=0.001,
        spread_m=1.8,
        low_ess_epoch_median_cycles=0.18,
        low_ess_max_ratio=0.01,
        low_ess_max_spread_m=2.0,
        low_ess_require_no_dd_pr=True,
    ) == 0.18
    assert _effective_dd_carrier_epoch_median_gate(
        _DummyDDResult(5),
        base_epoch_median_cycles=0.20,
        ess_ratio=0.001,
        spread_m=1.8,
        low_ess_epoch_median_cycles=0.18,
        low_ess_max_ratio=0.01,
        low_ess_max_spread_m=2.0,
        low_ess_require_no_dd_pr=True,
    ) == 0.20
    assert _effective_dd_carrier_epoch_median_gate(
        None,
        base_epoch_median_cycles=0.20,
        ess_ratio=0.02,
        spread_m=1.8,
        low_ess_epoch_median_cycles=0.18,
        low_ess_max_ratio=0.01,
        low_ess_max_spread_m=2.0,
        low_ess_require_no_dd_pr=True,
    ) == 0.20


def test_prepare_dd_carrier_undiff_fallback_marks_weak_dd_replacement():
    tracker, carrier_rows, receiver_state_cur, tow_cur, _expected = _make_tracker_and_rows()

    attempt = _prepare_dd_carrier_undiff_fallback(
        measurements=[],
        sat_ecef=np.zeros((0, 3), dtype=np.float64),
        pseudoranges=np.zeros(0, dtype=np.float64),
        spp_pos_check=receiver_state_cur[:3],
        tracker=tracker,
        carrier_rows=carrier_rows,
        carrier_state=receiver_state_cur,
        tow=tow_cur,
        enabled=True,
        mupf_enabled=False,
        dd_carrier_result=_DummyDDResult(4),
        used_carrier_anchor=False,
        snr_min=25.0,
        elev_min=0.15,
        fallback_sigma_cycles=0.10,
        fallback_min_sats=4,
        prefer_tracked=True,
        tracked_min_stable_epochs=2,
        tracked_min_sats=2,
        tracked_continuity_good_m=None,
        tracked_continuity_bad_m=None,
        tracked_sigma_min_scale=1.0,
        tracked_sigma_max_scale=1.0,
        max_age_s=1.0,
        max_continuity_residual_m=1.5,
        allow_weak_dd=True,
        weak_dd_max_pairs=4,
    )

    assert attempt.replaced_weak_dd is True
    assert attempt.afv is not None
    assert attempt.sigma_cycles == 0.10

    pf = _DummyPF()
    attempt = _apply_dd_carrier_undiff_fallback(pf, attempt)
    assert attempt.used is True
    assert pf.last_carrier_afv is not None


def test_expand_cli_preset_argv_inlines_odaiba_reference_flags():
    expanded = _expand_cli_preset_argv([
        "--preset", "odaiba_reference",
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--max-epochs", "10",
    ])

    assert "--preset" not in expanded
    assert "--runs" in expanded
    assert expanded[expanded.index("--runs") + 1] == "Odaiba"
    assert "--dd-pseudorange" in expanded
    assert expanded[expanded.index("--imu-stop-sigma-pos") + 1] == "0.1"
    assert "--mupf-dd-fallback-undiff" in expanded
    assert expanded[expanded.index("--mupf-dd-gate-adaptive-floor-cycles") + 1] == "0.18"
    assert expanded[-2:] == ["--max-epochs", "10"]


def test_odaiba_reference_preset_keeps_late_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--carrier-anchor-max-residual-m", "1.0",
        "--max-epochs", "7",
    ]))

    assert args.predict_guide == "imu"
    assert args.imu_stop_sigma_pos == 0.1
    assert args.smoother is True
    assert args.dd_pseudorange is True
    assert args.mupf_dd is True
    assert args.carrier_anchor is True
    assert args.carrier_anchor_max_residual_m == 1.0
    assert args.max_epochs == 7

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["predict_guide"] == "imu"
    assert run_kwargs["imu_stop_sigma_pos"] == 0.1
    assert run_kwargs["mupf_dd_fallback_undiff"] is True
    assert run_kwargs["carrier_anchor_max_residual_m"] == 1.0


def test_odaiba_presets_keep_targeted_dd_gate_floors():
    parser = build_arg_parser()

    reference_args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
    ]))
    guarded_args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference_guarded",
    ]))
    stop_detect_args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_stop_detect",
    ]))

    assert reference_args.mupf_dd_gate_adaptive_floor_cycles == 0.18
    assert guarded_args.mupf_dd_gate_adaptive_floor_cycles == 0.18
    assert stop_detect_args.mupf_dd_gate_adaptive_floor_cycles == 0.25


def test_parser_maps_weak_dd_fallback_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--mupf-dd-fallback-weak-dd-max-pairs", "4",
        "--mupf-dd-fallback-weak-dd-max-ess-ratio", "0.01",
        "--mupf-dd-fallback-weak-dd-min-raw-afv-median-cycles", "0.15",
        "--mupf-dd-fallback-weak-dd-require-no-dd-pr",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["mupf_dd_fallback_weak_dd_max_pairs"] == 4
    assert run_kwargs["mupf_dd_fallback_weak_dd_max_ess_ratio"] == 0.01
    assert run_kwargs["mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles"] == 0.15
    assert run_kwargs["mupf_dd_fallback_weak_dd_require_no_dd_pr"] is True


def test_parser_maps_low_support_skip_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--mupf-dd-skip-low-support-ess-ratio", "0.01",
        "--mupf-dd-skip-low-support-max-spread-m", "2.0",
        "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.20",
        "--mupf-dd-skip-low-support-require-no-dd-pr",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["mupf_dd_skip_low_support_ess_ratio"] == 0.01
    assert run_kwargs["mupf_dd_skip_low_support_max_spread_m"] == 2.0
    assert run_kwargs["mupf_dd_skip_low_support_min_raw_afv_median_cycles"] == 0.20
    assert run_kwargs["mupf_dd_skip_low_support_require_no_dd_pr"] is True


def test_parser_maps_low_ess_dd_gate_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--mupf-dd-gate-low-ess-epoch-median-cycles", "0.18",
        "--mupf-dd-gate-low-ess-max-ratio", "0.01",
        "--mupf-dd-gate-low-ess-max-spread-m", "2.0",
        "--mupf-dd-gate-low-ess-require-no-dd-pr",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["mupf_dd_gate_low_ess_epoch_median_cycles"] == 0.18
    assert run_kwargs["mupf_dd_gate_low_ess_max_ratio"] == 0.01
    assert run_kwargs["mupf_dd_gate_low_ess_max_spread_m"] == 2.0
    assert run_kwargs["mupf_dd_gate_low_ess_require_no_dd_pr"] is True


def test_parser_maps_dd_huber_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--mupf-dd-huber-k", "1.5",
        "--dd-pseudorange-huber-k", "2.0",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["mupf_dd_huber_k"] == 1.5
    assert run_kwargs["dd_pseudorange_huber_k"] == 2.0


@pytest.mark.skipif(
    not (REFERENCE_DATA_ROOT / "Odaiba").exists(),
    reason="UrbanNav Odaiba reference data not available",
)
def test_odaiba_reference_preset_smoke_regression():
    if not LIBGNSSPP_BUILD.exists():
        pytest.skip("libgnsspp build binding not available")
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", str(REFERENCE_DATA_ROOT),
        "--preset", "odaiba_reference",
        "--n-particles", "5000",
        "--max-epochs", "10",
    ]))
    run_dir = REFERENCE_DATA_ROOT / "Odaiba"
    pos_sigma = args.position_update_sigma if args.position_update_sigma >= 0 else None
    dataset = load_pf_smoother_dataset(run_dir, rover_source=args.urban_rover)

    out = run_pf_with_optional_smoother(
        run_dir,
        "Odaiba",
        dataset=dataset,
        **_namespace_to_run_kwargs(
            args,
            position_update_sigma=pos_sigma,
            use_smoother=args.smoother,
        ),
    )

    assert out["forward_metrics"] is not None
    assert out["smoothed_metrics"] is not None
    # P50 bounds: tightened from 0-20m to 0-5m for 10-epoch smoke
    assert 0.0 < out["forward_metrics"]["p50"] < 5.0
    assert 0.0 < out["smoothed_metrics"]["p50"] < 5.0
    # DD carrier must fire on most of the 10 epochs
    assert int(out["n_dd_used"]) >= 5
    # DD pseudorange must fire at least once
    assert int(out["n_dd_pr_used"]) >= 1


@pytest.mark.skipif(
    not (REFERENCE_DATA_ROOT / "Odaiba").exists(),
    reason="UrbanNav Odaiba reference data not available",
)
def test_odaiba_reference_preset_50ep_regression():
    """Medium regression: 50 epochs to catch counter and metric drift."""
    if not LIBGNSSPP_BUILD.exists():
        pytest.skip("libgnsspp build binding not available")
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", str(REFERENCE_DATA_ROOT),
        "--preset", "odaiba_reference",
        "--n-particles", "5000",
        "--max-epochs", "50",
    ]))
    run_dir = REFERENCE_DATA_ROOT / "Odaiba"
    pos_sigma = args.position_update_sigma if args.position_update_sigma >= 0 else None
    dataset = load_pf_smoother_dataset(run_dir, rover_source=args.urban_rover)

    out = run_pf_with_optional_smoother(
        run_dir,
        "Odaiba",
        dataset=dataset,
        **_namespace_to_run_kwargs(
            args,
            position_update_sigma=pos_sigma,
            use_smoother=args.smoother,
        ),
    )

    # Metrics: tighter bounds for 50-epoch run
    assert 0.0 < out["forward_metrics"]["p50"] < 5.0
    assert 0.0 < out["smoothed_metrics"]["p50"] < 5.0
    assert 0.0 < out["forward_metrics"]["rms_2d"] < 5.0
    assert 0.0 < out["smoothed_metrics"]["rms_2d"] < 5.0
    # DD carrier must fire on all 50 epochs (no base-data gap in first 50)
    assert int(out["n_dd_used"]) >= 40
    assert int(out["n_dd_skip"]) <= 10
    # DD pseudorange must fire on a few epochs
    assert int(out["n_dd_pr_used"]) >= 2


def test_expand_cli_preset_argv_inlines_odaiba_stop_detect_flags():
    expanded = _expand_cli_preset_argv([
        "--preset", "odaiba_stop_detect",
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--max-epochs", "10",
    ])

    assert "--preset" not in expanded
    assert "--runs" in expanded
    assert expanded[expanded.index("--runs") + 1] == "Odaiba"
    assert "--imu-stop-sigma-pos" in expanded
    assert expanded[expanded.index("--imu-stop-sigma-pos") + 1] == "0.1"
    assert "--dd-pseudorange" in expanded
    assert "--mupf-dd-fallback-undiff" in expanded
    assert expanded[-2:] == ["--max-epochs", "10"]


def test_expand_cli_preset_argv_inlines_odaiba_reference_guarded_flags():
    expanded = _expand_cli_preset_argv([
        "--preset", "odaiba_reference_guarded",
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--max-epochs", "10",
    ])

    assert "--preset" not in expanded
    assert "--smoother-tail-guard-ess-max-ratio" in expanded
    assert expanded[expanded.index("--smoother-tail-guard-ess-max-ratio") + 1] == "0.001"
    assert "--smoother-tail-guard-min-shift-m" in expanded
    assert expanded[expanded.index("--smoother-tail-guard-min-shift-m") + 1] == "4.0"
    assert expanded[-2:] == ["--max-epochs", "10"]


def test_expand_cli_preset_argv_inlines_odaiba_best_accuracy_flags():
    expanded = _expand_cli_preset_argv([
        "--preset", "odaiba_best_accuracy",
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--max-epochs", "10",
    ])

    assert "--preset" not in expanded
    assert expanded[expanded.index("--n-particles") + 1] == "200000"
    assert expanded[expanded.index("--imu-stop-sigma-pos") + 1] == "0.1"
    assert expanded[expanded.index("--carrier-anchor-sigma-m") + 1] == "0.15"
    assert expanded[expanded.index("--mupf-dd-gate-adaptive-floor-cycles") + 1] == "0.18"
    assert "--smoother-tail-guard-ess-max-ratio" in expanded
    assert expanded[-2:] == ["--max-epochs", "10"]


def test_odaiba_stop_detect_preset_parses_imu_stop_sigma_pos():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_stop_detect",
    ]))

    assert args.predict_guide == "imu"
    assert args.imu_stop_sigma_pos == 0.1
    assert args.smoother is True
    assert args.dd_pseudorange is True
    assert args.mupf_dd is True

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["imu_stop_sigma_pos"] == 0.1


@pytest.mark.skipif(
    not (REFERENCE_DATA_ROOT / "Odaiba").exists(),
    reason="UrbanNav Odaiba reference data not available",
)
def test_odaiba_stop_detect_preset_smoke_regression():
    if not LIBGNSSPP_BUILD.exists():
        pytest.skip("libgnsspp build binding not available")
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", str(REFERENCE_DATA_ROOT),
        "--preset", "odaiba_stop_detect",
        "--n-particles", "5000",
        "--max-epochs", "10",
    ]))
    run_dir = REFERENCE_DATA_ROOT / "Odaiba"
    pos_sigma = args.position_update_sigma if args.position_update_sigma >= 0 else None
    dataset = load_pf_smoother_dataset(run_dir, rover_source=args.urban_rover)

    out = run_pf_with_optional_smoother(
        run_dir,
        "Odaiba",
        dataset=dataset,
        **_namespace_to_run_kwargs(
            args,
            position_update_sigma=pos_sigma,
            use_smoother=args.smoother,
        ),
    )

    assert out["forward_metrics"] is not None
    assert out["smoothed_metrics"] is not None
    assert 0.0 < out["forward_metrics"]["p50"] < 5.0
    assert 0.0 < out["smoothed_metrics"]["p50"] < 5.0
    assert int(out["n_dd_used"]) >= 5
    assert int(out["n_dd_pr_used"]) >= 1
    assert args.imu_stop_sigma_pos == 0.1
