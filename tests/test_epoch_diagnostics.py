from dataclasses import dataclass

from gnss_gpu.epoch_diagnostics import (
    _build_epoch_diagnostic_row,
    _epoch_dd_carrier_diagnostics,
    _epoch_dd_pseudorange_gate_diagnostics,
    _epoch_widelane_diagnostics,
    build_stop_segment_diagnostic_lines,
)


@dataclass
class _DummyDDResult:
    n_dd: int


@dataclass
class _DummyGateStats:
    n_kept_pairs: int
    n_pair_rejected: int
    rejected_by_epoch: bool
    pair_threshold: float
    metric_median: float
    metric_max: float


@dataclass
class _DummyAnchorAttempt:
    update: object | None = None
    stats: dict | None = None
    used: bool = False
    propagated_rows: int = 0


@dataclass
class _DummyFallbackAttempt:
    afv: dict | None = None
    tracked_stats: dict | None = None
    sigma_cycles: float | None = None
    sigma_scale: float = 1.0
    used: bool = False
    attempted_tracked: bool = False
    used_tracked: bool = False
    replaced_weak_dd: bool = False


def test_epoch_dd_carrier_diagnostics_reports_gate_anchor_and_fallback_fields():
    row = _epoch_dd_carrier_diagnostics(
        _DummyDDResult(4),
        _DummyGateStats(
            n_kept_pairs=3,
            n_pair_rejected=1,
            rejected_by_epoch=False,
            pair_threshold=0.2,
            metric_median=0.05,
            metric_max=0.1,
        ),
        _DummyAnchorAttempt(
            update={"sat_ecef": []},
            stats={
                "n_sat": 4,
                "residual_median_m": 0.2,
                "residual_max_m": 0.4,
                "continuity_median_m": 0.1,
                "continuity_max_m": 0.3,
                "max_age_s": 0.8,
            },
            used=True,
            propagated_rows=2,
        ),
        _DummyFallbackAttempt(
            afv={"n_sat": 5},
            tracked_stats={
                "n_sat": 6,
                "n_tracked_consistent_sat": 4,
                "continuity_median_m": 0.1,
                "continuity_max_m": 0.2,
                "stable_epochs_median": 3.0,
            },
            sigma_cycles=0.15,
            sigma_scale=1.5,
            used=True,
            attempted_tracked=True,
            used_tracked=True,
            replaced_weak_dd=True,
        ),
        dd_cp_input_pairs=4,
        dd_cp_gate_scale=0.8,
        dd_cp_raw_abs_afv_median_cycles=0.06,
        dd_cp_raw_abs_afv_max_cycles=0.11,
        dd_cp_sigma_support_scale=1.2,
        dd_cp_sigma_afv_scale=1.1,
        dd_cp_sigma_ess_scale=1.3,
        dd_cp_sigma_scale=1.7,
        dd_cp_sigma_cycles=0.085,
        dd_cp_support_skip=False,
        carrier_anchor_sigma_m=0.25,
    )

    assert row["used_dd_carrier"] is True
    assert row["dd_cp_kept_pairs"] == 3
    assert row["dd_cp_pair_rejected"] == 1
    assert row["carrier_anchor_n_sat"] == 4
    assert row["carrier_anchor_sigma_m"] == 0.25
    assert row["used_dd_carrier_fallback_weak_dd"] is True
    assert row["dd_carrier_fallback_tracked_candidate_n_sat"] == 4
    assert row["dd_carrier_fallback_sigma_cycles"] == 0.15


def test_epoch_dd_carrier_diagnostics_uses_input_pairs_without_gate_stats():
    row = _epoch_dd_carrier_diagnostics(
        None,
        None,
        _DummyAnchorAttempt(),
        _DummyFallbackAttempt(),
        dd_cp_input_pairs=2,
        dd_cp_gate_scale=None,
        dd_cp_raw_abs_afv_median_cycles=None,
        dd_cp_raw_abs_afv_max_cycles=None,
        dd_cp_sigma_support_scale=1.0,
        dd_cp_sigma_afv_scale=1.0,
        dd_cp_sigma_ess_scale=1.0,
        dd_cp_sigma_scale=1.0,
        dd_cp_sigma_cycles=None,
        dd_cp_support_skip=True,
        carrier_anchor_sigma_m=0.25,
    )

    assert row["used_dd_carrier"] is False
    assert row["dd_cp_kept_pairs"] == 2
    assert row["dd_cp_pair_rejected"] == 0
    assert row["dd_cp_support_skip"] is True
    assert row["carrier_anchor_n_sat"] == 0
    assert row["dd_carrier_fallback_n_sat"] == 0


def test_epoch_widelane_diagnostics_formats_stats_and_gate_info():
    wl_stats = type(
        "Stats",
        (),
        {
            "reason": "ok",
            "ratio_min": 3.1,
            "ratio_median": 4.2,
            "residual_abs_median_cycles": 0.1,
            "residual_abs_max_cycles": 0.2,
            "std_median_cycles": 0.03,
        },
    )()
    row = _epoch_widelane_diagnostics(
        wl_stats,
        {
            "reason": "pair_gate",
            "pair_rejected": 2,
            "raw_abs_res_median_m": 1.0,
            "raw_abs_res_max_m": 3.0,
            "kept_abs_res_median_m": 0.8,
            "kept_abs_res_max_m": 1.2,
        },
        used_widelane_epoch=True,
        wl_input_pairs=6,
        wl_fixed_pairs=5,
        wl_fix_rate=0.83,
    )

    assert row["used_widelane"] is True
    assert row["widelane_input_pairs"] == 6
    assert row["widelane_gate_pair_rejected"] == 2
    assert row["widelane_raw_abs_res_median_m"] == 1.0


def test_epoch_dd_pseudorange_gate_diagnostics_reports_gate_stats():
    row = _epoch_dd_pseudorange_gate_diagnostics(
        _DummyGateStats(
            n_kept_pairs=4,
            n_pair_rejected=1,
            rejected_by_epoch=False,
            pair_threshold=10.0,
            metric_median=1.5,
            metric_max=3.0,
        ),
        dd_pr_input_pairs=5,
        dd_pr_gate_scale=0.9,
        dd_pr_raw_abs_res_median_m=2.0,
        dd_pr_raw_abs_res_max_m=4.0,
    )

    assert row["dd_pr_input_pairs"] == 5
    assert row["dd_pr_kept_pairs"] == 4
    assert row["dd_pr_pair_rejected"] == 1
    assert row["dd_pr_gate_pair_threshold_m"] == 10.0


def test_build_epoch_diagnostic_row_composes_sections_in_expected_order():
    row = _build_epoch_diagnostic_row(
        run_name="Odaiba",
        tow=100.0,
        aligned_epoch_index=2,
        store_epoch_index=3,
        gt_index=4,
        n_measurements=12,
        used_imu=True,
        used_tdcp=False,
        used_tdcp_pu_epoch=True,
        tdcp_pu_rms=0.7,
        tdcp_pu_spp_diff_mps=1.2,
        tdcp_pu_gate_reason="ok",
        imu_stop_detected=False,
        used_imu_tight_epoch=True,
        rbpf_velocity_kf=True,
        doppler_update_epoch={"sat_ecef": []},
        doppler_kf_gate_reason=None,
        dd_pr_result=_DummyDDResult(3),
        dd_pr_sigma_epoch=0.75,
        dd_pr_gate_stats=None,
        dd_pr_input_pairs=3,
        dd_pr_gate_scale=1.0,
        dd_pr_raw_abs_res_median_m=None,
        dd_pr_raw_abs_res_max_m=None,
        gate_ess_ratio=0.5,
        gate_spread_m=1.5,
        wl_stats=None,
        wl_gate_info={"reason": None, "pair_rejected": 0},
        used_widelane_epoch=False,
        wl_input_pairs=0,
        wl_fixed_pairs=0,
        wl_fix_rate=None,
        dd_carrier_result=_DummyDDResult(4),
        dd_gate_stats=None,
        anchor_attempt=_DummyAnchorAttempt(),
        fallback_attempt=_DummyFallbackAttempt(),
        dd_cp_input_pairs=4,
        dd_cp_gate_scale=1.0,
        dd_cp_raw_abs_afv_median_cycles=0.05,
        dd_cp_raw_abs_afv_max_cycles=0.1,
        dd_cp_sigma_support_scale=1.0,
        dd_cp_sigma_afv_scale=1.0,
        dd_cp_sigma_ess_scale=1.0,
        dd_cp_sigma_scale=1.0,
        dd_cp_sigma_cycles=0.05,
        dd_cp_support_skip=False,
        carrier_anchor_sigma_m=0.25,
    )

    assert row["run"] == "Odaiba"
    assert row["used_doppler_kf"] is True
    assert row["used_dd_pseudorange"] is True
    assert row["used_dd_carrier"] is True
    assert row["forward_error_2d"] is None
    keys = list(row)
    assert keys.index("used_widelane") < keys.index("dd_pr_sigma_m")
    assert keys.index("dd_pr_input_pairs") < keys.index("dd_cp_input_pairs")
    assert keys.index("used_dd_carrier") < keys.index("gate_ess_ratio")


def test_build_stop_segment_diagnostic_lines_ranks_segments_and_skips_short_runs():
    rows = [
        {
            "tow": 10.0,
            "imu_stop_detected": True,
            "smoothed_error_2d": 1.0,
            "forward_error_2d": 2.0,
            "stop_segment_radius_m": 0.5,
            "dd_cp_kept_pairs": 8,
            "used_dd_carrier_fallback": False,
        },
        {
            "tow": 10.1,
            "imu_stop_detected": True,
            "smoothed_error_2d": 3.0,
            "forward_error_2d": 4.0,
            "stop_segment_radius_m": 0.7,
            "dd_cp_kept_pairs": 6,
            "used_dd_carrier_fallback": True,
        },
        {"tow": 10.2, "imu_stop_detected": "False"},
        {
            "tow": 10.3,
            "imu_stop_detected": True,
            "smoothed_error_2d": 9.0,
            "forward_error_2d": 7.0,
            "dd_cp_kept_pairs": 2,
            "used_dd_carrier_fallback": False,
        },
        {
            "tow": 10.4,
            "imu_stop_detected": False,
        },
        {
            "tow": 10.5,
            "imu_stop_detected": True,
            "smoothed_error_2d": 5.0,
            "forward_error_2d": 1.0,
            "stop_segment_radius_m": 4.0,
            "dd_cp_kept_pairs": 3,
            "used_dd_carrier_fallback": "True",
        },
        {
            "tow": 10.6,
            "imu_stop_detected": True,
            "smoothed_error_2d": 7.0,
            "forward_error_2d": 2.0,
            "stop_segment_radius_m": 5.0,
            "dd_cp_kept_pairs": 5,
            "used_dd_carrier_fallback": "True",
        },
    ]

    lines = build_stop_segment_diagnostic_lines(rows, top_k=2, min_epochs=2)

    assert lines[0] == "  [stop_diag] worst 2 stop segments by smoothed p50:"
    assert "seg=1" in lines[1]
    assert "tow=10.5-10.6" in lines[1]
    assert "smth_p50=6.00m" in lines[1]
    assert "fwd_p50=1.50m" in lines[1]
    assert "radius=4.50m" in lines[1]
    assert "dd_cp_med=4.0" in lines[1]
    assert "fallback=2" in lines[1]
    assert "seg=0" in lines[2]
    assert "smth_p50=2.00m" in lines[2]
