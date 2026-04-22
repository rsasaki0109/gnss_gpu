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

import experiments.exp_pf_smoother_eval as pf_eval
from experiments.exp_pf_smoother_eval import (
    _expand_cli_preset_argv,
    _namespace_to_run_config,
    _namespace_to_run_kwargs,
    build_arg_parser,
    load_pf_smoother_dataset,
    run_pf_with_optional_smoother,
)
from gnss_gpu.pf_smoother_config import PfSmootherConfig, validate_pf_smoother_config

REFERENCE_DATA_ROOT = Path("/tmp/UrbanNav-Tokyo")
LIBGNSSPP_BUILD = _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python" / "libgnsspp" / "_libgnsspp.cpython-312-x86_64-linux-gnu.so"

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


def test_namespace_to_run_config_matches_legacy_kwargs():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--max-epochs", "7",
    ]))
    config = _namespace_to_run_config(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    legacy_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )

    assert config.n_particles == args.n_particles
    assert config.predict_guide == "imu"
    assert config.max_epochs == 7
    assert config.to_kwargs() == legacy_kwargs
    assert config.with_overrides(max_epochs=3).max_epochs == 3
    parts = config.parts()
    assert parts.run_selection.max_epochs == 7
    assert parts.particle_filter.n_particles == args.n_particles
    assert parts.motion.predict_guide == "imu"
    assert parts.observations.dd_pseudorange.enabled is True
    assert parts.observations.dd_carrier.enabled is True
    assert parts.observations.carrier_rescue.fallback_undiff is True


def test_run_config_grouped_validation_rejects_incompatible_modes():
    config = PfSmootherConfig(
        n_particles=100,
        sigma_pos=1.2,
        sigma_pr=3.0,
        position_update_sigma=1.9,
        predict_guide="imu",
        use_smoother=True,
        dd_pseudorange=True,
        use_gmm=True,
    )

    with pytest.raises(ValueError, match="dd_pseudorange"):
        validate_pf_smoother_config(config)


def test_run_pf_with_optional_smoother_accepts_config_and_overrides(monkeypatch):
    captured = {}

    def fake_impl(run_dir, run_name, **kwargs):
        captured["run_dir"] = run_dir
        captured["run_name"] = run_name
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(pf_eval, "_run_pf_with_optional_smoother_impl", fake_impl)
    config = PfSmootherConfig(
        n_particles=100,
        sigma_pos=1.2,
        sigma_pr=3.0,
        position_update_sigma=1.9,
        predict_guide="imu",
        use_smoother=True,
        max_epochs=5,
    )

    out = run_pf_with_optional_smoother(
        Path("/tmp/example"),
        "Odaiba",
        dataset={"epochs": []},
        config=config,
        max_epochs=2,
    )

    assert out == {"ok": True}
    assert captured["run_dir"] == Path("/tmp/example")
    assert captured["run_name"] == "Odaiba"
    assert captured["dataset"] == {"epochs": []}
    assert captured["run_config"].max_epochs == 2
    assert captured["n_particles"] == 100
    assert captured["max_epochs"] == 2


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

def test_parser_maps_per_particle_nlos_gate_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--per-particle-nlos-gate",
        "--per-particle-nlos-dd-pr-threshold-m", "5.0",
        "--per-particle-nlos-dd-carrier-threshold-cycles", "0.3",
        "--per-particle-nlos-undiff-pr-threshold-m", "25.0",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["per_particle_nlos_gate"] is True
    assert run_kwargs["per_particle_nlos_dd_pr_threshold_m"] == 5.0
    assert run_kwargs["per_particle_nlos_dd_carrier_threshold_cycles"] == 0.3
    assert run_kwargs["per_particle_nlos_undiff_pr_threshold_m"] == 25.0

def test_parser_maps_per_particle_huber_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--per-particle-huber",
        "--per-particle-huber-dd-pr-k", "1.0",
        "--per-particle-huber-dd-carrier-k", "2.0",
        "--per-particle-huber-undiff-pr-k", "3.0",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["per_particle_huber"] is True
    assert run_kwargs["per_particle_huber_dd_pr_k"] == 1.0
    assert run_kwargs["per_particle_huber_dd_carrier_k"] == 2.0
    assert run_kwargs["per_particle_huber_undiff_pr_k"] == 3.0

def test_parser_maps_widelane_gate_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--widelane",
        "--widelane-gate-min-fixed-pairs", "5",
        "--widelane-gate-min-fix-rate", "0.4",
        "--widelane-gate-min-spread-m", "2.5",
        "--widelane-gate-max-epoch-median-residual-m", "8.0",
        "--widelane-gate-max-pair-residual-m", "15.0",
        "--smoother-skip-widelane-dd-pseudorange",
        "--smoother-widelane-forward-guard",
        "--smoother-widelane-forward-guard-min-shift-m", "1.0",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["widelane"] is True
    assert run_kwargs["widelane_gate_min_fixed_pairs"] == 5
    assert run_kwargs["widelane_gate_min_fix_rate"] == 0.4
    assert run_kwargs["widelane_gate_min_spread_m"] == 2.5
    assert run_kwargs["widelane_gate_max_epoch_median_residual_m"] == 8.0
    assert run_kwargs["widelane_gate_max_pair_residual_m"] == 15.0
    assert run_kwargs["smoother_skip_widelane_dd_pseudorange"] is True
    assert run_kwargs["smoother_widelane_forward_guard"] is True
    assert run_kwargs["smoother_widelane_forward_guard_min_shift_m"] == 1.0

def test_parser_maps_doppler_per_particle_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--doppler-per-particle",
        "--doppler-sigma-mps", "0.75",
        "--doppler-velocity-update-gain", "0.5",
        "--doppler-max-velocity-update-mps", "4.0",
        "--doppler-min-sats", "5",
        "--pf-sigma-vel", "0.2",
        "--pf-velocity-guide-alpha", "0.75",
        "--pf-init-spread-vel", "1.5",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["doppler_per_particle"] is True
    assert run_kwargs["doppler_sigma_mps"] == 0.75
    assert run_kwargs["doppler_velocity_update_gain"] == 0.5
    assert run_kwargs["doppler_max_velocity_update_mps"] == 4.0
    assert run_kwargs["doppler_min_sats"] == 5
    assert run_kwargs["rbpf_velocity_kf"] is False
    assert run_kwargs["pf_sigma_vel"] == 0.2
    assert run_kwargs["pf_velocity_guide_alpha"] == 0.75
    assert run_kwargs["pf_init_spread_vel"] == 1.5

def test_parser_maps_rbpf_velocity_kf_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--rbpf-velocity-kf",
        "--rbpf-velocity-init-sigma", "3.0",
        "--rbpf-velocity-process-noise", "0.25",
        "--rbpf-doppler-sigma", "0.8",
        "--rbpf-velocity-kf-gate-min-dd-pairs", "15",
        "--rbpf-velocity-kf-gate-min-ess-ratio", "0.02",
        "--rbpf-velocity-kf-gate-max-spread-m", "3.5",
        "--doppler-min-sats", "6",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["rbpf_velocity_kf"] is True
    assert run_kwargs["rbpf_velocity_init_sigma"] == 3.0
    assert run_kwargs["rbpf_velocity_process_noise"] == 0.25
    assert run_kwargs["rbpf_doppler_sigma"] == 0.8
    assert run_kwargs["rbpf_velocity_kf_gate_min_dd_pairs"] == 15
    assert run_kwargs["rbpf_velocity_kf_gate_min_ess_ratio"] == 0.02
    assert run_kwargs["rbpf_velocity_kf_gate_max_spread_m"] == 3.5
    assert run_kwargs["doppler_per_particle"] is False
    assert run_kwargs["doppler_min_sats"] == 6

def test_parser_maps_tdcp_position_update_gate_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--tdcp-position-update",
        "--tdcp-pu-sigma", "1.0",
        "--tdcp-pu-rms-max", "2.0",
        "--tdcp-pu-spp-max-diff-mps", "4.0",
        "--tdcp-pu-gate-dd-carrier-min-pairs", "8",
        "--tdcp-pu-gate-dd-carrier-max-pairs", "4",
        "--tdcp-pu-gate-dd-pseudorange-max-pairs", "3",
        "--tdcp-pu-gate-min-spread-m", "2.5",
        "--tdcp-pu-gate-max-spread-m", "5.0",
        "--tdcp-pu-gate-min-ess-ratio", "0.02",
        "--tdcp-pu-gate-max-ess-ratio", "0.01",
        "--tdcp-pu-gate-dd-pr-max-raw-median-m", "4.0",
        "--tdcp-pu-gate-dd-cp-max-raw-afv-median-cycles", "0.25",
        "--tdcp-pu-gate-logic", "all",
        "--tdcp-pu-gate-stop-mode", "moving",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["tdcp_position_update"] is True
    assert run_kwargs["tdcp_pu_sigma"] == 1.0
    assert run_kwargs["tdcp_pu_rms_max"] == 2.0
    assert run_kwargs["tdcp_pu_spp_max_diff_mps"] == 4.0
    assert run_kwargs["tdcp_pu_gate_dd_carrier_min_pairs"] == 8
    assert run_kwargs["tdcp_pu_gate_dd_carrier_max_pairs"] == 4
    assert run_kwargs["tdcp_pu_gate_dd_pseudorange_max_pairs"] == 3
    assert run_kwargs["tdcp_pu_gate_min_spread_m"] == 2.5
    assert run_kwargs["tdcp_pu_gate_max_spread_m"] == 5.0
    assert run_kwargs["tdcp_pu_gate_min_ess_ratio"] == 0.02
    assert run_kwargs["tdcp_pu_gate_max_ess_ratio"] == 0.01
    assert run_kwargs["tdcp_pu_gate_dd_pr_max_raw_median_m"] == 4.0
    assert run_kwargs["tdcp_pu_gate_dd_cp_max_raw_afv_median_cycles"] == 0.25
    assert run_kwargs["tdcp_pu_gate_logic"] == "all"
    assert run_kwargs["tdcp_pu_gate_stop_mode"] == "moving"

def test_parser_maps_stop_segment_constant_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_best_accuracy",
        "--seed", "7",
        "--smoother-position-update-sigma", "2.2",
        "--stop-segment-constant",
        "--stop-segment-min-epochs", "8",
        "--stop-segment-source", "combined_auto_tail",
        "--stop-segment-max-radius-m", "3.5",
        "--stop-segment-blend", "0.75",
        "--stop-segment-density-neighbors", "123",
        "--stop-segment-static-gnss",
        "--stop-segment-static-min-observations", "55",
        "--stop-segment-static-prior-sigma-m", "30.0",
        "--stop-segment-static-pr-sigma-m", "6.5",
        "--stop-segment-static-dd-pr-sigma-m", "3.5",
        "--stop-segment-static-dd-cp-sigma-cycles", "0.4",
        "--stop-segment-static-max-update-m", "15.0",
        "--stop-segment-static-blend", "0.8",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["stop_segment_constant"] is True
    assert run_kwargs["seed"] == 7
    assert run_kwargs["smoother_position_update_sigma"] == 2.2
    assert run_kwargs["stop_segment_min_epochs"] == 8
    assert run_kwargs["stop_segment_source"] == "combined_auto_tail"
    assert run_kwargs["stop_segment_max_radius_m"] == 3.5
    assert run_kwargs["stop_segment_blend"] == 0.75
    assert run_kwargs["stop_segment_density_neighbors"] == 123
    assert run_kwargs["stop_segment_static_gnss"] is True
    assert run_kwargs["stop_segment_static_min_observations"] == 55
    assert run_kwargs["stop_segment_static_prior_sigma_m"] == 30.0
    assert run_kwargs["stop_segment_static_pr_sigma_m"] == 6.5
    assert run_kwargs["stop_segment_static_dd_pr_sigma_m"] == 3.5
    assert run_kwargs["stop_segment_static_dd_cp_sigma_cycles"] == 0.4
    assert run_kwargs["stop_segment_static_max_update_m"] == 15.0
    assert run_kwargs["stop_segment_static_blend"] == 0.8

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

def test_parser_maps_local_fgo_flags_and_auto_requests_diagnostics():
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv([
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--preset", "odaiba_reference",
        "--fgo-local-window", "auto",
        "--fgo-local-window-min-epochs", "123",
        "--fgo-local-dd-huber-k", "1.25",
        "--fgo-local-prior-sigma-m", "0.4",
        "--fgo-local-dd-sigma-cycles", "0.3",
        "--fgo-local-pr-sigma-m", "4.0",
        "--fgo-local-lambda",
        "--fgo-local-lambda-ratio-threshold", "3.5",
        "--fgo-local-lambda-sigma-cycles", "0.04",
        "--fgo-local-lambda-min-epochs", "45",
        "--fgo-local-motion-source", "prefer_tdcp",
        "--fgo-local-tdcp-rms-max-m", "2.5",
        "--fgo-local-tdcp-spp-max-diff-mps", "4.5",
        "--fgo-local-two-step",
        "--fgo-local-stage1-prior-sigma-m", "0.8",
        "--fgo-local-stage1-motion-sigma-m", "0.6",
        "--fgo-local-stage1-pr-sigma-m", "7.0",
    ]))

    run_kwargs = _namespace_to_run_kwargs(
        args,
        position_update_sigma=args.position_update_sigma,
        use_smoother=args.smoother,
    )
    assert run_kwargs["fgo_local_window"] == "auto"
    assert run_kwargs["fgo_local_window_min_epochs"] == 123
    assert run_kwargs["fgo_local_dd_huber_k"] == 1.25
    assert run_kwargs["fgo_local_prior_sigma_m"] == 0.4
    assert run_kwargs["fgo_local_dd_sigma_cycles"] == 0.3
    assert run_kwargs["fgo_local_pr_sigma_m"] == 4.0
    assert run_kwargs["fgo_local_lambda"] is True
    assert run_kwargs["fgo_local_lambda_ratio_threshold"] == 3.5
    assert run_kwargs["fgo_local_lambda_sigma_cycles"] == 0.04
    assert run_kwargs["fgo_local_lambda_min_epochs"] == 45
    assert run_kwargs["fgo_local_motion_source"] == "prefer_tdcp"
    assert run_kwargs["fgo_local_tdcp_rms_max_m"] == 2.5
    assert run_kwargs["fgo_local_tdcp_spp_max_diff_mps"] == 4.5
    assert run_kwargs["fgo_local_two_step"] is True
    assert run_kwargs["fgo_local_stage1_prior_sigma_m"] == 0.8
    assert run_kwargs["fgo_local_stage1_motion_sigma_m"] == 0.6
    assert run_kwargs["fgo_local_stage1_pr_sigma_m"] == 7.0
    assert run_kwargs["collect_epoch_diagnostics"] is True

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
    assert "--stop-segment-constant" in expanded
    assert expanded[expanded.index("--stop-segment-source") + 1] == "smoothed_auto_tail"
    assert expanded[expanded.index("--stop-segment-density-neighbors") + 1] == "200"
    assert expanded[expanded.index("--smoother-tail-guard-ess-max-ratio") + 1] == "0.0001"
    assert expanded[expanded.index("--smoother-tail-guard-min-shift-m") + 1] == "9.0"
    assert expanded[expanded.index("--smoother-tail-guard-expand-epochs") + 1] == "10"
    assert (
        expanded[
            expanded.index("--smoother-tail-guard-expand-dd-pseudorange-max-pairs") + 1
        ]
        == "0"
    )
    assert "--no-doppler-per-particle" in expanded
    assert expanded[-2:] == ["--max-epochs", "10"]

def test_expand_cli_preset_argv_inlines_odaiba_rbpf_velocity_flags():
    expanded = _expand_cli_preset_argv([
        "--preset", "odaiba_rbpf_velocity",
        "--data-root", "/tmp/UrbanNav-Tokyo",
        "--max-epochs", "10",
    ])

    assert "--preset" not in expanded
    assert expanded[expanded.index("--n-particles") + 1] == "200000"
    assert expanded[expanded.index("--carrier-anchor-sigma-m") + 1] == "0.15"
    assert "--no-doppler-per-particle" in expanded
    assert "--doppler-per-particle" not in expanded
    assert "--rbpf-velocity-kf" in expanded
    assert expanded[expanded.index("--rbpf-velocity-init-sigma") + 1] == "2.0"
    assert expanded[expanded.index("--rbpf-velocity-process-noise") + 1] == "1.0"
    assert expanded[expanded.index("--rbpf-doppler-sigma") + 1] == "0.5"
    assert expanded[expanded.index("--doppler-min-sats") + 1] == "4"
    assert expanded[expanded.index("--pf-sigma-vel") + 1] == "0.0"
    assert expanded[expanded.index("--pf-velocity-guide-alpha") + 1] == "1.0"
    assert expanded[expanded.index("--pf-init-spread-vel") + 1] == "0.0"
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
