from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser
from gnss_gpu.pf_smoother_results import build_pf_smoother_result_row


def test_build_pf_smoother_result_row_maps_cli_args_output_counters_and_metrics():
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.2)
    args = parser.parse_args(
        [
            "--data-root",
            "/tmp/UrbanNav-Tokyo",
            "--predict-guide",
            "imu",
            "--n-particles",
            "123",
            "--tdcp-position-update",
            "--fgo-local-window",
            "auto",
            "--fgo-local-two-step",
        ]
    )
    out = {
        "n_tdcp_pu_used": 2,
        "n_doppler_kf_gate_skip": 3,
        "n_widelane_forward_guard_applied": 4,
        "smoothed_metrics_before_fgo": {"p50": 1.5, "rms_2d": 2.5},
        "fgo_local_applied": True,
        "fgo_local_info": {
            "window": (10, 20),
            "solve_window": (9, 21),
            "motion_tdcp_edges": 8,
            "motion_tdcp_selected_edges": 5,
            "initial_error": 100.0,
            "final_error": 25.0,
            "lambda": {
                "n_fixed": 7,
                "n_fixed_observations": 11,
            },
        },
    }

    row = build_pf_smoother_result_row(
        args=args,
        out=out,
        run_name="Odaiba",
        label="with_smoother",
        use_smoother=True,
        position_update_sigma=1.9,
        forward_metrics={"p50": 3.0, "p95": 9.0, "rms_2d": 4.0},
        smoothed_metrics={"p50": 2.0, "p95": 8.0, "rms_2d": 3.0},
        n_epochs=42,
        ms_per_epoch=0.7,
    )

    assert row["run"] == "Odaiba"
    assert row["variant"] == "with_smoother"
    assert row["predict_guide"] == "imu"
    assert row["tdcp_position_update"] is True
    assert row["n_tdcp_pu_used"] == 2
    assert row["n_doppler_kf_gate_skip"] == 3
    assert row["n_widelane_forward_guard_applied"] == 4
    assert row["fgo_local_window"] == "auto"
    assert row["fgo_local_two_step"] is True
    assert row["fgo_local_lambda_fixed"] == 7
    assert row["fgo_local_lambda_fixed_observations"] == 11
    assert row["fgo_local_resolved_window"] == (10, 20)
    assert row["fgo_local_solve_window"] == (9, 21)
    assert row["fgo_local_initial_error"] == 100.0
    assert row["fgo_local_final_error"] == 25.0
    assert row["smoother"] is True
    assert row["n_particles"] == 123
    assert row["position_update_sigma"] == 1.9
    assert row["forward_p50"] == 3.0
    assert row["smoothed_p50_before_fgo"] == 1.5
    assert row["smoothed_p50"] == 2.0
    assert row["n_epochs"] == 42
    assert row["ms_per_epoch"] == 0.7


def test_build_pf_smoother_result_row_handles_disabled_updates_and_missing_metrics():
    parser = build_pf_smoother_arg_parser(default_sigma_pos=1.2)
    args = parser.parse_args(["--data-root", "/tmp/UrbanNav-Tokyo"])

    row = build_pf_smoother_result_row(
        args=args,
        out={},
        run_name="Odaiba",
        label="forward_only",
        use_smoother=False,
        position_update_sigma=None,
        forward_metrics=None,
        smoothed_metrics=None,
        n_epochs=0,
        ms_per_epoch=0.0,
    )

    assert row["position_update_sigma"] == "off"
    assert row["fgo_local_applied"] is False
    assert row["fgo_local_resolved_window"] is None
    assert row["forward_p50"] is None
    assert row["smoothed_p50"] is None
    assert row["n_epochs"] == 0
