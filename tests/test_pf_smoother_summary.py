from gnss_gpu.pf_smoother_summary import (
    build_pf_smoother_summary_lines,
    print_pf_smoother_run_summary,
)


def test_build_pf_smoother_summary_lines_covers_motion_and_imu_sections():
    lines = build_pf_smoother_summary_lines(
        {
            "predict_guide": "tdcp_adaptive",
            "tdcp_rms_threshold": 2.5,
            "n_tdcp_used": 7,
            "n_tdcp_fallback": 3,
            "tdcp_position_update": True,
            "n_tdcp_pu_used": 4,
            "n_tdcp_pu_skip": 6,
            "n_tdcp_pu_gate_skip": 2,
            "doppler_per_particle": True,
            "n_doppler_pp_used": 5,
            "n_doppler_pp_skip": 1,
            "rbpf_velocity_kf": True,
            "n_doppler_kf_used": 8,
            "n_doppler_kf_skip": 2,
            "n_doppler_kf_gate_skip": 1,
        }
    )

    assert lines == [
        "  [tdcp_adaptive] TDCP used 7/10 epochs, fallback 3/10 (rms_threshold=2.5m)",
        "  [tdcp_position_update] used 4/10 epochs, skip 6/10, gate_skip=2",
        "  [doppler_per_particle] used 5/6 epochs, skip 1/6",
        "  [rbpf_velocity_kf] Doppler KF used 8/10 epochs, skip 2/10, gate_skip=1",
    ]


def test_build_pf_smoother_summary_lines_covers_carrier_pr_and_widelane_sections():
    lines = build_pf_smoother_summary_lines(
        {
            "predict_guide": "imu",
            "n_imu_used": 9,
            "n_imu_fallback": 1,
            "n_imu_stop_detected": 2,
            "imu_stop_sigma_pos": 0.1,
            "imu_tight_coupling": True,
            "n_imu_tight_used": 3,
            "n_imu_tight_skip": 7,
            "mupf_dd": True,
            "n_dd_used": 4,
            "n_dd_skip": 6,
            "mupf_dd_gate_adaptive_floor_cycles": 0.18,
            "n_dd_gate_pairs_rejected": 5,
            "n_dd_gate_epoch_skip": 1,
            "n_dd_skip_support_guard": 2,
            "n_dd_sigma_relaxed": 3,
            "mean_dd_sigma_scale": 1.23456,
            "n_carrier_anchor_used": 2,
            "n_carrier_anchor_propagated": 11,
            "n_dd_fallback_undiff_used": 1,
            "n_dd_fallback_tracked_attempted": 2,
            "n_dd_fallback_tracked_used": 3,
            "n_dd_fallback_weak_dd_replaced": 4,
            "dd_pseudorange": True,
            "n_dd_pr_used": 6,
            "n_dd_pr_skip": 4,
            "dd_pseudorange_gate_residual_m": 3.0,
            "n_dd_pr_gate_pairs_rejected": 7,
            "n_dd_pr_gate_epoch_skip": 2,
            "widelane": True,
            "n_wl_used": 5,
            "n_wl_skip": 5,
            "n_wl_candidate_pairs": 20,
            "n_wl_fixed_pairs": 10,
            "n_wl_low_fix_rate": 1,
            "n_wl_gate_skip": 2,
            "n_wl_gate_pair_rejected": 3,
        }
    )

    assert lines == [
        "  [imu] IMU used 9/10 epochs, fallback 1/10",
        "  [imu_stop_detect] stop epochs=2, sigma_pos=0.1",
        "  [imu_tight] IMU position_update used 3/10 epochs, skip 7/10",
        "  [mupf_dd] DD-AFV used 4/10 epochs, skip 6/10",
        "  [mupf_dd_gate] pair_rejected=5 epoch_skip=1",
        "  [mupf_dd_support_skip] epochs=2",
        "  [mupf_dd_sigma_relax] epochs=3 mean_scale=1.235",
        "  [carrier_anchor] epochs=2",
        "  [carrier_anchor_tdcp] propagated_rows=11",
        "  [mupf_dd_fallback_undiff] epochs=1",
        "  [mupf_dd_fallback_tracked_attempt] epochs=2",
        "  [mupf_dd_fallback_tracked] epochs=3",
        "  [mupf_dd_fallback_weak_dd] epochs=4",
        "  [dd_pseudorange] used 6/10 epochs, skip 4/10",
        "  [dd_pseudorange_gate] pair_rejected=7 epoch_skip=2",
        "  [widelane] used 5/10 epochs, fixed_pairs=10/20 (50.0%), low_fix_rate_epochs=1",
        "  [widelane_gate] epoch_skip=2 pair_rejected=3",
    ]


def test_print_pf_smoother_run_summary_uses_injected_printer():
    printed = []

    print_pf_smoother_run_summary(
        {
            "predict_guide": "imu_spp_blend",
            "n_imu_used": 1,
            "n_imu_fallback": 0,
        },
        print_func=printed.append,
    )

    assert printed == ["  [imu_spp_blend] IMU used 1/1 epochs, fallback 0/1"]
