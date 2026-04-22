import gnss_gpu.pf_smoother_run_result as run_result
from gnss_gpu.pf_smoother_run_result import build_initial_pf_smoother_run_result


def _minimal_context(**overrides):
    context = {key: None for key in run_result._COPY_KEYS}
    context.update(
        {
            "run_name": "Odaiba",
            "dd_sigma_scale_sum": 0.0,
            "n_dd_sigma_relaxed": 0,
            "fgo_motion_source": "predict",
            "fgo_local_lambda": False,
        }
    )
    context.update(overrides)
    return context


def test_build_initial_pf_smoother_run_result_copies_config_counters_and_defaults():
    result = build_initial_pf_smoother_run_result(
        _minimal_context(
            n_particles=123,
            predict_guide="imu",
            use_smoother=True,
            tdcp_position_update=True,
            mupf_dd=True,
            n_dd_used=4,
            n_dd_skip=6,
            n_wl_candidate_pairs=20,
            n_wl_fixed_pairs=10,
            elapsed_ms=12.5,
            fgo_motion_source="prefer_tdcp",
            fgo_local_lambda=True,
        )
    )

    assert result["run"] == "Odaiba"
    assert result["n_particles"] == 123
    assert result["predict_guide"] == "imu"
    assert result["use_smoother"] is True
    assert result["tdcp_position_update"] is True
    assert result["mupf_dd"] is True
    assert result["n_dd_used"] == 4
    assert result["n_dd_skip"] == 6
    assert result["n_wl_candidate_pairs"] == 20
    assert result["n_wl_fixed_pairs"] == 10
    assert result["elapsed_ms"] == 12.5
    assert result["fgo_local_motion_source"] == "prefer_tdcp"
    assert result["fgo_local_lambda"] is True
    assert result["forward_metrics"] is None
    assert result["smoothed_metrics"] is None
    assert result["epoch_diagnostics"] is None
    assert result["n_tail_guard_applied"] == 0
    assert result["fgo_local_applied"] is False
    assert result["fgo_local_info"] is None


def test_build_initial_pf_smoother_run_result_computes_mean_dd_sigma_scale():
    relaxed = build_initial_pf_smoother_run_result(
        _minimal_context(
            n_dd_sigma_relaxed=4,
            dd_sigma_scale_sum=6.0,
        )
    )
    unrelaxed = build_initial_pf_smoother_run_result(
        _minimal_context(
            n_dd_sigma_relaxed=0,
            dd_sigma_scale_sum=6.0,
        )
    )

    assert relaxed["mean_dd_sigma_scale"] == 1.5
    assert unrelaxed["mean_dd_sigma_scale"] is None
