from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats


def test_forward_run_stats_defaults_match_result_counter_keys():
    stats = ForwardRunStats()
    context = stats.as_result_context()

    assert context["n_imu_used"] == 0
    assert context["n_dd_pr_used"] == 0
    assert context["n_wl_used"] == 0
    assert context["n_dd_used"] == 0
    assert context["n_doppler_kf_gate_skip"] == 0
    assert context["dd_sigma_scale_sum"] == 0.0


def test_forward_run_stats_records_relaxed_dd_sigma_scale():
    stats = ForwardRunStats()

    stats.record_dd_sigma_relaxed(1.25)
    stats.record_dd_sigma_relaxed(1.75)

    assert stats.n_dd_sigma_relaxed == 2
    assert stats.dd_sigma_scale_sum == 3.0
    assert stats.as_result_context()["n_dd_sigma_relaxed"] == 2
