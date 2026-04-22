import numpy as np

import gnss_gpu.pf_smoother_postrun as postrun
from gnss_gpu.pf_smoother_config import (
    LocalFgoPostprocessConfig,
    SmootherPostprocessConfig,
)
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


class _FakeParticleFilter:
    def __init__(self, smoothed=None):
        self.smoothed = smoothed
        self.smooth_calls = []

    def smooth(self, **kwargs):
        self.smooth_calls.append(kwargs)
        return np.asarray(self.smoothed, dtype=np.float64), None


def _metrics(positions, ground_truth):
    pos = np.asarray(positions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    return {
        "n_epochs": int(len(pos)),
        "p50": float(np.median(np.linalg.norm(pos - gt, axis=1))),
        "p95": float(np.percentile(np.linalg.norm(pos - gt, axis=1), 95)),
        "rms_2d": float(np.sqrt(np.mean(np.sum((pos[:, :2] - gt[:, :2]) ** 2, axis=1)))),
    }


def _ecef_errors(positions, ground_truth):
    pos = np.asarray(positions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    err3 = np.linalg.norm(pos - gt, axis=1)
    err2 = np.linalg.norm(pos[:, :2] - gt[:, :2], axis=1)
    return err2, err3


def test_finalize_pf_smoother_postrun_returns_initial_result_without_alignment():
    result = {"run": "Odaiba", "forward_metrics": None}

    finalized = postrun.finalize_pf_smoother_postrun(
        result,
        _FakeParticleFilter(),
        ForwardRunBuffers(),
        position_update_sigma=None,
        use_smoother=True,
        smoother_config=SmootherPostprocessConfig(),
        local_fgo_config=LocalFgoPostprocessConfig(),
        fgo_motion_source="predict",
        collect_epoch_diagnostics=True,
        compute_metrics_func=_metrics,
        ecef_errors_func=_ecef_errors,
    )

    assert finalized.result is result
    assert result["forward_metrics"] is None
    assert finalized.forward_positions.shape == (0,)
    assert finalized.smoothed_positions is None


def test_finalize_pf_smoother_postrun_adds_forward_metrics_and_diagnostics():
    buffers = ForwardRunBuffers()
    buffers.forward_aligned = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
    ]
    buffers.gt_aligned = [
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    ]
    buffers.aligned_epoch_diagnostics = [{}, {}]
    result = {"run": "Odaiba"}

    finalized = postrun.finalize_pf_smoother_postrun(
        result,
        _FakeParticleFilter(),
        buffers,
        position_update_sigma=None,
        use_smoother=False,
        smoother_config=SmootherPostprocessConfig(),
        local_fgo_config=LocalFgoPostprocessConfig(),
        fgo_motion_source="predict",
        collect_epoch_diagnostics=True,
        compute_metrics_func=_metrics,
        ecef_errors_func=_ecef_errors,
    )

    assert finalized.result["forward_metrics"]["n_epochs"] == 2
    assert finalized.result["epoch_diagnostics"] is buffers.aligned_epoch_diagnostics
    assert buffers.aligned_epoch_diagnostics[0]["forward_error_2d"] == 1.0
    assert buffers.aligned_epoch_diagnostics[1]["forward_error_3d"] == 2.0
    assert finalized.smoothed_positions is None


def test_finalize_pf_smoother_postrun_applies_smoother_and_local_fgo(monkeypatch):
    captured = {}

    def fake_local_fgo(
        smoothed_aligned,
        aligned_indices,
        stored_motion_deltas,
        stored_tdcp_motion_deltas,
        stored_dd_carrier,
        stored_dd_pseudorange,
        stored_undiff_pr,
        epoch_diagnostics,
        *,
        window_spec,
        min_epochs,
        dd_max_pairs,
        config,
        lambda_config,
        motion_source,
        two_step,
        stage1_prior_sigma_m,
        stage1_motion_sigma_m,
        stage1_pr_sigma_m,
    ):
        captured.update(
            {
                "aligned_indices": aligned_indices,
                "window_spec": window_spec,
                "min_epochs": min_epochs,
                "dd_max_pairs": dd_max_pairs,
                "config_prior": config.prior_sigma_m,
                "lambda_ratio": lambda_config.ratio_threshold,
                "motion_source": motion_source,
                "two_step": two_step,
                "stage1_prior": stage1_prior_sigma_m,
            }
        )
        return np.asarray(smoothed_aligned, dtype=np.float64) + 1.0, {
            "applied": True,
            "window": "0:1",
            "solve_window": "0:1",
            "factor_counts": {"prior": 1},
            "initial_error": 4.0,
            "final_error": 2.0,
            "lambda": {"n_fixed": 1, "n_fixed_observations": 2},
            "stage1": {"applied": False, "reason": "disabled"},
            "motion_source": motion_source,
            "motion_tdcp_selected_edges": 1,
        }

    monkeypatch.setattr(postrun, "_apply_local_fgo_postprocess", fake_local_fgo)

    buffers = ForwardRunBuffers()
    buffers.append_smoother_observations(None, None, None)
    buffers.append_aligned_epoch(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        stop_flag=False,
        use_smoother=True,
    )
    buffers.append_smoother_observations(None, None, None)
    buffers.append_aligned_epoch(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        stop_flag=False,
        use_smoother=True,
    )
    buffers.aligned_epoch_diagnostics = [{"forward_error_2d": 0.0}, {"forward_error_2d": 1.0}]
    pf = _FakeParticleFilter(
        smoothed=[
            [0.25, 0.0, 0.0],
            [1.25, 0.0, 0.0],
        ],
    )
    printed = []

    finalized = postrun.finalize_pf_smoother_postrun(
        {"run": "Odaiba"},
        pf,
        buffers,
        position_update_sigma=1.9,
        use_smoother=True,
        smoother_config=SmootherPostprocessConfig(
            position_update_sigma=2.2,
            skip_widelane_dd_pseudorange=True,
        ),
        local_fgo_config=LocalFgoPostprocessConfig(
            window="0:1",
            window_min_epochs=2,
            dd_max_pairs=3,
            prior_sigma_m=0.4,
            lambda_enabled=True,
            lambda_ratio_threshold=3.5,
            motion_source="prefer_tdcp",
            two_step=True,
            stage1_prior_sigma_m=0.7,
        ),
        fgo_motion_source="prefer_tdcp",
        collect_epoch_diagnostics=True,
        compute_metrics_func=_metrics,
        ecef_errors_func=_ecef_errors,
        print_func=printed.append,
    )

    assert pf.smooth_calls == [
        {
            "position_update_sigma": 2.2,
            "skip_widelane_dd_pseudorange": True,
        }
    ]
    assert captured["aligned_indices"] == [0, 1]
    assert captured["window_spec"] == "0:1"
    assert captured["min_epochs"] == 2
    assert captured["dd_max_pairs"] == 3
    assert captured["config_prior"] == 0.4
    assert captured["lambda_ratio"] == 3.5
    assert captured["motion_source"] == "prefer_tdcp"
    assert captured["two_step"] is True
    assert captured["stage1_prior"] == 0.7
    assert finalized.result["fgo_local_applied"] is True
    assert finalized.result["smoother_position_update_sigma"] == 2.2
    assert finalized.result["smoothed_metrics_before_fgo"]["n_epochs"] == 2
    assert finalized.result["smoothed_metrics"]["n_epochs"] == 2
    np.testing.assert_allclose(finalized.smoothed_positions[0], [1.25, 1.0, 1.0])
    assert buffers.aligned_epoch_diagnostics[0]["smoothed_error_2d"] > 0.0
    assert "lambda_fixed=1/obs=2" in printed[0]
