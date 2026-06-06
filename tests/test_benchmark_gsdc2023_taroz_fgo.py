from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark_gsdc2023_taroz_fgo import (
    DEFAULT_VARIANTS,
    apply_matlab_residual_diagnostics_mask_config,
    base_config_from_args,
    config_for_variant,
    parse_variants,
    result_row,
    run_benchmark,
    summarize_results,
)
from experiments.gsdc2023_bridge_config import TAROZ_FGO_WEIGHT_MODE, BridgeConfig
from experiments.gsdc2023_per_type_kernel import PerTypeKernel


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "motion_sigma_m": 0.2,
        "factor_dt_max_s": 1.5,
        "fgo_iters": 8,
        "weight_mode": "sin2el",
        "fgo_weight_mode": "same",
        "position_source": "gated",
        "chunk_epochs": 200,
        "gated_threshold": 500.0,
        "vd": True,
        "multi_gnss": True,
        "tdcp": True,
        "tdcp_geometry_correction": True,
        "dual_frequency": True,
        "apply_observation_mask": False,
        "trip": ["train/course/pixel4"],
        "trip_file": None,
        "phone": [],
        "trip_type": [],
        "sample_per_type": 0,
        "limit": 0,
        "variants": DEFAULT_VARIANTS,
        "data_root": None,
        "max_epochs": 20,
        "start_epoch": 0,
        "use_matlab_residual_diagnostics_mask": False,
        "keep_going": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _kernel() -> PerTypeKernel:
    return PerTypeKernel(
        trip_type="Street",
        pr_huber_k=0.1,
        doppler_huber_k=0.4,
        carrier_huber_k=0.2,
        motion_sigma_m=0.05,
    )


def test_parse_variants_rejects_unknown() -> None:
    assert parse_variants("baseline,taroz_pr") == ("baseline", "taroz_pr")
    with pytest.raises(Exception):
        parse_variants("baseline,nope")


def test_config_for_variant_wires_taroz_factor_huber_ablation() -> None:
    base = BridgeConfig(motion_sigma_m=0.2, fgo_weight_mode=None)
    kernel = _kernel()

    pr = config_for_variant("taroz_pr", base, kernel)
    assert pr.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert pr.fgo_huber_k_pr == pytest.approx(0.1)
    assert pr.fgo_huber_k_doppler == 0.0
    assert pr.fgo_huber_k_tdcp == 0.0
    assert pr.per_type_kernel_enabled is False

    pr_d_l = config_for_variant("taroz_pr_d_l", base, kernel)
    assert pr_d_l.fgo_huber_k_pr == pytest.approx(0.1)
    assert pr_d_l.fgo_huber_k_doppler == pytest.approx(0.4)
    assert pr_d_l.fgo_huber_k_tdcp == pytest.approx(0.2)

    gnss_only = config_for_variant(
        "taroz_gnss_only",
        BridgeConfig(
            stop_velocity_sigma_mps=3.0,
            stop_position_sigma_m=4.0,
            stop_attitude_sigma_rad=5.0,
            apply_imu_prior=True,
            imu_accel_bias_state=True,
            graph_relative_height=True,
            apply_absolute_height=True,
            apply_relative_height=True,
        ),
        kernel,
    )
    assert gnss_only.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert gnss_only.clock_drift_sigma_m == pytest.approx(0.1)
    assert gnss_only.clock_use_average_drift is True
    assert gnss_only.per_type_kernel_enabled is True
    assert gnss_only.per_type_kernel_motion_enabled is True
    assert gnss_only.stop_velocity_sigma_mps == 0.0
    assert gnss_only.stop_position_sigma_m == 0.0
    assert gnss_only.stop_attitude_sigma_rad == 0.0
    assert gnss_only.apply_imu_prior is False
    assert gnss_only.imu_attitude_state is False
    assert gnss_only.imu_diagonal_covariance is False
    assert gnss_only.imu_accel_bias_state is False
    assert gnss_only.graph_relative_height is False
    assert gnss_only.apply_absolute_height is False
    assert gnss_only.apply_relative_height is False

    phone_aware_pixel = config_for_variant("taroz_phone_aware", base, kernel, phone="pixel5")
    assert phone_aware_pixel.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert phone_aware_pixel.clock_drift_sigma_m == pytest.approx(0.1)
    assert phone_aware_pixel.clock_use_average_drift is True
    assert phone_aware_pixel.per_type_kernel_enabled is True
    assert phone_aware_pixel.per_type_kernel_motion_enabled is True
    assert phone_aware_pixel.stop_velocity_sigma_mps == 0.0
    assert phone_aware_pixel.stop_attitude_sigma_rad == 0.0
    assert phone_aware_pixel.graph_relative_height is False

    phone_aware_non_pixel = config_for_variant("taroz_phone_aware", base, kernel, phone="sm-a205u")
    assert phone_aware_non_pixel.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert phone_aware_non_pixel.clock_drift_sigma_m == pytest.approx(base.clock_drift_sigma_m)
    assert phone_aware_non_pixel.clock_use_average_drift is base.clock_use_average_drift
    assert phone_aware_non_pixel.fgo_huber_k_pr == 0.0
    assert phone_aware_non_pixel.fgo_huber_k_doppler == 0.0
    assert phone_aware_non_pixel.fgo_huber_k_tdcp == 0.0
    assert phone_aware_non_pixel.per_type_kernel_enabled is False
    assert phone_aware_non_pixel.per_type_kernel_motion_enabled is base.per_type_kernel_motion_enabled

    gnss_no_motion = config_for_variant("taroz_gnss_no_motion", base, kernel)
    assert gnss_no_motion.clock_drift_sigma_m == pytest.approx(0.1)
    assert gnss_no_motion.clock_use_average_drift is True
    assert gnss_no_motion.per_type_kernel_enabled is True
    assert gnss_no_motion.per_type_kernel_motion_enabled is False
    assert gnss_no_motion.stop_velocity_sigma_mps == 0.0
    assert gnss_no_motion.graph_relative_height is False

    gnss_no_clock = config_for_variant("taroz_gnss_no_clock", base, kernel)
    assert gnss_no_clock.clock_drift_sigma_m == pytest.approx(base.clock_drift_sigma_m)
    assert gnss_no_clock.clock_use_average_drift is base.clock_use_average_drift
    assert gnss_no_clock.per_type_kernel_enabled is True
    assert gnss_no_clock.per_type_kernel_motion_enabled is True
    assert gnss_no_clock.tdcp_enabled is True
    assert gnss_no_clock.stop_velocity_sigma_mps == 0.0
    assert gnss_no_clock.graph_relative_height is False

    gnss_no_tdcp = config_for_variant("taroz_gnss_no_tdcp", base, kernel)
    assert gnss_no_tdcp.clock_drift_sigma_m == pytest.approx(0.1)
    assert gnss_no_tdcp.clock_use_average_drift is True
    assert gnss_no_tdcp.per_type_kernel_enabled is True
    assert gnss_no_tdcp.per_type_kernel_motion_enabled is True
    assert gnss_no_tdcp.tdcp_enabled is False
    assert gnss_no_tdcp.stop_velocity_sigma_mps == 0.0
    assert gnss_no_tdcp.graph_relative_height is False

    gnss_no_clock_no_tdcp = config_for_variant("taroz_gnss_no_clock_no_tdcp", base, kernel)
    assert gnss_no_clock_no_tdcp.clock_drift_sigma_m == pytest.approx(base.clock_drift_sigma_m)
    assert gnss_no_clock_no_tdcp.clock_use_average_drift is base.clock_use_average_drift
    assert gnss_no_clock_no_tdcp.per_type_kernel_enabled is True
    assert gnss_no_clock_no_tdcp.per_type_kernel_motion_enabled is True
    assert gnss_no_clock_no_tdcp.tdcp_enabled is False
    assert gnss_no_clock_no_tdcp.stop_velocity_sigma_mps == 0.0
    assert gnss_no_clock_no_tdcp.graph_relative_height is False

    full = config_for_variant("taroz_full", base, kernel)
    assert full.per_type_kernel_enabled is True
    assert full.per_type_kernel_motion_enabled is True
    assert full.clock_drift_sigma_m == pytest.approx(0.1)
    assert full.clock_use_average_drift is True
    assert full.stop_attitude_sigma_rad == pytest.approx(np.deg2rad(0.1))
    assert full.graph_relative_height is True

    no_clock = config_for_variant("taroz_full_no_clock", base, kernel)
    assert no_clock.clock_drift_sigma_m == pytest.approx(base.clock_drift_sigma_m)
    assert no_clock.clock_use_average_drift is True
    assert no_clock.stop_velocity_sigma_mps == pytest.approx(0.01)
    assert no_clock.stop_attitude_sigma_rad == pytest.approx(np.deg2rad(0.1))
    assert no_clock.graph_relative_height is True

    assert config_for_variant("taroz_full_clock0p2", base, kernel).clock_drift_sigma_m == pytest.approx(0.2)
    assert config_for_variant("taroz_full_clock0p5", base, kernel).clock_drift_sigma_m == pytest.approx(0.5)
    assert config_for_variant("taroz_full_clock1p0", base, kernel).clock_drift_sigma_m == pytest.approx(1.0)

    no_stop = config_for_variant("taroz_full_no_stop", base, kernel)
    assert no_stop.clock_drift_sigma_m == pytest.approx(0.1)
    assert no_stop.stop_velocity_sigma_mps == pytest.approx(base.stop_velocity_sigma_mps)
    assert no_stop.stop_position_sigma_m == pytest.approx(base.stop_position_sigma_m)
    assert no_stop.stop_attitude_sigma_rad == pytest.approx(base.stop_attitude_sigma_rad)

    no_height = config_for_variant("taroz_full_no_height", base, kernel)
    assert no_height.graph_relative_height is base.graph_relative_height
    assert no_height.apply_absolute_height is base.apply_absolute_height
    assert no_height.relative_height_sigma_m == pytest.approx(base.relative_height_sigma_m)

    no_priors = config_for_variant("taroz_full_no_priors", base, kernel)
    assert no_priors.per_type_kernel_enabled is True
    assert no_priors.per_type_kernel_motion_enabled is True
    assert no_priors.clock_drift_sigma_m == pytest.approx(base.clock_drift_sigma_m)
    assert no_priors.stop_velocity_sigma_mps == pytest.approx(base.stop_velocity_sigma_mps)
    assert no_priors.stop_attitude_sigma_rad == pytest.approx(base.stop_attitude_sigma_rad)
    assert no_priors.graph_relative_height is base.graph_relative_height


def test_result_row_includes_delta_and_config_knobs() -> None:
    payload = {
        "n_epochs": 10,
        "selected_source_mode": "gated",
        "selected_score_m": 3.0,
        "kaggle_wls_score_m": 5.0,
        "raw_wls_score_m": 4.0,
        "fgo_score_m": 2.0,
        "selected_mse_pr": 7.0,
        "baseline_mse_pr": 9.0,
        "raw_wls_mse_pr": 8.0,
        "fgo_mse_pr": 6.0,
        "selected_metrics": {"p50_m": 1.0, "p95_m": 5.0},
        "fgo_metrics": {"p50_m": 0.8, "p95_m": 4.0},
        "selected_source_counts": {"baseline": 2, "raw_wls": 3, "fgo": 5, "auto": 7, "fgo_no_tdcp": 11},
    }
    baseline = {"selected_score_m": 4.0, "selected_mse_pr": 8.0, "fgo_score_m": 3.0, "fgo_mse_pr": 7.0}
    cfg = BridgeConfig(fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE, fgo_huber_k_pr=0.1)

    row = result_row(
        variant="taroz_pr",
        trip="train/course/pixel4",
        trip_type="Street",
        kernel=_kernel(),
        config=cfg,
        payload=payload,
        baseline_payload=baseline,
        elapsed_s=1.5,
    )

    assert row["selected_fgo_epochs"] == 5
    assert row["selected_auto_epochs"] == 7
    assert row["selected_fgo_no_tdcp_epochs"] == 11
    assert row["selected_p95_m"] == 5.0
    assert row["config_fgo_weight_mode"] == TAROZ_FGO_WEIGHT_MODE
    assert row["config_tdcp_enabled"] is True
    assert row["config_tdcp_geometry_correction"] is True
    assert row["config_apply_observation_mask"] is False
    assert row["config_matlab_residual_diagnostics_mask_path"] is None
    assert row["delta_selected_score_m_vs_baseline"] == pytest.approx(-1.0)
    assert row["delta_fgo_mse_pr_vs_baseline"] == pytest.approx(-1.0)


def test_apply_matlab_residual_diagnostics_mask_config_requires_sidecar(tmp_path) -> None:
    trip = "train/course/pixel4"
    cfg = BridgeConfig()

    assert apply_matlab_residual_diagnostics_mask_config(
        cfg,
        data_root=tmp_path,
        trip=trip,
        enabled=False,
    ) is cfg

    with pytest.raises(FileNotFoundError):
        apply_matlab_residual_diagnostics_mask_config(
            cfg,
            data_root=tmp_path,
            trip=trip,
            enabled=True,
        )

    diagnostics_path = tmp_path / trip / "phone_data_residual_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True)
    diagnostics_path.write_text("freq,utcTimeMillis,sys,svid,p_factor_finite,d_factor_finite,l_factor_finite\n")

    masked = apply_matlab_residual_diagnostics_mask_config(
        cfg,
        data_root=tmp_path,
        trip=trip,
        enabled=True,
    )
    assert masked.matlab_residual_diagnostics_mask_path == diagnostics_path


def test_base_config_from_args_wires_mask_and_tdcp_geometry_flags() -> None:
    cfg = base_config_from_args(
        _args(
            apply_observation_mask=True,
            tdcp_geometry_correction=False,
        ),
    )

    assert cfg.apply_observation_mask is True
    assert cfg.tdcp_geometry_correction is False


def test_summarize_results_groups_by_variant_and_type() -> None:
    frame = pd.DataFrame(
        [
            {"variant": "baseline", "trip_type": "Street", "trip": "a", "status": "ok", "selected_score_m": 4.0},
            {
                "variant": "taroz_pr",
                "trip_type": "Street",
                "trip": "a",
                "status": "ok",
                "selected_score_m": 3.0,
                "delta_selected_score_m_vs_baseline": -1.0,
            },
            {"variant": "taroz_pr", "trip_type": "Street", "trip": "b", "status": "error", "selected_score_m": 99.0},
        ],
    )

    summary = summarize_results(frame)

    assert summary.loc[summary["variant"].eq("taroz_pr"), "trip_count"].iloc[0] == 1
    assert summary.loc[summary["variant"].eq("taroz_pr"), "mean_selected_score_m"].iloc[0] == pytest.approx(3.0)


def test_run_benchmark_uses_fake_validator(monkeypatch) -> None:
    import experiments.benchmark_gsdc2023_taroz_fgo as bench

    class FakeResult:
        def __init__(self, score: float):
            self.score = score

        def metrics_payload(self) -> dict[str, object]:
            return {
                "n_epochs": 4,
                "selected_source_mode": "gated",
                "selected_score_m": self.score,
                "selected_mse_pr": self.score + 10.0,
                "fgo_score_m": self.score + 1.0,
                "fgo_mse_pr": self.score + 11.0,
                "selected_source_counts": {"fgo": 4},
            }

    scores = {"baseline": 5.0, "taroz_pr": 4.0}

    def fake_validate(_data_root, _trip, *, max_epochs, start_epoch, config):
        del max_epochs, start_epoch
        if config.fgo_huber_k_pr > 0.0:
            return FakeResult(scores["taroz_pr"])
        return FakeResult(scores["baseline"])

    monkeypatch.setattr(bench, "validate_raw_gsdc2023_trip", fake_validate)
    monkeypatch.setattr(bench, "kernel_for_trip", lambda _root, _trip: ("Street", _kernel()))

    frame = run_benchmark(
        _args(
            data_root="unused",
            variants=("baseline", "taroz_pr"),
        ),
    )

    assert frame["variant"].tolist() == ["baseline", "taroz_pr"]
    assert frame.loc[1, "delta_selected_score_m_vs_baseline"] == pytest.approx(-1.0)


def test_run_benchmark_can_apply_matlab_residual_diagnostics_mask(monkeypatch, tmp_path) -> None:
    import experiments.benchmark_gsdc2023_taroz_fgo as bench

    trip = "train/course/pixel4"
    diagnostics_path = tmp_path / trip / "phone_data_residual_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True)
    diagnostics_path.write_text("freq,utcTimeMillis,sys,svid,p_factor_finite,d_factor_finite,l_factor_finite\n")
    seen_paths = []

    class FakeResult:
        def metrics_payload(self) -> dict[str, object]:
            return {
                "n_epochs": 4,
                "selected_source_mode": "gated",
                "selected_score_m": 5.0,
                "selected_mse_pr": 15.0,
                "fgo_score_m": 4.0,
                "fgo_mse_pr": 14.0,
                "selected_source_counts": {"baseline": 4},
            }

    def fake_validate(_data_root, _trip, *, max_epochs, start_epoch, config):
        del max_epochs, start_epoch
        seen_paths.append(config.matlab_residual_diagnostics_mask_path)
        return FakeResult()

    monkeypatch.setattr(bench, "validate_raw_gsdc2023_trip", fake_validate)
    monkeypatch.setattr(bench, "kernel_for_trip", lambda _root, _trip: ("Street", _kernel()))

    frame = run_benchmark(
        _args(
            data_root=tmp_path,
            variants=("baseline",),
            trip=[trip],
            use_matlab_residual_diagnostics_mask=True,
        ),
    )

    assert seen_paths == [diagnostics_path]
    assert frame.loc[0, "config_matlab_residual_diagnostics_mask_path"] == str(diagnostics_path)


def test_run_benchmark_keep_going_records_missing_matlab_mask(monkeypatch, tmp_path) -> None:
    import experiments.benchmark_gsdc2023_taroz_fgo as bench

    trip = "train/course/pixel4"
    called = False

    def fake_validate(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("validator should not run when diagnostics mask is missing")

    monkeypatch.setattr(bench, "validate_raw_gsdc2023_trip", fake_validate)
    monkeypatch.setattr(bench, "kernel_for_trip", lambda _root, _trip: ("Street", _kernel()))

    frame = run_benchmark(
        _args(
            data_root=tmp_path,
            variants=("baseline",),
            trip=[trip],
            use_matlab_residual_diagnostics_mask=True,
            keep_going=True,
        ),
    )

    assert called is False
    assert frame.loc[0, "status"] == "error"
    assert frame.loc[0, "error_type"] == "FileNotFoundError"
    assert "phone_data_residual_diagnostics.csv" in frame.loc[0, "error"]
