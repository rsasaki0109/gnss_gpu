from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.build_gsdc2023_bridge_submission import (
    apply_taroz_phone_aware_preset,
    build_config,
    bridge_cache_metadata,
    bridge_output_dir,
    bridge_trip_id,
    load_cached_bridge_trip,
    ordered_trip_ids,
    phone_from_sample_trip_id,
    run_bridge_submission,
    run_one_bridge_trip,
    submission_from_bridge_tables,
    use_taroz_gnss_only_for_phone,
)
from experiments.gsdc2023_bridge_config import TAROZ_FGO_WEIGHT_MODE, BridgeConfig
from experiments.gsdc2023_output import TAROZ_FGO_CANDIDATE_SOURCES


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "motion_sigma_m": 0.2,
        "factor_dt_max_s": 1.5,
        "fgo_iters": 8,
        "position_source": "gated",
        "chunk_epochs": 200,
        "gated_threshold": 500.0,
        "vd": True,
        "multi_gnss": True,
        "tdcp": True,
        "tdcp_weight_scale": 1.0e-4,
        "tdcp_l5_weight_scale": 25.0,
        "tdcp_geometry_correction": True,
        "dual_frequency": True,
        "ct_rbpf_fgo": False,
        "ct_rbpf_motion_sigma_m": 0.2,
        "dd_carrier_fgo": False,
        "dd_carrier_base_obs_template": None,
        "dd_carrier_require_base_obs_template": False,
        "dd_carrier_tow_snap_tolerance_s": 0.6,
        "dd_carrier_min_dd_pairs": 4,
        "dd_carrier_smooth_corrections": False,
        "dd_carrier_min_anchor_coverage": 0.6,
        "fgo_raw_wls_proxy_rescue": False,
        "fgo_raw_wls_proxy_rescue_phones": "pixel4",
        "fgo_raw_wls_proxy_rescue_mse_ratio_max": 1.15,
        "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": 1.25,
        "fgo_raw_wls_proxy_rescue_quality_delta_max": -0.20,
        "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": 0.0,
        "taroz_fgo_candidates": False,
        "taroz_fgo_candidate_sources": ",".join(TAROZ_FGO_CANDIDATE_SOURCES),
        "taroz_fgo": False,
        "taroz_marupaku": False,
        "taroz_imu_factor_mask_csv": None,
        "taroz_phone_aware": False,
        "fgo_huber_k_pr": 0.0,
        "fgo_huber_k_doppler": 0.0,
        "fgo_huber_k_tdcp": 0.0,
        "gate_fgo_low_baseline_mse_pr_max": None,
        "gate_fgo_baseline_mse_pr_min": None,
        "gate_fgo_baseline_gap_p95_floor_m": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bridge_trip_id_accepts_sample_or_test_trip_id() -> None:
    assert bridge_trip_id("course/phone") == "test/course/phone"
    assert bridge_trip_id("test/course/phone") == "test/course/phone"
    assert str(bridge_output_dir(Path("/tmp/bridge"), "test/course/phone")).endswith("bridge/course/phone")
    assert phone_from_sample_trip_id("test/course/pixel5") == "pixel5"


def test_load_cached_bridge_trip_requires_valid_metrics(tmp_path: Path) -> None:
    root = tmp_path / "bridge"
    trip_dir = root / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [37.0],
            "LongitudeDegrees": [-122.0],
        },
    ).to_csv(trip_dir / "bridge_positions.csv", index=False)

    assert load_cached_bridge_trip(root, "course/phone") is None

    (trip_dir / "bridge_metrics.json").write_text('{"fgo_iters": 4, "mse_pr": 12.5}\n', encoding="utf-8")

    table, metrics = load_cached_bridge_trip(root, "test/course/phone")

    assert table["UnixTimeMillis"].tolist() == [1000]
    assert metrics["mse_pr"] == 12.5


def test_load_cached_bridge_trip_rejects_mismatched_run_request(tmp_path: Path) -> None:
    root = tmp_path / "bridge"
    trip_dir = root / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [37.0],
            "LongitudeDegrees": [-122.0],
        },
    ).to_csv(trip_dir / "bridge_positions.csv", index=False)
    config = BridgeConfig(fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE)
    metrics = {
        "fgo_iters": 4,
        "mse_pr": 12.5,
        **bridge_cache_metadata(max_epochs=300, start_epoch=0, config=config),
    }
    (trip_dir / "bridge_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    assert (
        load_cached_bridge_trip(root, "course/phone", max_epochs=0, start_epoch=0, config=config)
        is None
    )
    assert (
        load_cached_bridge_trip(
            root,
            "course/phone",
            max_epochs=300,
            start_epoch=0,
            config=BridgeConfig(fgo_weight_mode="sin2el"),
        )
        is None
    )

    cached = load_cached_bridge_trip(root, "course/phone", max_epochs=300, start_epoch=0, config=config)

    assert cached is not None
    table, payload = cached
    assert table["UnixTimeMillis"].tolist() == [1000]
    assert payload["bridge_cache"]["max_epochs"] == 300


def test_run_one_bridge_trip_returns_cache_metadata_for_solved_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def positions_table(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "UnixTimeMillis": [1000],
                    "LatitudeDegrees": [37.0],
                    "LongitudeDegrees": [-122.0],
                },
            )

        def states_table(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "UnixTimeMillis": [1000],
                    "X": [1.0],
                    "Y": [2.0],
                    "Z": [3.0],
                },
            )

        def fgo_vd_state_table(self) -> None:
            return None

        def metrics_payload(self) -> dict[str, object]:
            return {
                "fgo_iters": 4,
                "mse_pr": 12.5,
                "n_epochs": 20,
                "selected_source_counts": {"baseline": 20},
            }

    import experiments.build_gsdc2023_bridge_submission as module

    monkeypatch.setattr(module, "validate_raw_gsdc2023_trip", lambda *args, **kwargs: FakeResult())
    config = BridgeConfig(fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE)

    item = run_one_bridge_trip(
        data_root=tmp_path,
        sample_trip_id="course/phone",
        max_epochs=20,
        start_epoch=3,
        config=config,
        bridge_output_root=tmp_path / "bridge",
        resume_existing=False,
    )

    assert item["cached"] is False
    assert item["metrics"]["bridge_cache"]["max_epochs"] == 20
    assert item["metrics"]["bridge_cache"]["start_epoch"] == 3
    assert item["metrics"]["bridge_cache"]["config"]["fgo_weight_mode"] == TAROZ_FGO_WEIGHT_MODE

    exported = json.loads((tmp_path / "bridge" / "course" / "phone" / "bridge_metrics.json").read_text())
    assert exported["bridge_cache"] == item["metrics"]["bridge_cache"]


def test_submission_from_bridge_tables_patches_selected_coordinates() -> None:
    sample = pd.DataFrame(
        {
            "tripId": ["course/a", "course/a", "course/b"],
            "UnixTimeMillis": [1000, 2000, 3000],
            "LatitudeDegrees": [1.0, 1.0, 1.0],
            "LongitudeDegrees": [2.0, 2.0, 2.0],
        },
    )
    bridge_a = pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [37.0, 37.1],
            "LongitudeDegrees": [-122.0, -122.1],
            "SelectedSource": ["baseline", "fgo"],
        },
    )
    bridge_b = pd.DataFrame(
        {
            "UnixTimeMillis": [3000],
            "LatitudeDegrees": [36.5],
            "LongitudeDegrees": [-121.5],
            "SelectedSource": ["baseline"],
        },
    )

    output, summary = submission_from_bridge_tables(sample, {"course/a": bridge_a, "course/b": bridge_b})

    assert ordered_trip_ids(output) == ["course/a", "course/b"]
    assert output["LatitudeDegrees"].tolist() == [37.0, 37.1, 36.5]
    assert output["LongitudeDegrees"].tolist() == [-122.0, -122.1, -121.5]
    assert summary["patched_rows"] == 3
    assert summary["missing_rows"] == 0
    assert summary["selected_source_counts"] == {"baseline": 2, "fgo": 1}
    assert summary["coordinate_sanity_pass"] is True


def test_submission_from_bridge_tables_rejects_missing_timestamp_unless_partial() -> None:
    sample = pd.DataFrame(
        {
            "tripId": ["course/a", "course/a"],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [37.4, 37.4],
            "LongitudeDegrees": [-122.2, -122.2],
        },
    )
    bridge = pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [37.0],
            "LongitudeDegrees": [-122.0],
        },
    )

    with pytest.raises(ValueError, match="missing 1 sample timestamp"):
        submission_from_bridge_tables(sample, {"course/a": bridge})

    output, summary = submission_from_bridge_tables(sample, {"course/a": bridge}, allow_partial=True)

    assert output["LatitudeDegrees"].tolist() == [37.0, 37.4]
    assert output["LongitudeDegrees"].tolist() == [-122.0, -122.2]
    assert summary["patched_rows"] == 1
    assert summary["missing_rows"] == 1


def test_submission_from_bridge_tables_can_interpolate_missing_timestamp() -> None:
    sample = pd.DataFrame(
        {
            "tripId": ["course/a", "course/a", "course/a"],
            "UnixTimeMillis": [1000, 1500, 2000],
            "LatitudeDegrees": [37.4, 37.4, 37.4],
            "LongitudeDegrees": [-122.2, -122.2, -122.2],
        },
    )
    bridge = pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [37.0, 38.0],
            "LongitudeDegrees": [-122.0, -124.0],
            "SelectedSource": ["baseline", "fgo"],
        },
    )

    output, summary = submission_from_bridge_tables(
        sample,
        {"course/a": bridge},
        interpolate_missing=True,
    )

    assert output["LatitudeDegrees"].tolist() == [37.0, 37.5, 38.0]
    assert output["LongitudeDegrees"].tolist() == [-122.0, -123.0, -124.0]
    assert summary["patched_rows"] == 3
    assert summary["interpolated_rows"] == 1
    assert summary["missing_rows"] == 0
    assert summary["selected_source_counts"] == {"baseline": 1, "fgo": 1, "interpolated": 1}


def test_build_config_wires_ct_rbpf_and_dd_carrier_candidates() -> None:
    cfg = build_config(
        _args(
            ct_rbpf_fgo=True,
            ct_rbpf_motion_sigma_m=0.35,
            dd_carrier_fgo=True,
            dd_carrier_base_obs_template="{base}_1hz.obs",
            dd_carrier_require_base_obs_template=True,
            dd_carrier_tow_snap_tolerance_s=0.4,
            dd_carrier_min_dd_pairs=5,
            dd_carrier_smooth_corrections=True,
        ),
    )

    assert cfg.ct_rbpf_fgo_enabled is True
    assert cfg.ct_rbpf_motion_sigma_m == 0.35
    assert cfg.dd_carrier_fgo_enabled is True
    assert cfg.dd_carrier_base_obs_template == "{base}_1hz.obs"
    assert cfg.dd_carrier_require_base_obs_template is True
    assert cfg.dd_carrier_tow_snap_tolerance_s == 0.4
    assert cfg.dd_carrier_min_dd_pairs == 5
    assert cfg.dd_carrier_smooth_corrections is True


@pytest.mark.parametrize(
    ("source", "enabled_field"),
    [
        ("fgo_ct_rbpf", "ct_rbpf_fgo_enabled"),
        ("fgo_dd_carrier", "dd_carrier_fgo_enabled"),
    ],
)
def test_build_config_direct_candidate_sources_auto_enable_required_candidate(
    source: str,
    enabled_field: str,
) -> None:
    cfg = build_config(_args(position_source=source))

    assert getattr(cfg, enabled_field) is True


def test_build_config_taroz_fgo_preset_enables_fgo_bundle() -> None:
    cfg = build_config(_args(taroz_fgo=True))

    assert cfg.weight_mode == "sin2el"
    assert cfg.fgo_weight_mode == "taroz_sn"
    assert cfg.clock_drift_sigma_m == pytest.approx(0.1)
    assert cfg.stop_velocity_sigma_mps == pytest.approx(0.01)
    assert cfg.stop_position_sigma_m == pytest.approx(0.02)
    assert cfg.stop_attitude_sigma_rad == pytest.approx(np.deg2rad(0.1))
    assert cfg.stop_velocity_huber_k == pytest.approx(0.5)
    assert cfg.stop_position_huber_k == pytest.approx(0.5)
    assert cfg.per_type_kernel_enabled is True
    assert cfg.per_type_kernel_motion_enabled is True
    assert cfg.graph_relative_height is True
    assert cfg.relative_height_sigma_m == pytest.approx(0.1)
    assert cfg.relative_height_huber_k == pytest.approx(0.5)
    assert cfg.apply_absolute_height is True
    assert cfg.absolute_height_sigma_m == pytest.approx(0.1)
    assert cfg.absolute_height_huber_k == pytest.approx(0.5)
    assert cfg.apply_observation_mask is True
    assert cfg.observation_min_elevation_deg == pytest.approx(5.0)
    assert cfg.pseudorange_residual_mask_l5_m == pytest.approx(15.0)


def test_build_config_taroz_marupaku_forces_direct_fgo_output() -> None:
    cfg = build_config(_args(taroz_marupaku=True, position_source="gated"))

    assert cfg.position_source == "fgo"
    assert cfg.weight_mode == "taroz_sn"
    assert cfg.fgo_weight_mode == "taroz_sn"
    assert cfg.clock_drift_sigma_m == pytest.approx(0.1)
    assert cfg.stop_velocity_sigma_mps == pytest.approx(0.01)
    assert cfg.stop_attitude_sigma_rad == pytest.approx(np.deg2rad(0.1))
    assert cfg.stop_velocity_huber_k == pytest.approx(0.5)
    assert cfg.per_type_kernel_enabled is True
    assert cfg.per_type_kernel_motion_enabled is True
    assert cfg.graph_relative_height is True
    assert cfg.apply_absolute_height is True
    assert cfg.absolute_height_huber_k == pytest.approx(0.5)
    assert cfg.apply_observation_mask is True
    assert cfg.observation_min_elevation_deg == pytest.approx(5.0)
    assert cfg.apply_imu_prior is True
    assert cfg.imu_frame == "taroz_body"
    assert cfg.imu_sample_dt_mode == "taroz"
    assert cfg.imu_attitude_state is True
    assert cfg.imu_attitude_sigma_rad == pytest.approx(0.0005)
    assert cfg.imu_preintegration_velocity_noise_mps_sqrt_hz == pytest.approx(0.05)
    assert cfg.imu_preintegration_attitude_noise_rad_sqrt_hz == pytest.approx(0.001)
    assert cfg.imu_diagonal_covariance is True
    assert cfg.imu_preintegration_covariance is True
    assert cfg.imu_factor_use_next_bias is True
    assert cfg.imu_bias_between_sample_count_scaling is True
    assert cfg.imu_accel_bias_state is True
    assert cfg.taroz_imu_noise_enabled is True
    assert cfg.imu_velocity_sigma_mps == pytest.approx(0.025)
    assert cfg.imu_acc_sigma_mps2_sqrt_hz == pytest.approx(0.05)
    assert cfg.imu_gyro_sigma_radps_sqrt_hz == pytest.approx(0.001)


def test_run_bridge_submission_rejects_taroz_marupaku_with_phone_aware(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text(
        "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n"
        "course/pixel5,1000,37.0,-122.0\n",
        encoding="utf-8",
    )
    args = _args(
        taroz_marupaku=True,
        taroz_phone_aware=True,
        sample_submission=sample_path,
        data_root=tmp_path,
        output=tmp_path / "submission.csv",
        bridge_output_root=None,
        jobs=1,
        trip=[],
        limit=0,
        allow_partial=False,
        interpolate_missing=False,
        max_epochs=0,
        start_epoch=0,
        resume_existing=False,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_bridge_submission(args)


def test_apply_taroz_phone_aware_preset_uses_gnss_only_for_pixel() -> None:
    cfg = apply_taroz_phone_aware_preset(BridgeConfig(stop_velocity_sigma_mps=3.0), "test/course/pixel5")

    assert use_taroz_gnss_only_for_phone("pixel5") is True
    assert cfg.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert cfg.clock_drift_sigma_m == pytest.approx(0.1)
    assert cfg.clock_use_average_drift is True
    assert cfg.per_type_kernel_enabled is True
    assert cfg.per_type_kernel_motion_enabled is True
    assert cfg.stop_velocity_sigma_mps == 0.0
    assert cfg.stop_attitude_sigma_rad == 0.0
    assert cfg.graph_relative_height is False


def test_apply_taroz_phone_aware_preset_uses_weights_only_for_non_pixel() -> None:
    base = BridgeConfig(clock_drift_sigma_m=1.0, fgo_huber_k_pr=0.0)

    cfg = apply_taroz_phone_aware_preset(base, "test/course/sm-a205u")

    assert use_taroz_gnss_only_for_phone("sm-a205u") is False
    assert cfg.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert cfg.clock_drift_sigma_m == pytest.approx(1.0)
    assert cfg.clock_use_average_drift is None
    assert cfg.fgo_huber_k_pr == 0.0
    assert cfg.fgo_huber_k_doppler == 0.0
    assert cfg.fgo_huber_k_tdcp == 0.0
    assert cfg.per_type_kernel_enabled is False


def test_build_config_wires_factor_huber_thresholds() -> None:
    cfg = build_config(
        _args(
            fgo_huber_k_pr=0.1,
            fgo_huber_k_doppler=0.4,
            fgo_huber_k_tdcp=0.2,
        ),
    )

    assert cfg.fgo_huber_k_pr == pytest.approx(0.1)
    assert cfg.fgo_huber_k_doppler == pytest.approx(0.4)
    assert cfg.fgo_huber_k_tdcp == pytest.approx(0.2)


def test_build_config_wires_taroz_fgo_candidates() -> None:
    cfg = build_config(
        _args(
            taroz_fgo_candidates=True,
            taroz_fgo_candidate_sources="fgo_taroz_pr,fgo_taroz_pr_d_l",
        ),
    )

    assert cfg.taroz_fgo_candidate_enabled is True
    assert cfg.taroz_fgo_candidate_sources == ("fgo_taroz_pr", "fgo_taroz_pr_d_l")


def test_build_config_wires_fgo_gate_gap_floor() -> None:
    cfg = build_config(_args(gate_fgo_baseline_gap_p95_floor_m=20.0))

    assert cfg.gate_fgo_baseline_gap_p95_floor_m == pytest.approx(20.0)
