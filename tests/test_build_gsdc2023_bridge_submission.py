from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.build_gsdc2023_bridge_submission import (
    build_config,
    bridge_output_dir,
    bridge_trip_id,
    load_cached_bridge_trip,
    ordered_trip_ids,
    submission_from_bridge_tables,
)


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
        "fgo_raw_wls_proxy_rescue_mse_ratio_max": 1.20,
        "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": 1.25,
        "fgo_raw_wls_proxy_rescue_quality_delta_max": -0.35,
        "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bridge_trip_id_accepts_sample_or_test_trip_id() -> None:
    assert bridge_trip_id("course/phone") == "test/course/phone"
    assert bridge_trip_id("test/course/phone") == "test/course/phone"
    assert str(bridge_output_dir(Path("/tmp/bridge"), "test/course/phone")).endswith("bridge/course/phone")


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
