from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from experiments.compare_gsdc2023_taroz_imu_state import (
    compare_native_to_matlab_state,
    compare_taroz_imu_state_tables,
    finite_delta_summary,
    infer_origin_ecef_from_first_pair,
    run_comparison,
    source_prefix,
    taroz_imu_state_delta_summary,
    taroz_preprocessing_origin_ecef,
)
from experiments.gsdc2023_imu import enu_to_ecef_relative


def test_compare_native_to_matlab_state_joins_and_reports_enu_delta() -> None:
    origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    matlab_enu = np.array(
        [
            [10.0, 2.0, 1.0],
            [11.0, 4.0, 1.5],
        ],
        dtype=np.float64,
    )
    native_enu = matlab_enu + np.array([0.5, -0.25, 0.1], dtype=np.float64)
    native_xyz = enu_to_ecef_relative(native_enu, origin)
    bridge_states = pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "FgoEcefXMeters": native_xyz[:, 0],
            "FgoEcefYMeters": native_xyz[:, 1],
            "FgoEcefZMeters": native_xyz[:, 2],
        },
    )
    matlab_state = pd.DataFrame(
        {
            "epoch_index": [1, 2],
            "utcTimeMillis": [1000, 2000],
            "position_x": matlab_enu[:, 0],
            "position_y": matlab_enu[:, 1],
            "position_z": matlab_enu[:, 2],
        },
    )

    comparison = compare_native_to_matlab_state(
        bridge_states,
        matlab_state,
        origin_ecef=origin,
        source="fgo",
    )
    summary = finite_delta_summary(comparison)

    np.testing.assert_allclose(comparison[["delta_east_m", "delta_north_m", "delta_up_m"]], [[0.5, -0.25, 0.1]] * 2)
    assert summary["finite_rows"] == 2
    assert summary["mean_horizontal_m"] == pytest.approx(np.hypot(0.5, -0.25))
    assert summary["max_3d_m"] == pytest.approx(np.linalg.norm([0.5, -0.25, 0.1]))


def test_infer_origin_ecef_from_first_pair_maps_matlab_enu_to_native_ecef() -> None:
    true_origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    matlab_enu = np.array([120.0, -40.0, 5.0], dtype=np.float64)
    native_xyz = enu_to_ecef_relative(matlab_enu.reshape(1, 3), true_origin)[0]

    inferred = infer_origin_ecef_from_first_pair(native_xyz, matlab_enu)
    remapped = enu_to_ecef_relative(matlab_enu.reshape(1, 3), inferred)[0]

    np.testing.assert_allclose(remapped, native_xyz, atol=1e-6)


def test_taroz_preprocessing_origin_uses_first_valid_unique_wls(tmp_path) -> None:
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "utcTimeMillis": [2000, 1000, 1000, 3000],
            "BiasUncertaintyNanos": [2.0e4, 3.0, 3.0, 4.0],
            "WlsPositionXEcefMeters": [1.0, 10.0, 20.0, 30.0],
            "WlsPositionYEcefMeters": [2.0, 11.0, 21.0, 31.0],
            "WlsPositionZEcefMeters": [3.0, 12.0, 22.0, 32.0],
        },
    ).to_csv(trip_dir / "device_gnss.csv", index=False)

    origin = taroz_preprocessing_origin_ecef(trip_dir)

    np.testing.assert_allclose(origin, np.array([10.0, 11.0, 12.0]))


def test_source_prefix_accepts_native_names() -> None:
    assert source_prefix("fgo") == "Fgo"
    assert source_prefix("fgo_vd") == "FgoVd"
    assert source_prefix("FgoVd") == "FgoVd"
    assert source_prefix("raw_wls") == "RawWls"
    with pytest.raises(ValueError, match="unsupported state source"):
        source_prefix("bad")


def test_compare_taroz_imu_state_tables_summarizes_full_state_groups() -> None:
    matlab = pd.DataFrame(
        {
            "epoch_index": [1, 2],
            "utcTimeMillis": [1000, 2000],
            "position_x": [1.0, 2.0],
            "position_y": [3.0, 4.0],
            "position_z": [5.0, 6.0],
            "roll": [0.1, 0.2],
            "pitch": [0.3, 0.4],
            "yaw": [0.5, 0.6],
            "velocity_x": [0.0, 1.0],
            "velocity_y": [2.0, 3.0],
            "velocity_z": [4.0, 5.0],
            "clock_bias_m_0": [10.0, 11.0],
            "clock_drift_mps": [0.01, 0.02],
            "bias_acc_x": [0.1, 0.2],
            "bias_acc_y": [0.3, 0.4],
            "bias_acc_z": [0.5, 0.6],
            "bias_gyro_x": [0.001, 0.002],
            "bias_gyro_y": [0.003, 0.004],
            "bias_gyro_z": [0.005, 0.006],
        }
    )
    native = matlab.copy()
    native[["position_x", "position_y", "position_z"]] += np.array([0.1, -0.2, 0.3])
    native[["velocity_x", "velocity_y", "velocity_z"]] += np.array([0.01, 0.02, -0.03])
    native["clock_bias_m_0"] += 0.5
    native["bias_gyro_z"] -= 0.0001

    comparison = compare_taroz_imu_state_tables(native, matlab)
    summary = taroz_imu_state_delta_summary(comparison)

    np.testing.assert_allclose(comparison[["delta_position_x", "delta_position_y", "delta_position_z"]], [[0.1, -0.2, 0.3]] * 2)
    assert summary["matched_rows"] == 2
    assert summary["groups"]["position_m"]["mean_norm"] == pytest.approx(np.linalg.norm([0.1, -0.2, 0.3]))
    assert summary["groups"]["velocity_mps"]["component_max_abs"] == pytest.approx(0.03)
    assert summary["groups"]["clock_bias_m"]["mean_norm"] == pytest.approx(0.5)
    assert summary["groups"]["bias_gyro_radps"]["component_max_abs"] == pytest.approx(0.0001)


def test_run_comparison_accepts_native_taroz_imu_state_mode(tmp_path) -> None:
    matlab = pd.DataFrame(
        {
            "utcTimeMillis": [1000, 2000],
            "position_x": [1.0, 2.0],
            "position_y": [3.0, 4.0],
            "position_z": [5.0, 6.0],
            "velocity_x": [0.0, 1.0],
            "velocity_y": [2.0, 3.0],
            "velocity_z": [4.0, 5.0],
        }
    )
    native = matlab.copy()
    native[["position_x", "position_y", "position_z"]] += np.array([0.1, 0.0, -0.2])
    native["velocity_z"] += 0.03
    native_path = tmp_path / "native_imu_state.csv"
    matlab_path = tmp_path / "matlab_imu_state.csv"
    output_path = tmp_path / "delta.csv"
    summary_path = tmp_path / "summary.json"
    native.to_csv(native_path, index=False)
    matlab.to_csv(matlab_path, index=False)

    summary = run_comparison(
        argparse.Namespace(
            bridge_states=None,
            native_imu_state=native_path,
            matlab_imu_state=matlab_path,
            source="fgo",
            origin_mode="first_pair",
            origin_ecef=None,
            origin_llh_deg=None,
            initial_origin_ecef=None,
            data_root=tmp_path,
            trip="train/course/phone",
            output=output_path,
            summary=summary_path,
        )
    )

    assert summary["mode"] == "taroz_imu_state"
    assert summary["delta_stats"]["matched_rows"] == 2
    assert summary["delta_stats"]["groups"]["position_m"]["mean_norm"] == pytest.approx(np.linalg.norm([0.1, 0.0, -0.2]))
    assert summary["delta_stats"]["groups"]["velocity_mps"]["component_max_abs"] == pytest.approx(0.03)
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary == summary
    written_delta = pd.read_csv(output_path)
    np.testing.assert_allclose(written_delta["delta_velocity_z"], [0.03, 0.03])
