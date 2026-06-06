from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import gsdc2023_raw_bridge as raw_bridge
from experiments.export_bridge_fgo_vd_state_as_taroz_imu_state import (
    bridge_fgo_vd_state_to_taroz_imu_state,
    gtsam_rotm_to_rzryrx,
    infer_origin_ecef_from_bridge_fgo_vd_state,
)
from experiments.gsdc2023_imu import (
    enu_to_ecef_relative,
    gtsam_rzryrx_to_rotm,
    rotm_to_rotvec,
)


def test_gtsam_rotm_to_rzryrx_round_trips_rotation() -> None:
    rpy = np.array([[0.1, -0.2, 0.3], [-0.4, 0.25, -0.6]], dtype=np.float64)
    recovered = gtsam_rotm_to_rzryrx(gtsam_rzryrx_to_rotm(rpy))

    np.testing.assert_allclose(recovered, rpy, atol=1e-12)


def test_bridge_fgo_vd_state_to_taroz_imu_state_round_trips_loader(tmp_path) -> None:
    origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    pos_enu = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    vel_enu = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    rpy = np.array([[0.1, -0.2, 0.3], [0.2, -0.1, 0.4]], dtype=np.float64)
    pos_ecef = enu_to_ecef_relative(pos_enu, origin)
    vel_ecef = enu_to_ecef_relative(vel_enu, origin) - origin
    enu_basis_ecef = enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin) - origin
    rot_ecef_enu = enu_basis_ecef.T
    rot_ecef_body = np.einsum("ij,njk->nik", rot_ecef_enu, gtsam_rzryrx_to_rotm(rpy))
    rotvec = np.vstack([rotm_to_rotvec(rot) for rot in rot_ecef_body])
    bridge = pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "FgoVdEcefXMeters": pos_ecef[:, 0],
            "FgoVdEcefYMeters": pos_ecef[:, 1],
            "FgoVdEcefZMeters": pos_ecef[:, 2],
            "FgoVdVelocityXMps": vel_ecef[:, 0],
            "FgoVdVelocityYMps": vel_ecef[:, 1],
            "FgoVdVelocityZMps": vel_ecef[:, 2],
            "FgoVdClockBiasMeters0": [10.0, 11.0],
            "FgoVdClockBiasMeters1": [20.0, 21.0],
            "FgoVdClockDriftMps": [0.01, 0.02],
            "FgoVdStateExtra0": rotvec[:, 0],
            "FgoVdStateExtra1": rotvec[:, 1],
            "FgoVdStateExtra2": rotvec[:, 2],
            "FgoVdStateExtra3": [0.01, 0.02],
            "FgoVdStateExtra4": [0.03, 0.04],
            "FgoVdStateExtra5": [0.05, 0.06],
            "FgoVdStateExtra6": [0.001, 0.002],
            "FgoVdStateExtra7": [0.003, 0.004],
            "FgoVdStateExtra8": [0.005, 0.006],
        }
    )
    template = pd.DataFrame({"epoch_index": [1, 2], "utcTimeMillis": [1000, 2000]})

    out = bridge_fgo_vd_state_to_taroz_imu_state(
        bridge,
        origin_ecef=origin,
        template_state=template,
    )

    np.testing.assert_allclose(out[["position_x", "position_y", "position_z"]], pos_enu, atol=1e-9)
    np.testing.assert_allclose(out[["velocity_x", "velocity_y", "velocity_z"]], vel_enu, atol=1e-12)
    np.testing.assert_allclose(out[["roll", "pitch", "yaw"]], rpy, atol=1e-12)

    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    (trip_dir / "device_gnss.csv").write_text(
        "\n".join(
            [
                "utcTimeMillis,BiasUncertaintyNanos,WlsPositionXEcefMeters,WlsPositionYEcefMeters,WlsPositionZEcefMeters",
                f"1000,10,{origin[0]},{origin[1]},{origin[2]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seed_csv = tmp_path / "phone_data_imu_state.csv"
    out.to_csv(seed_csv, index=False)
    batch = raw_bridge.TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.zeros((2, 1, 3), dtype=np.float64),
        pseudorange=np.zeros((2, 1), dtype=np.float64),
        weights=np.zeros((2, 1), dtype=np.float64),
        kaggle_wls=np.zeros((2, 3), dtype=np.float64),
        truth=np.full((2, 3), np.nan, dtype=np.float64),
        max_sats=1,
        has_truth=False,
        n_clock=2,
    )

    loaded = raw_bridge._load_taroz_fgo_seed_state(seed_csv, batch, trip_dir=trip_dir)

    np.testing.assert_allclose(loaded[:, :3], pos_ecef, atol=1e-9)
    np.testing.assert_allclose(loaded[:, 3:6], vel_ecef, atol=1e-12)
    np.testing.assert_allclose(loaded[:, 6:8], [[10.0, 20.0], [11.0, 21.0]])
    np.testing.assert_allclose(loaded[:, 8], [0.01, 0.02])
    np.testing.assert_allclose(loaded[:, 9:12], rotvec, atol=1e-12)
    np.testing.assert_allclose(loaded[:, 12:15], [[0.01, 0.03, 0.05], [0.02, 0.04, 0.06]])
    np.testing.assert_allclose(loaded[:, 15:18], [[0.001, 0.003, 0.005], [0.002, 0.004, 0.006]])


def test_bridge_fgo_vd_state_to_taroz_imu_state_uses_split_pose_position() -> None:
    origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    x_enu = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    pose_enu = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
    vel_enu = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    rpy = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    x_ecef = enu_to_ecef_relative(x_enu, origin)
    pose_ecef = enu_to_ecef_relative(pose_enu, origin)
    vel_ecef = enu_to_ecef_relative(vel_enu, origin) - origin
    enu_basis_ecef = enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin) - origin
    rot_ecef_enu = enu_basis_ecef.T
    rotvec = np.vstack([rotm_to_rotvec(rot_ecef_enu @ rot) for rot in gtsam_rzryrx_to_rotm(rpy)])
    bridge = pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "FgoVdEcefXMeters": x_ecef[:, 0],
            "FgoVdEcefYMeters": x_ecef[:, 1],
            "FgoVdEcefZMeters": x_ecef[:, 2],
            "FgoVdVelocityXMps": vel_ecef[:, 0],
            "FgoVdVelocityYMps": vel_ecef[:, 1],
            "FgoVdVelocityZMps": vel_ecef[:, 2],
            "FgoVdClockBiasMeters0": [10.0],
            "FgoVdClockDriftMps": [0.01],
            "FgoVdStateExtra0": pose_ecef[:, 0],
            "FgoVdStateExtra1": pose_ecef[:, 1],
            "FgoVdStateExtra2": pose_ecef[:, 2],
            "FgoVdStateExtra3": rotvec[:, 0],
            "FgoVdStateExtra4": rotvec[:, 1],
            "FgoVdStateExtra5": rotvec[:, 2],
            "FgoVdStateExtra6": [0.01],
            "FgoVdStateExtra7": [0.02],
            "FgoVdStateExtra8": [0.03],
            "FgoVdStateExtra9": [0.001],
            "FgoVdStateExtra10": [0.002],
            "FgoVdStateExtra11": [0.003],
        }
    )

    out = bridge_fgo_vd_state_to_taroz_imu_state(bridge, origin_ecef=origin)

    np.testing.assert_allclose(out[["position_x", "position_y", "position_z"]], pose_enu, atol=1e-9)
    np.testing.assert_allclose(out[["roll", "pitch", "yaw"]], rpy, atol=1e-12)
    np.testing.assert_allclose(out[["bias_acc_x", "bias_acc_y", "bias_acc_z"]], [[0.01, 0.02, 0.03]])
    np.testing.assert_allclose(out[["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"]], [[0.001, 0.002, 0.003]])


def test_infer_origin_ecef_from_bridge_fgo_vd_state_uses_split_pose_pair() -> None:
    true_origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    wrong_origin = true_origin + np.array([5.0, -2.0, 1.0], dtype=np.float64)
    pose_enu = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
    x_enu = np.array([[40.0, 50.0, 60.0]], dtype=np.float64)
    pose_ecef = enu_to_ecef_relative(pose_enu, true_origin)
    x_ecef = enu_to_ecef_relative(x_enu, true_origin)
    bridge = pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "FgoVdEcefXMeters": x_ecef[:, 0],
            "FgoVdEcefYMeters": x_ecef[:, 1],
            "FgoVdEcefZMeters": x_ecef[:, 2],
            **{f"FgoVdStateExtra{idx}": [0.0] for idx in range(12)},
        }
    )
    bridge.loc[0, "FgoVdStateExtra0"] = pose_ecef[0, 0]
    bridge.loc[0, "FgoVdStateExtra1"] = pose_ecef[0, 1]
    bridge.loc[0, "FgoVdStateExtra2"] = pose_ecef[0, 2]
    template = pd.DataFrame(
        {
            "utcTimeMillis": [1000],
            "position_x": pose_enu[:, 0],
            "position_y": pose_enu[:, 1],
            "position_z": pose_enu[:, 2],
        }
    )

    inferred = infer_origin_ecef_from_bridge_fgo_vd_state(
        bridge,
        template,
        initial_origin_ecef=wrong_origin,
    )

    np.testing.assert_allclose(inferred, true_origin, atol=1e-6)
