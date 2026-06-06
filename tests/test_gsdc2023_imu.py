import numpy as np
import pandas as pd

from experiments.evaluate import lla_to_ecef
from experiments.gsdc2023_imu import (
    GYRO_TIME_OFFSET_MS,
    IMU_GRAVITY_MPS2,
    IMU_MOUNTING_ANGLE_RAD,
    IMU_TAROZ_BODY_DELTA_FRAME,
    IMUPreintegration,
    IMUMeasurements,
    ProcessedIMU,
    ecef_delta_from_enu_delta,
    eul_xyz_to_rotm,
    gtsam_rzryrx_to_rotm,
    imu_preintegration_gravity_segment,
    imu_preintegration_segment,
    load_device_imu_measurements,
    load_taroz_imu_preintegration_csv,
    preintegrate_processed_imu,
    process_device_imu,
)


def _rot_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_load_device_imu_measurements_uses_injected_reader(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    (trip / "device_imu.csv").write_text("stub\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            {
                "MessageType": "UncalAccel",
                "utcTimeMillis": 2000,
                "elapsedRealtimeNanos": 2.0e9,
                "MeasurementX": 2.0,
                "MeasurementY": 0.0,
                "MeasurementZ": 9.8,
                "BiasX": 0.2,
                "BiasY": 0.0,
                "BiasZ": 0.0,
            },
            {
                "MessageType": "UncalAccel",
                "utcTimeMillis": 1000,
                "elapsedRealtimeNanos": 1.0e9,
                "MeasurementX": 1.0,
                "MeasurementY": 0.0,
                "MeasurementZ": 9.8,
                "BiasX": 0.1,
                "BiasY": 0.0,
                "BiasZ": 0.0,
            },
            {
                "MessageType": "UncalAccel",
                "utcTimeMillis": 2000,
                "elapsedRealtimeNanos": 2.1e9,
                "MeasurementX": 9.0,
                "MeasurementY": 0.0,
                "MeasurementZ": 9.8,
                "BiasX": 0.9,
                "BiasY": 0.0,
                "BiasZ": 0.0,
            },
            {
                "MessageType": "UncalGyro",
                "utcTimeMillis": 1000,
                "elapsedRealtimeNanos": 1.0e9,
                "MeasurementX": 0.01,
                "MeasurementY": 0.0,
                "MeasurementZ": 0.0,
                "BiasX": 0.001,
                "BiasY": 0.0,
                "BiasZ": 0.0,
            },
            {
                "MessageType": "UncalMag",
                "utcTimeMillis": 1000,
                "elapsedRealtimeNanos": 1.0e9,
                "MeasurementX": 30.0,
                "MeasurementY": -20.0,
                "MeasurementZ": 5.0,
                "BiasX": 1.0,
                "BiasY": 2.0,
                "BiasZ": 3.0,
            },
        ],
    )
    calls: list[dict[str, object]] = []

    def read_csv_fn(_path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("nrows") == 0:
            return frame.head(0)
        usecols = kwargs.get("usecols")
        return frame.loc[:, usecols] if usecols is not None else frame.copy()

    acc, gyro, mag = load_device_imu_measurements(trip, read_csv_fn=read_csv_fn)

    assert acc is not None and gyro is not None and mag is not None
    assert calls[0] == {"nrows": 0}
    assert "usecols" in calls[1]
    np.testing.assert_allclose(acc.times_ms, [1000.0, 2000.0])
    np.testing.assert_allclose(acc.xyz[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(gyro.bias[:, 0], [0.001])
    np.testing.assert_allclose(mag.bias, [[1.0, 2.0, 3.0]])


def test_load_taroz_imu_preintegration_csv_aligns_and_reads_extended_blocks(tmp_path):
    path = tmp_path / "phone_data_imu_preintegration.csv"
    jac = np.arange(1.0, 10.0, dtype=np.float64)
    cov = np.eye(9, dtype=np.float64).reshape(-1)
    row = {
        "epoch_index": 2,
        "utcTimeMillis": 2000,
        "next_epoch_index": 3,
        "nextUtcTimeMillis": 3000,
        "sample_count": 5,
        "graph_dt_s": 1.0,
        "preintegrated_dt_s": 1.02,
        "delta_r_x": 0.1,
        "delta_r_y": 0.2,
        "delta_r_z": 0.3,
        "delta_p_x": 1.0,
        "delta_p_y": 2.0,
        "delta_p_z": 3.0,
        "delta_v_x": 4.0,
        "delta_v_y": 5.0,
        "delta_v_z": 6.0,
        "gravity_x": 0.0,
        "gravity_y": 0.0,
        "gravity_z": -9.80665,
    }
    for prefix in (
        "delta_p_bias_accel_jac",
        "delta_v_bias_accel_jac",
        "delta_p_bias_gyro_jac",
        "delta_v_bias_gyro_jac",
        "delta_r_bias_gyro_jac",
    ):
        for idx, value in enumerate(jac):
            row[f"{prefix}_{idx // 3}_{idx % 3}"] = value
    for idx, value in enumerate(cov):
        row[f"preint_meas_cov_{idx // 9}_{idx % 9}"] = value
    pd.DataFrame([row]).to_csv(path, index=False)

    preint = load_taroz_imu_preintegration_csv(
        path,
        epoch_times_ms=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
    )

    assert preint.delta_frame == IMU_TAROZ_BODY_DELTA_FRAME
    np.testing.assert_array_equal(preint.sample_count, np.array([0, 5], dtype=np.int32))
    assert np.isnan(preint.delta_t_s[0])
    assert preint.delta_t_s[1] == 1.02
    np.testing.assert_allclose(preint.delta_p_body[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(preint.delta_v_body[1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(preint.delta_angle_rad[1], [0.1, 0.2, 0.3])
    assert preint.delta_p_bias_accel_jac is not None
    np.testing.assert_allclose(preint.delta_p_bias_accel_jac[1], jac.reshape(3, 3))
    assert preint.delta_angle_bias_gyro_jac is not None
    np.testing.assert_allclose(preint.delta_angle_bias_gyro_jac[1], jac.reshape(3, 3))
    assert preint.preint_meas_cov is not None
    np.testing.assert_allclose(preint.preint_meas_cov[1], np.eye(9))
    assert preint.gravity_ecef is None
    assert preint.gravity_nav is not None
    np.testing.assert_allclose(preint.gravity_nav[1], [0.0, 0.0, -9.80665])


def test_process_device_imu_without_elapsed_uses_utc_offsets():
    acc = IMUMeasurements(
        times_ms=np.array([1000.0, 2000.0, 3000.0]),
        elapsed_ns=np.array([1.0e9, 2.0e9, 3.0e9]),
        xyz=np.tile(np.array([0.0, 0.0, IMU_GRAVITY_MPS2]), (3, 1)),
        bias=np.zeros((3, 3), dtype=np.float64),
    )
    gyro = IMUMeasurements(
        times_ms=np.array([1000.0, 2000.0, 3000.0]),
        elapsed_ns=np.array([1.0e9, 2.0e9, 3.0e9]),
        xyz=np.zeros((3, 3), dtype=np.float64),
        bias=np.ones((3, 3), dtype=np.float64) * 0.01,
    )

    acc_proc, gyro_proc, idx_stop = process_device_imu(
        acc,
        gyro,
        np.array([1000.0, 2000.0, 3000.0]),
        None,
    )

    np.testing.assert_allclose(acc_proc.times_ms, gyro.times_ms + GYRO_TIME_OFFSET_MS)
    np.testing.assert_allclose(gyro_proc.times_ms, gyro.times_ms + GYRO_TIME_OFFSET_MS)
    assert acc_proc.sync_coefficient == 1.0
    assert gyro_proc.sync_coefficient == 1.0
    assert idx_stop.tolist() == [True, True, True]


def test_preintegrate_processed_imu_ecef_stationary_gravity():
    times_ms = np.array([0.0, 500.0, 1000.0])
    rot_body_sensor = eul_xyz_to_rotm(IMU_MOUNTING_ANGLE_RAD.reshape(1, 3))[0]
    stationary_acc_sensor = rot_body_sensor.T @ np.array([0.0, 0.0, IMU_GRAVITY_MPS2], dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(stationary_acc_sensor, (times_ms.size, 1)),
        dt_s=np.full(times_ms.size, 0.5),
        norm_3d=np.full(times_ms.size, IMU_GRAVITY_MPS2),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.zeros((times_ms.size, 3), dtype=np.float64),
        dt_s=np.full(times_ms.size, 0.5),
        norm_3d=np.zeros(times_ms.size),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    reference_xyz = np.tile(origin_xyz.reshape(1, 3), (times_ms.size, 1))

    preint = preintegrate_processed_imu(acc, gyro, times_ms, delta_frame="ecef", reference_xyz_ecef=reference_xyz)

    assert preint.delta_frame == "ecef"
    np.testing.assert_allclose(preint.delta_v_body, 0.0, atol=1e-9)
    np.testing.assert_allclose(preint.delta_p_body, 0.0, atol=1e-9)


def test_preintegrate_processed_imu_taroz_body_exports_body_deltas_and_gravity():
    times_ms = np.array([0.0, 500.0, 1000.0])
    rot_body_sensor = gtsam_rzryrx_to_rotm(IMU_MOUNTING_ANGLE_RAD.reshape(1, 3))[0]
    stationary_acc_sensor = rot_body_sensor.T @ np.array([0.0, 0.0, IMU_GRAVITY_MPS2], dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(stationary_acc_sensor, (times_ms.size, 1)),
        dt_s=np.full(times_ms.size, 0.5),
        norm_3d=np.full(times_ms.size, IMU_GRAVITY_MPS2),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.zeros((times_ms.size, 3), dtype=np.float64),
        dt_s=np.full(times_ms.size, 0.5),
        norm_3d=np.zeros(times_ms.size),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    reference_xyz = np.tile(origin_xyz.reshape(1, 3), (times_ms.size, 1))

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        times_ms,
        delta_frame=IMU_TAROZ_BODY_DELTA_FRAME,
        reference_xyz_ecef=reference_xyz,
    )

    assert preint.delta_frame == IMU_TAROZ_BODY_DELTA_FRAME
    expected_delta_v = np.tile(np.array([0.0, 0.0, IMU_GRAVITY_MPS2 * 0.5]), (2, 1))
    expected_delta_p = np.tile(np.array([0.0, 0.0, 0.5 * IMU_GRAVITY_MPS2 * 0.5 * 0.5]), (2, 1))
    np.testing.assert_allclose(preint.delta_v_body, expected_delta_v, atol=1e-9)
    np.testing.assert_allclose(preint.delta_p_body, expected_delta_p, atol=1e-9)
    assert preint.gravity_ecef is not None
    expected_gravity = ecef_delta_from_enu_delta(np.array([[0.0, 0.0, -IMU_GRAVITY_MPS2]]), origin_xyz)[0]
    np.testing.assert_allclose(preint.gravity_ecef, np.tile(expected_gravity, (2, 1)), atol=1e-9)
    gravity_segment = imu_preintegration_gravity_segment(preint, 0, 3)
    assert gravity_segment is not None
    np.testing.assert_allclose(gravity_segment, preint.gravity_ecef)


def test_preintegrate_processed_imu_taroz_sample_dt_uses_processed_dt_s():
    times_ms = np.array([0.0, 500.0, 1000.0, 1500.0])
    dt_s = np.full(times_ms.size, 0.5, dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([2.0, 0.0, 0.0]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 2.0),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([0.0, 0.0, 0.1]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 0.1),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        np.array([0.0, 1000.0, 1500.0]),
        sample_dt_mode="taroz",
    )

    np.testing.assert_allclose(preint.delta_t_s, np.array([1.5, 1.0]))
    np.testing.assert_array_equal(preint.sample_count, np.array([3, 2], dtype=np.int32))

    eye3 = np.eye(3, dtype=np.float64)
    r1 = _rot_z(0.05)
    r2 = _rot_z(0.10)
    acc_vec = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    expected_v0 = 0.5 * (eye3 + r1 + r2) @ acc_vec
    expected_p0 = (0.625 * eye3 + 0.375 * r1 + 0.125 * r2) @ acc_vec
    expected_v1 = 0.5 * (eye3 + r1) @ acc_vec
    expected_p1 = (0.375 * eye3 + 0.125 * r1) @ acc_vec

    np.testing.assert_allclose(preint.delta_v_body[0], expected_v0)
    np.testing.assert_allclose(preint.delta_p_body[0], expected_p0)
    np.testing.assert_allclose(preint.delta_v_body[1], expected_v1)
    np.testing.assert_allclose(preint.delta_p_body[1], expected_p1)
    np.testing.assert_allclose(preint.delta_angle_rad[:, 2], np.array([0.15, 0.1]))
    assert preint.delta_p_bias_accel_jac is not None
    assert preint.delta_v_bias_accel_jac is not None
    assert preint.delta_p_bias_gyro_jac is not None
    assert preint.delta_v_bias_gyro_jac is not None
    assert preint.delta_angle_bias_gyro_jac is not None
    np.testing.assert_allclose(preint.delta_v_bias_accel_jac[0], 0.5 * (eye3 + r1 + r2))
    np.testing.assert_allclose(preint.delta_p_bias_accel_jac[0], 0.625 * eye3 + 0.375 * r1 + 0.125 * r2)
    np.testing.assert_allclose(preint.delta_v_bias_accel_jac[1], 0.5 * (eye3 + r1))
    np.testing.assert_allclose(preint.delta_p_bias_accel_jac[1], 0.375 * eye3 + 0.125 * r1)
    assert float(np.linalg.norm(preint.delta_p_bias_gyro_jac)) > 0.0
    assert float(np.linalg.norm(preint.delta_v_bias_gyro_jac)) > 0.0
    np.testing.assert_allclose(preint.delta_angle_bias_gyro_jac[:, 2, 2], np.array([1.5, 1.0]))


def test_imu_preintegration_segment_masks_invalid_intervals():
    preint = IMUPreintegration(
        epoch_times_ms=np.array([0.0, 1000.0, 2000.0, 3000.0]),
        delta_t_s=np.array([1.0, 0.0, 1.0]),
        delta_v_body=np.array([[0.1, 0.2, 0.3], [9.0, 9.0, 9.0], [0.4, 0.5, 0.6]], dtype=np.float64),
        delta_p_body=np.array([[1.0, 2.0, 3.0], [8.0, 8.0, 8.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        delta_angle_rad=np.zeros((3, 3), dtype=np.float64),
        sample_count=np.array([5, 0, 7], dtype=np.int32),
    )

    delta_p, delta_v, count = imu_preintegration_segment(preint, 0, 4)

    assert count == 2
    assert delta_p is not None and delta_v is not None
    np.testing.assert_allclose(delta_p[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(delta_v[2], [0.4, 0.5, 0.6])
    assert np.isnan(delta_p[1]).all()
    assert np.isnan(delta_v[1]).all()
