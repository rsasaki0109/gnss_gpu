import numpy as np
import pandas as pd

from experiments.evaluate import lla_to_ecef
from experiments.gsdc2023_imu import (
    GYRO_TIME_OFFSET_MS,
    IMU_GRAVITY_MPS2,
    IMU_MOUNTING_ANGLE_RAD,
    IMUPreintegration,
    IMUMeasurements,
    ProcessedIMU,
    eul_xyz_to_rotm,
    imu_preintegration_segment,
    load_device_imu_measurements,
    preintegrate_processed_imu,
    process_device_imu,
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
