from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gnss_gpu.basin_imu_bridge import (
    CausalBasinImuPredictor,
    PPCImuSamples,
    load_ppc_imu_csv,
)


def _stationary_samples() -> PPCImuSamples:
    tow = np.arange(0.0, 2.01, 0.01)
    accel = np.tile([0.2, -0.1, 9.86], (tow.size, 1))
    gyro = np.tile([0.001, -0.002, 0.003], (tow.size, 1))
    return PPCImuSamples(tow, accel, gyro)


def test_causal_bias_calibration_and_stationary_prediction() -> None:
    predictor = CausalBasinImuPredictor(_stationary_samples())
    assert predictor.calibrate_before(1.0, window_s=1.0) == 100
    result = predictor.predict_interval(
        1.0,
        2.0,
        position_ecef_m=np.array([6_378_137.0, 0.0, 0.0]),
        velocity_ecef_mps=np.zeros(3),
    )
    assert result is not None
    assert result.sample_count >= 99
    np.testing.assert_allclose(result.cv_position_correction_ecef_m, 0.0, atol=1e-9)
    np.testing.assert_allclose(result.delta_velocity_ecef_mps, 0.0, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(result.process_covariance)) > 0.0


def test_interval_rejects_insufficient_imu_coverage() -> None:
    predictor = CausalBasinImuPredictor(_stationary_samples())
    assert (
        predictor.predict_interval(
            2.0,
            3.0,
            position_ecef_m=np.array([6_378_137.0, 0.0, 0.0]),
            velocity_ecef_mps=np.zeros(3),
        )
        is None
    )


def test_ppc_loader_converts_gyro_degrees_to_radians(tmp_path: Path) -> None:
    path = tmp_path / "imu.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "GPS TOW (s)",
                "GPS Week",
                "Acc X (m/s^2)",
                "Acc Y (m/s^2)",
                "Acc Z (m/s^2)",
                "Ang Rate X (deg/s)",
                " Ang Rate Y (deg/s)",
                " Ang Rate Z (deg/s)",
            ]
        )
        writer.writerow([1.0, 2, 0, 0, 9.81, 180, 0, -90])
    samples = load_ppc_imu_csv(path)
    np.testing.assert_allclose(samples.gyro_body_radps[0], [np.pi, 0.0, -np.pi / 2])
