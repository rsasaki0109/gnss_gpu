import math

import numpy as np

from gnss_gpu.ins_ekf import INSEKF, INSConfig


def _aligned_ins() -> INSEKF:
    ins = INSEKF(INSConfig(align_min_static_samples=50))
    for k in range(50):
        ins.feed_imu_for_alignment(
            k * 0.01,
            np.array([0.0, 0.0, 9.81], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
    assert ins.aligned
    ins.initialize_position(np.zeros(3, dtype=np.float64))
    return ins


def _constant_samples(ins: INSEKF, accel: np.ndarray, gyro_dps: np.ndarray, duration_s: float) -> np.ndarray:
    dt = 0.01
    n = int(round(duration_s / dt))
    t0 = float(ins.last_t if ins.last_t is not None else 0.0)
    rows = np.zeros((n + 1, 7), dtype=np.float64)
    for k in range(n + 1):
        rows[k, 0] = t0 + k * dt
        rows[k, 1:4] = accel
        rows[k, 4:7] = gyro_dps
    return rows


def test_static_zero_drift():
    ins = _aligned_ins()
    samples = _constant_samples(
        ins,
        np.array([0.0, 0.0, 9.81], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        5.0,
    )

    ins.propagate(samples)

    np.testing.assert_allclose(ins.p, np.zeros(3), atol=1.0e-6)
    np.testing.assert_allclose(ins.v, np.zeros(3), atol=1.0e-6)


def test_constant_accel():
    ins = _aligned_ins()
    samples = _constant_samples(
        ins,
        np.array([1.0, 0.0, 9.81], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        5.0,
    )

    ins.propagate(samples)

    np.testing.assert_allclose(ins.v, np.array([5.0, 0.0, 0.0]), atol=1.0e-6)
    np.testing.assert_allclose(ins.p, np.array([12.5, 0.0, 0.0]), atol=1.0e-6)


def test_yaw_rotation():
    ins = _aligned_ins()
    samples = _constant_samples(
        ins,
        np.array([0.0, 0.0, 9.81], dtype=np.float64),
        np.array([0.0, 0.0, 10.0], dtype=np.float64),
        5.0,
    )

    ins.propagate(samples)

    assert math.isclose(ins.yaw_rad(), math.radians(50.0), abs_tol=1.0e-6)


def test_static_alignment():
    ins = _aligned_ins()

    assert ins.aligned
    np.testing.assert_allclose(ins.q, np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-6)
    np.testing.assert_allclose(ins.b_g, np.zeros(3), atol=1.0e-12)


def test_position_update_reduces_p():
    ins = _aligned_ins()
    ins.p = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    before_diag = np.diag(ins.P[0:3, 0:3]).copy()

    ins.update_position_enu(np.zeros(3, dtype=np.float64), (0.1, 0.1, 0.1))

    assert np.linalg.norm(ins.p) < 10.0
    assert np.all(np.diag(ins.P[0:3, 0:3]) < before_diag)
