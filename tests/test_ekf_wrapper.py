"""CPU-side validation tests for EKF wrappers."""

import numpy as np
import pytest

from gnss_gpu.ekf import EKFPositioner


def _generate_satellites(n_sat=4, seed=42):
    rng = np.random.RandomState(seed)
    R_orbit = 26_571_000.0
    theta = rng.uniform(0, 2 * np.pi, n_sat)
    phi = rng.uniform(-np.pi / 3, np.pi / 3, n_sat)
    sat = np.zeros((n_sat, 3))
    sat[:, 0] = R_orbit * np.cos(phi) * np.cos(theta)
    sat[:, 1] = R_orbit * np.cos(phi) * np.sin(theta)
    sat[:, 2] = R_orbit * np.sin(phi)
    return sat


def test_ekf_initialize_rejects_invalid_position_before_native_call():
    ekf = EKFPositioner()

    with pytest.raises(ValueError, match="position_ecef must have shape"):
        ekf.initialize([0.0, 0.0])
    with pytest.raises(ValueError, match="position_ecef must have shape"):
        ekf.initialize([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="position_ecef must be finite"):
        ekf.initialize([0.0, np.nan, 0.0])


def test_ekf_initialize_rejects_invalid_sigmas_before_native_call():
    ekf = EKFPositioner()
    pos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="sigma_pos must be positive"):
        ekf.initialize(pos, sigma_pos=0.0)
    with pytest.raises(ValueError, match="sigma_cb must be finite"):
        ekf.initialize(pos, sigma_cb=np.inf)


def test_ekf_predict_rejects_invalid_dt_before_native_call():
    ekf = EKFPositioner()
    ekf.initialize(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="dt must be positive"):
        ekf.predict(dt=0.0)
    with pytest.raises(ValueError, match="dt must be finite"):
        ekf.predict(dt=np.nan)


def test_ekf_update_rejects_invalid_inputs_before_native_call():
    ekf = EKFPositioner()
    ekf.initialize(np.array([1.0, 2.0, 3.0]))
    sat = _generate_satellites(4)
    pr = np.ones(4)

    with pytest.raises(ValueError, match="pseudoranges must contain at least"):
        ekf.update(sat, [])
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        ekf.update(sat[:2], pr)
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        bad_sat = sat.copy()
        bad_sat[0, 0] = np.inf
        ekf.update(bad_sat, pr)
    with pytest.raises(ValueError, match="pseudoranges must be finite"):
        ekf.update(sat, [1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights length must match"):
        ekf.update(sat, pr, weights=np.ones(3))
    with pytest.raises(ValueError, match="weights must be non-negative"):
        ekf.update(sat, pr, weights=[1.0, -1.0, 1.0, 1.0])
