from types import SimpleNamespace

import numpy as np

from gnss_gpu.dd_float_kf import DDFloatKalmanFilter


def _synthetic_dd(position, *, ref_id="G01", wavelength=0.19029367279836488):
    position = np.asarray(position, dtype=np.float64)
    base = position + np.array([-120.0, 45.0, 3.0])
    directions = np.array(
        [
            [0.45, 0.10, 0.89],
            [0.20, 0.75, 0.63],
            [-0.35, 0.55, 0.76],
            [-0.65, -0.10, 0.75],
            [0.05, -0.80, 0.60],
            [0.70, -0.45, 0.55],
            [-0.55, -0.60, 0.58],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    satellites = position + directions * 22_000_000.0
    sat_ref = np.repeat(satellites[:1], satellites.shape[0] - 1, axis=0)
    sat_k = satellites[1:]
    base_range_ref = np.linalg.norm(sat_ref - base, axis=1)
    base_range_k = np.linalg.norm(sat_k - base, axis=1)
    rover_range_ref = np.linalg.norm(sat_ref - position, axis=1)
    rover_range_k = np.linalg.norm(sat_k - position, axis=1)
    geometry = rover_range_k - rover_range_ref - base_range_k + base_range_ref
    sat_ids = tuple(f"G{i:02d}" for i in range(2, satellites.shape[0] + 1))
    refs = (ref_id,) * len(sat_ids)
    integers = np.array([12.0, -7.0, 31.0, 4.0, -19.0, 8.0])
    common = dict(
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_range_k,
        base_range_ref=base_range_ref,
        dd_weights=np.ones(len(sat_ids)),
        ref_sat_ids=refs,
        sat_ids=sat_ids,
        n_dd=len(sat_ids),
    )
    dd_pr = SimpleNamespace(dd_pseudorange_m=geometry.copy(), **common)
    dd_cp = SimpleNamespace(
        dd_carrier_cycles=geometry / wavelength + integers,
        wavelengths_m=np.full(len(sat_ids), wavelength),
        **common,
    )
    return dd_pr, dd_cp, integers


def test_float_kf_converges_and_exports_joint_lambda_seed():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    dd_pr, dd_cp, integers = _synthetic_dd(truth)
    kf = DDFloatKalmanFilter(
        truth + np.array([8.0, -5.0, 3.0]),
        position_sigma_m=30.0,
        velocity_sigma_mps=1.0,
    )

    for _ in range(5):
        pr_diag = kf.update_pseudorange(dd_pr, sigma_pr_m=0.2)
    cp_diag = kf.update_carrier(
        dd_cp,
        dd_pseudorange_result=dd_pr,
        sigma_cp_cycles=0.02,
    )
    seed = kf.ambiguity_seed()

    assert np.linalg.norm(kf.position_ecef - truth) < 0.05
    np.testing.assert_allclose(seed.ahat_cycles, integers, atol=1e-3)
    assert seed.qahat_cycles2.shape == (6, 6)
    assert seed.position_ambiguity_cov.shape == (3, 6)
    assert np.min(np.linalg.eigvalsh(seed.qahat_cycles2)) > 0.0
    assert pr_diag.normalized_innovation_sq >= 0.0
    assert cp_diag.covariance_min_eig > 0.0


def test_integer_conditioning_returns_finite_position_and_smaller_covariance():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    dd_pr, dd_cp, integers = _synthetic_dd(truth)
    kf = DDFloatKalmanFilter(truth + np.array([3.0, 2.0, -1.0]))
    kf.update_pseudorange(dd_pr, sigma_pr_m=1.0)
    kf.update_carrier(dd_cp, dd_pseudorange_result=dd_pr, sigma_cp_cycles=0.05)
    seed = kf.ambiguity_seed()

    fixed_pos, fixed_cov, distance = kf.condition_position_on_integers(
        seed.keys, integers
    )

    assert np.all(np.isfinite(fixed_pos))
    assert np.all(np.isfinite(fixed_cov))
    assert distance >= 0.0
    assert np.trace(fixed_cov) <= np.trace(kf.covariance[:3, :3]) + 1e-10


def test_predict_propagates_position_and_velocity_covariance():
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, -2.0, 0.5])
    kf = DDFloatKalmanFilter(position, velocity_ecef=velocity)
    before_trace = np.trace(kf.covariance[:6, :6])

    kf.predict(0.2)

    np.testing.assert_allclose(kf.position_ecef, position + velocity * 0.2)
    assert np.trace(kf.covariance[:6, :6]) > before_trace


def test_stale_or_explicitly_released_ambiguities_leave_state():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    dd_pr, dd_cp, _ = _synthetic_dd(truth)
    kf = DDFloatKalmanFilter(truth, max_track_age_epochs=0)
    kf.update_carrier(dd_cp, dd_pseudorange_result=dd_pr)
    keys = kf.ambiguity_seed().keys
    assert len(keys) == 6

    kf.release(keys[:2])
    assert len(kf.ambiguity_seed().keys) == 4
    kf.predict(0.2)
    assert len(kf.ambiguity_seed().keys) == 0
    assert kf.mean.size == kf.NAV_DIM


def test_large_carrier_innovation_resets_only_the_slipped_generation():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    dd_pr, dd_cp, _ = _synthetic_dd(truth)
    kf = DDFloatKalmanFilter(truth)
    kf.update_pseudorange(dd_pr, sigma_pr_m=0.5)
    kf.update_carrier(dd_cp, dd_pseudorange_result=dd_pr)

    slipped = SimpleNamespace(**vars(dd_cp))
    slipped.dd_carrier_cycles = np.asarray(dd_cp.dd_carrier_cycles).copy()
    slipped.dd_carrier_cycles[2] += 10.0
    diagnostics = kf.update_carrier(
        slipped,
        dd_pseudorange_result=dd_pr,
        slip_threshold_cycles=2.0,
    )

    assert diagnostics.ambiguities_reset == 1
    assert diagnostics.covariance_min_eig > 0.0
