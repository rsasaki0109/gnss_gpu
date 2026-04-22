import numpy as np

from gnss_gpu.imu_position_update import evaluate_imu_tight_position_update


def _sat_geometry(n_sats):
    sat = np.array(
        [[20_000_000.0 + 1000.0 * i, 10_000.0 * i, 5_000.0 * i] for i in range(n_sats)],
        dtype=np.float64,
    )
    spp = np.array([1_000.0, 2_000.0, 3_000.0], dtype=np.float64)
    ranges = np.linalg.norm(sat - spp, axis=1)
    return sat, spp, ranges


def test_imu_tight_position_update_applies_with_loose_sigma_for_clean_spp():
    sat, spp, ranges = _sat_geometry(8)
    decision = evaluate_imu_tight_position_update(
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 2.0, 3.0]),
        0.5,
        sat,
        ranges + 4.0,
        spp,
        n_measurements=8,
    )

    assert decision.apply_update is True
    assert decision.sigma_pos == 30.0
    assert decision.reason == "ok"
    np.testing.assert_allclose(decision.predicted_position, [10.5, 21.0, 31.5])
    assert decision.residual_rms == 0.0


def test_imu_tight_position_update_tightens_sigma_for_low_sat_count():
    sat, spp, ranges = _sat_geometry(5)
    decision = evaluate_imu_tight_position_update(
        np.zeros(3),
        np.ones(3),
        1.0,
        sat,
        ranges + 4.0,
        spp,
        n_measurements=5,
    )

    assert decision.apply_update is True
    assert decision.sigma_pos == 3.0


def test_imu_tight_position_update_tightens_sigma_for_large_spp_residual():
    sat, spp, ranges = _sat_geometry(8)
    pseudoranges = ranges + np.array([0.0, 0.0, 0.0, 0.0, 80.0, -80.0, 60.0, -60.0])
    decision = evaluate_imu_tight_position_update(
        np.zeros(3),
        np.ones(3),
        1.0,
        sat,
        pseudoranges,
        spp,
        n_measurements=8,
    )

    assert decision.apply_update is True
    assert decision.sigma_pos == 3.0
    assert decision.residual_rms is not None
    assert decision.residual_rms > 20.0


def test_imu_tight_position_update_skips_invalid_velocity():
    sat, spp, ranges = _sat_geometry(8)
    decision = evaluate_imu_tight_position_update(
        np.zeros(3),
        np.array([np.nan, 1.0, 2.0]),
        1.0,
        sat,
        ranges,
        spp,
        n_measurements=8,
    )

    assert decision.apply_update is False
    assert decision.predicted_position is None
    assert decision.sigma_pos is None
    assert decision.reason == "invalid_imu_velocity"
