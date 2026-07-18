from types import SimpleNamespace

import numpy as np

from gnss_gpu.ambiguity_respawn import (
    condition_respawn_position,
    ddpr_centered_ambiguity_seed,
)


def _carrier_result(position):
    position = np.asarray(position, dtype=np.float64)
    base = position + np.array([-100.0, 20.0, 2.0])
    directions = np.array(
        [[0.4, 0.1, 0.9], [0.1, 0.8, 0.6], [-0.5, 0.4, 0.75], [-0.6, -0.2, 0.75]]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    satellites = position + directions * 22_000_000.0
    sat_ref = np.repeat(satellites[:1], 3, axis=0)
    sat_k = satellites[1:]
    base_ref = np.linalg.norm(sat_ref - base, axis=1)
    base_k = np.linalg.norm(sat_k - base, axis=1)
    geometry = (
        np.linalg.norm(sat_k - position, axis=1)
        - np.linalg.norm(sat_ref - position, axis=1)
        - base_k
        + base_ref
    )
    wavelength = 0.19029367279836488
    integers = np.array([8, -13, 21])
    return SimpleNamespace(
        dd_carrier_cycles=geometry / wavelength + integers,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        dd_weights=np.ones(3),
        wavelengths_m=np.full(3, wavelength),
        ref_sat_ids=("G01",) * 3,
        sat_ids=("G02", "G03", "G04"),
        n_dd=3,
    ), integers


def test_ddpr_centered_seed_recovers_integers_at_linearization_point():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    result, integers = _carrier_result(truth)

    seed = ddpr_centered_ambiguity_seed(result, truth, np.eye(3) * 0.25)

    np.testing.assert_allclose(seed.ahat_cycles, integers, atol=1.0e-8)
    assert np.min(np.linalg.eigvalsh(seed.qahat_cycles2)) > 0.0


def test_integer_conditioning_moves_shifted_ddpr_center_toward_truth():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    result, integers = _carrier_result(truth)
    shifted = truth + np.array([2.0, -1.0, 1.0])
    seed = ddpr_centered_ambiguity_seed(result, shifted, np.eye(3) * 4.0)

    position, covariance, distance = condition_respawn_position(
        seed, seed.keys, integers
    )

    assert np.linalg.norm(position - truth) < np.linalg.norm(shifted - truth)
    assert np.trace(covariance) < np.trace(seed.position_covariance)
    assert distance >= 0.0
