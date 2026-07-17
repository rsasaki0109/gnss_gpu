from types import SimpleNamespace

import numpy as np

from gnss_gpu.ambiguity_basin_pf import (
    AmbiguityBasinParticleFilter,
    BasinKalmanState,
)
from gnss_gpu.dd_float_kf import _pair_keys


def _conditional(x=0.0):
    return BasinKalmanState.from_position(
        np.array([x, 0.0, 0.0]),
        np.eye(3),
        velocity_sigma_mps=1.0,
    )


def _versioned_key(sat="G02"):
    return (("G01", sat, 190293673), 0)


def _synthetic_carrier(position):
    position = np.asarray(position, dtype=np.float64)
    base = position + np.array([-100.0, 20.0, 2.0])
    directions = np.array(
        [[0.4, 0.1, 0.9], [0.1, 0.8, 0.6], [-0.5, 0.4, 0.75], [-0.6, -0.2, 0.75]],
        dtype=np.float64,
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
    result = SimpleNamespace(
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
    )
    return result, integers


def test_cumulative_likelihood_concentrates_gamma_and_requires_streak():
    correct = {_versioned_key(): 4}
    wrong = {_versioned_key(): 5}
    pf = AmbiguityBasinParticleFilter(
        fix_gamma_threshold=0.99, fix_min_streak=3, min_fixed_ambiguities=1
    )
    pf.spawn([correct, wrong], [_conditional(), _conditional()])
    correct_id = next(b.basin_id for b in pf.basins if b.assignment_dict == correct)
    wrong_id = next(b.basin_id for b in pf.basins if b.assignment_dict == wrong)

    states = []
    for _ in range(3):
        pf.update_log_likelihoods({correct_id: 0.0, wrong_id: -5.0})
        states.append(pf.posterior())

    assert states[0].gamma > 0.99
    assert not states[0].fixed
    assert not states[1].fixed
    assert states[2].fixed
    assert dict(states[2].map_assignment) == correct


def test_duplicate_assignments_merge_mass_and_conditionals():
    assignment = {_versioned_key(): 7}
    pf = AmbiguityBasinParticleFilter()
    pf.spawn([assignment, assignment], [_conditional(0.0), _conditional(2.0)])

    assert len(pf.basins) == 1
    assert np.exp(pf.basins[0].log_weight) == 1.0
    np.testing.assert_allclose(pf.basins[0].conditional.mean[0], 1.0)
    assert pf.basins[0].conditional.covariance[0, 0] > 1.0


def test_release_removes_generation_and_deduplicates_result():
    key = _versioned_key()
    pf = AmbiguityBasinParticleFilter()
    pf.spawn([{key: 1}, {key: 2}], [_conditional(), _conditional(1.0)])
    assert len(pf.basins) == 2

    pf.release([key])

    assert len(pf.basins) == 1
    assert pf.basins[0].assignment == ()


def test_fixed_carrier_marginal_likelihood_prefers_correct_integer_basin():
    truth = np.array([3_875_000.0, 3_325_000.0, 3_750_000.0])
    result, integers = _synthetic_carrier(truth)
    pair_keys = _pair_keys(result)
    generations = {key: 0 for key in pair_keys}
    correct = {(key, 0): int(value) for key, value in zip(pair_keys, integers)}
    wrong = {(key, 0): int(value + 4) for key, value in zip(pair_keys, integers)}
    correct_state = BasinKalmanState.from_position(truth, np.eye(3) * 0.1)
    wrong_state = correct_state.clone()

    ll_correct, n_correct = correct_state.update_fixed_carrier(
        result, correct, generations, sigma_cp_cycles=0.05
    )
    ll_wrong, n_wrong = wrong_state.update_fixed_carrier(
        result, wrong, generations, sigma_cp_cycles=0.05
    )

    assert n_correct == n_wrong == 3
    assert ll_correct > ll_wrong + 10.0
    assert np.linalg.norm(correct_state.mean[:3] - truth) < np.linalg.norm(
        wrong_state.mean[:3] - truth
    )


def test_respawn_preserves_parent_lineage_and_caps_population():
    pf = AmbiguityBasinParticleFilter(max_basins=3)
    pf.spawn([{_versioned_key(): 0}], [_conditional()])
    parent = pf.basins[0].basin_id
    candidates = [{_versioned_key(f"G{i:02d}"): i} for i in range(2, 7)]
    pf.spawn(candidates, [_conditional(float(i)) for i in range(5)], prior_mass=0.5, parent_id=parent)

    assert len(pf.basins) == 3
    assert any(parent in basin.lineage for basin in pf.basins)
    assert np.isclose(sum(np.exp(b.log_weight) for b in pf.basins), 1.0)
