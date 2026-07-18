from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.dd_integrity import multipivot_ddpr_scores, satellite_pair_costs


def _dd_result(positions, observations, reference, system="G"):
    base = np.array([1.0e6, 2.0e6, 3.0e6])
    sat_ids = [f"{system}{i + 1:02d}" for i in range(len(positions))]
    non_reference = [i for i in range(len(positions)) if i != reference]
    rover = np.zeros(3)
    base_ranges = np.linalg.norm(positions - base, axis=1)
    rover_ranges = np.linalg.norm(positions - rover, axis=1)
    single = np.asarray(observations, dtype=float)
    return SimpleNamespace(
        dd_pseudorange_m=np.asarray(
            [
                (rover_ranges[i] - base_ranges[i] + single[i])
                - (rover_ranges[reference] - base_ranges[reference] + single[reference])
                for i in non_reference
            ]
        ),
        sat_ecef_k=positions[non_reference],
        sat_ecef_ref=np.repeat(positions[reference][None, :], len(non_reference), axis=0),
        base_range_k=base_ranges[non_reference],
        base_range_ref=np.repeat(base_ranges[reference], len(non_reference)),
        ref_sat_ids=tuple(sat_ids[reference] for _ in non_reference),
        sat_ids=tuple(sat_ids[i] for i in non_reference),
        n_dd=len(non_reference),
    )


def _merge_dd_results(*results):
    array_fields = (
        "dd_pseudorange_m",
        "sat_ecef_k",
        "sat_ecef_ref",
        "base_range_k",
        "base_range_ref",
    )
    return SimpleNamespace(
        **{field: np.concatenate([getattr(result, field) for result in results]) for field in array_fields},
        ref_sat_ids=sum((result.ref_sat_ids for result in results), ()),
        sat_ids=sum((result.sat_ids for result in results), ()),
        n_dd=sum(result.n_dd for result in results),
    )


def test_multipivot_score_is_invariant_to_original_reference():
    satellites = np.array(
        [[20e6, 0, 0], [0, 21e6, 0], [0, 0, 22e6], [15e6, 15e6, 10e6], [-16e6, 8e6, 12e6]],
        dtype=float,
    )
    innovations = np.array([0.2, -0.1, 0.0, 12.0, 0.1])
    candidates = np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.0]])
    first = multipivot_ddpr_scores(_dd_result(satellites, innovations, 0), candidates)
    second = multipivot_ddpr_scores(_dd_result(satellites, innovations, 2), candidates)
    np.testing.assert_allclose(first.scores, second.scores, atol=1e-10)


def test_multipivot_tolerates_one_biased_satellite():
    satellites = np.array(
        [[20e6, 0, 0], [0, 21e6, 0], [0, 0, 22e6], [15e6, 15e6, 10e6], [-16e6, 8e6, 12e6]],
        dtype=float,
    )
    observations = np.array([0.0, 0.1, -0.1, 20.0, 0.05])
    candidates = np.array([[0.0, 0.0, 0.0], [8.0, -6.0, 4.0]])
    result = multipivot_ddpr_scores(
        _dd_result(satellites, observations, 3), candidates, scale_m=1.0, trim_largest_pairs=4
    )
    assert result.best_index == 0
    np.testing.assert_allclose(result.probabilities.sum(), 1.0)
    assert result.n_satellites == 5


def test_multipivot_combines_multiple_constellations():
    satellites = np.array(
        [[20e6, 0, 0], [0, 21e6, 0], [0, 0, 22e6], [15e6, 15e6, 10e6]],
        dtype=float,
    )
    gps = _dd_result(satellites, np.array([0.1, -0.1, 0.0, 0.2]), 0, "G")
    galileo = _dd_result(satellites * 1.01, np.array([-0.1, 0.0, 0.2, 0.1]), 1, "E")
    candidates = np.array([[0.0, 0.0, 0.0], [10.0, -8.0, 5.0]])
    result = multipivot_ddpr_scores(_merge_dd_results(gps, galileo), candidates)
    assert result.best_index == 0
    assert result.n_constellations == 2
    assert result.n_satellites == 8
    assert np.all(np.isfinite(result.scores))
    np.testing.assert_allclose(result.probabilities.sum(), 1.0)


def test_multipivot_rejects_insufficient_support():
    empty = SimpleNamespace(
        dd_pseudorange_m=np.empty(0),
        sat_ecef_k=np.empty((0, 3)),
        sat_ecef_ref=np.empty((0, 3)),
        base_range_k=np.empty(0),
        base_range_ref=np.empty(0),
        ref_sat_ids=(),
        sat_ids=(),
        n_dd=0,
    )
    with pytest.raises(ValueError, match="insufficient"):
        multipivot_ddpr_scores(empty, np.zeros((1, 3)))


def test_multipivot_can_exclude_biased_original_reference():
    satellites = np.array(
        [[20e6, 0, 0], [0, 21e6, 0], [0, 0, 22e6], [15e6, 15e6, 10e6]],
        dtype=float,
    )
    observations = np.array([30.0, 0.1, -0.1, 0.0])
    candidates = np.array([[0.0, 0.0, 0.0], [10.0, -8.0, 5.0]])
    dd_result = _dd_result(satellites, observations, 0)
    result = multipivot_ddpr_scores(
        dd_result,
        candidates,
        scale_m=1.0,
        excluded_satellites=("G01",),
    )
    assert result.best_index == 0
    assert result.n_satellites == 3


def test_satellite_pair_costs_are_pivot_invariant_and_find_bias():
    satellites = np.array(
        [[20e6, 0, 0], [0, 21e6, 0], [0, 0, 22e6], [15e6, 15e6, 10e6]],
        dtype=float,
    )
    innovations = np.array([0.0, 0.1, 25.0, -0.1])
    first = satellite_pair_costs(
        _dd_result(satellites, innovations, 0), np.zeros(3), scale_m=1.0
    )
    second = satellite_pair_costs(
        _dd_result(satellites, innovations, 1), np.zeros(3), scale_m=1.0
    )
    assert first.satellite_ids == second.satellite_ids
    np.testing.assert_allclose(first.mean_pair_costs, second.mean_pair_costs, atol=1e-8)
    assert first.satellite_ids[int(np.argmax(first.mean_pair_costs))] == "G03"
