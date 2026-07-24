import csv

import numpy as np

from experiments.exp_wp23b_basin_ar import (
    _append_distinct_position_seed,
    _load_diverse_position_seeds,
    _select_ambiguity_indices,
    _widelane_integer_residual,
)


def test_append_distinct_position_seed_returns_stable_source_index():
    seeds = (np.array([1.0, 2.0, 3.0]),)

    unchanged, existing_index = _append_distinct_position_seed(
        seeds, np.array([1.0, 2.0, 3.0005])
    )
    appended, new_index = _append_distinct_position_seed(
        seeds, np.array([2.0, 2.0, 3.0])
    )

    assert len(unchanged) == 1
    assert existing_index == 0
    assert len(appended) == 2
    assert new_index == 1


def test_load_diverse_position_seeds_ranks_and_caps_by_epoch(tmp_path):
    path = tmp_path / "shadow_basins.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["epoch", "log_weight", "ecef_x", "ecef_y", "ecef_z"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 0, "log_weight": -2, "ecef_x": 1.0, "ecef_y": 0, "ecef_z": 0},
                {"epoch": 0, "log_weight": 0, "ecef_x": 0.0, "ecef_y": 0, "ecef_z": 0},
                {"epoch": 0, "log_weight": -1, "ecef_x": 0.2, "ecef_y": 0, "ecef_z": 0},
                {"epoch": 1, "log_weight": 0, "ecef_x": 2.0, "ecef_y": 0, "ecef_z": 0},
            ]
        )

    seeds = _load_diverse_position_seeds(
        path,
        separation_m=0.5,
        max_positions=2,
    )

    assert tuple(seeds) == (0, 1)
    assert len(seeds[0]) == 2
    np.testing.assert_allclose(seeds[0][0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(seeds[0][1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(seeds[1][0], [2.0, 0.0, 0.0])


def test_widelane_integer_residual_matches_l1_minus_l2():
    assignment = (
        ((("G01@L1_E1_B1", "G02@L1_E1_B1", 190293673), 3), 12),
        ((("G01@L2_E5B_B2", "G02@L2_E5B_B2", 244210213), 1), 7),
        ((("E01@L1_E1_B1", "E02@L1_E1_B1", 190293673), 0), 99),
    )

    n_pairs, squared_residual = _widelane_integer_residual(
        assignment,
        (("G01", "G02", 4),),
    )

    assert n_pairs == 1
    assert squared_residual == 1.0


def test_select_ambiguity_indices_prefers_complete_l1_l2_pairs():
    keys = (
        ("E01@L1_E1_B1", "E02@L1_E1_B1", 190293673),
        ("G01@L1_E1_B1", "G02@L1_E1_B1", 190293673),
        ("G01@L2_E5B_B2", "G02@L2_E5B_B2", 244210213),
        ("G01@L1_E1_B1", "G03@L1_E1_B1", 190293673),
        ("G01@L2_E5B_B2", "G03@L2_E5B_B2", 244210213),
    )

    selected, ranked = _select_ambiguity_indices(
        keys,
        np.diag([0.01, 1.0, 1.0, 2.0, 2.0]),
        np.arange(5),
        4,
        prefer_multifrequency_pairs=True,
    )

    np.testing.assert_array_equal(selected, [1, 2, 3, 4])
    assert set(ranked[:4]) == {1, 2, 3, 4}
