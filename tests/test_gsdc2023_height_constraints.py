from __future__ import annotations

import numpy as np
from scipy.io import savemat

from experiments.evaluate import lla_to_ecef
from experiments.gsdc2023_height_constraints import (
    apply_phone_position_offset,
    apply_phone_position_offset_state,
    apply_relative_height_constraint,
    build_relative_height_groups,
    ecef_to_enu_relative,
    enu_to_ecef_relative,
    enu_up_ecef_from_origin,
    load_absolute_height_reference_ecef,
    phone_position_offset,
    relative_height_star_edges_for_reference,
)


def test_phone_position_offset_policy_direct():
    assert phone_position_offset("pixel4") == (-0.0, -0.15)
    assert phone_position_offset("pixel5") == (-0.10, -0.30)
    assert phone_position_offset("sm-a205u") == (0.30, -0.25)
    assert phone_position_offset("unknown") is None


def test_apply_phone_position_offset_shifts_along_heading_direct():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.column_stack(
        [
            np.linspace(0.0, 49.0, 50, dtype=np.float64),
            np.zeros(50, dtype=np.float64),
            np.zeros(50, dtype=np.float64),
        ],
    )
    xyz = enu_to_ecef_relative(enu, origin_xyz)

    offset_xyz = apply_phone_position_offset(xyz, "pixel4")
    offset_enu = ecef_to_enu_relative(offset_xyz, origin_xyz)

    delta = offset_enu - enu
    assert np.allclose(delta[:, 0], -0.15, atol=1e-3)
    assert np.allclose(delta[:, 1], 0.0, atol=1e-3)
    assert np.allclose(delta[:, 2], 0.0, atol=1e-3)


def test_apply_phone_position_offset_state_preserves_extra_columns():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.column_stack([np.linspace(0.0, 9.0, 10), np.zeros(10), np.zeros(10)])
    state = np.column_stack([enu_to_ecef_relative(enu, origin_xyz), np.arange(10.0)])

    shifted = apply_phone_position_offset_state(state, "unknown")

    np.testing.assert_allclose(shifted, state)


def test_relative_height_groups_and_star_edges_direct():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 5.0],
            [120.0, 0.0, 12.0],
            [60.0, 0.0, 25.0],
            [2.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    xyz = enu_to_ecef_relative(enu, origin_xyz)

    groups = build_relative_height_groups(xyz)
    edge_i, edge_j = relative_height_star_edges_for_reference(xyz)

    assert {tuple(group.tolist()) for group in groups} == {(0, 4), (1, 3)}
    np.testing.assert_array_equal(edge_i, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(edge_j, np.array([4, 3], dtype=np.int32))


def test_apply_relative_height_constraint_equalizes_revisit_up_direct():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 5.0],
            [120.0, 0.0, 12.0],
            [60.0, 0.0, 25.0],
            [2.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    xyz = enu_to_ecef_relative(enu, origin_xyz)

    corrected_xyz = apply_relative_height_constraint(xyz, xyz)
    corrected_enu = ecef_to_enu_relative(corrected_xyz, origin_xyz)

    assert np.allclose(corrected_enu[[0, 4], 2], 15.0, atol=1e-3)
    assert np.allclose(corrected_enu[[1, 3], 2], 15.0, atol=1e-3)
    assert np.isclose(corrected_enu[2, 2], enu[2, 2], atol=1e-3)


def test_apply_relative_height_constraint_skips_stop_epochs_direct():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 5.0],
            [120.0, 0.0, 12.0],
            [60.0, 0.0, 25.0],
            [2.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    xyz = enu_to_ecef_relative(enu, origin_xyz)
    stop_mask = np.array([False, False, False, False, True], dtype=bool)

    corrected_xyz = apply_relative_height_constraint(xyz, xyz, stop_mask)
    corrected_enu = ecef_to_enu_relative(corrected_xyz, origin_xyz)

    assert np.isclose(corrected_enu[0, 2], enu[0, 2], atol=1e-3)
    assert np.isclose(corrected_enu[4, 2], enu[4, 2], atol=1e-3)


def test_load_absolute_height_reference_ecef_maps_nearby_ref_hight_direct(tmp_path):
    course_dir = tmp_path / "train" / "course"
    course_dir.mkdir(parents=True)
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    query_enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 9.0],
            [0.0, 5.0, 3.0],
        ],
        dtype=np.float64,
    )
    query_xyz = enu_to_ecef_relative(query_enu, origin_xyz)
    savemat(
        course_dir / "ref_hight.mat",
        {
            "posgt": {
                "enu": np.array([[1.0, 1.0, 100.0], [100.0, 100.0, 50.0]], dtype=np.float64),
                "up": np.array([100.0, 50.0], dtype=np.float64),
            },
        },
    )

    ref_xyz, count = load_absolute_height_reference_ecef(course_dir, query_xyz, dist_m=15.0)

    assert count == 2
    assert ref_xyz is not None
    ref_enu = ecef_to_enu_relative(ref_xyz, origin_xyz)
    np.testing.assert_allclose(ref_enu[[0, 2], 2], np.array([100.0, 100.0]), atol=1e-3)
    assert not np.isfinite(ref_enu[1]).any()


def test_enu_up_ecef_from_origin_direct():
    up = enu_up_ecef_from_origin(np.array([6378137.0, 0.0, 0.0], dtype=np.float64))

    np.testing.assert_allclose(up, np.array([1.0, 0.0, 0.0]), atol=1e-12)
