from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace
from shapely.geometry import LineString
from shapely.strtree import STRtree

from experiments.refine_wp31_moving_block_ambiguity import (
    bias_correct_ddpr_epoch,
    choose_evidence_phase,
    CarrierRow,
    covariance_guided_partial_ar_subsets,
    cp_pr_consistency,
    estimate_arc_integers,
    float_ambiguity_seeds,
    fixed_boundary_affine_route,
    gsi_moving_up_prior,
    optimize_fixed_integers,
    segment_carrier_arcs,
    rank_road_translation_seeds,
    phase_epochs,
    _external_seed_offsets,
    _load_right_boundary_profile,
    shared_road_seed_offsets,
)
from gnss_gpu.local_fgo import DDPseudorangeEpoch


def test_ddpr_diagnostic_loads_explicit_trajectory_csv(tmp_path) -> None:
    from experiments.analyze_ppc_dd_pr_anchor import _load_trajectory_csv

    path = tmp_path / "trajectory.csv"
    path.write_text(
        "tow,ecef_x,ecef_y,ecef_z\n100.04,1,2,3\n100.06,4,5,6\n",
        encoding="utf-8",
    )

    trajectory = _load_trajectory_csv(path)

    np.testing.assert_allclose(trajectory[100.0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(trajectory[100.1], [4.0, 5.0, 6.0])


def test_phase_epochs_aligns_global_stride_inside_block() -> None:
    assert list(phase_epochs(5015, 5030, 5, 2)) == [5017, 5022, 5027]


def test_choose_evidence_phase_uses_supply_then_lowest_phase() -> None:
    diagnostics = [
        {"phase": 0, "evidence_epochs": 0, "raw_carrier_rows": 0, "ddpr_epochs": 0},
        {"phase": 2, "evidence_epochs": 3, "raw_carrier_rows": 20, "ddpr_epochs": 3},
        {"phase": 3, "evidence_epochs": 3, "raw_carrier_rows": 19, "ddpr_epochs": 3},
    ]
    assert choose_evidence_phase(diagnostics) == 2


def test_fixed_boundary_affine_route_blends_to_promoted_offset() -> None:
    route = {10: np.asarray([1.0, 2.0, 3.0]), 11: np.asarray([2.0, 3.0, 4.0])}
    adjusted, scales = fixed_boundary_affine_route(
        route,
        start=10,
        boundary_epoch=12,
        boundary_offset_ecef_m=np.asarray([4.0, 6.0, 8.0]),
    )

    assert scales == {10: 1.0, 11: 0.5}
    np.testing.assert_allclose(adjusted[10], route[10])
    np.testing.assert_allclose(adjusted[11], [4.0, 6.0, 8.0])


def test_fixed_boundary_loader_accepts_chained_fixed_profile(tmp_path) -> None:
    path = tmp_path / "promotion.json"
    path.write_text(
        '{"production_input_truth":false,"production_promoted":true,'
        '"profile_mode":"right_boundary_affine_fixed","segment":[12,20],'
        '"offset_ecef_m":[1,2,3],"reason":"accepted_test"}',
        encoding="utf-8",
    )

    result = _load_right_boundary_profile(path, 12)

    assert result["epoch"] == 12
    np.testing.assert_allclose(result["offset_ecef_m"], [1.0, 2.0, 3.0])


def test_bias_correct_ddpr_epoch_filters_stale_rows_and_subtracts_difference() -> None:
    obs = DDPseudorangeEpoch(
        dd_pseudorange_m=np.asarray([13.0, 25.0]),
        sat_ecef_k=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        sat_ecef_ref=np.zeros((2, 3)),
        base_range_k=np.asarray([1.0, 2.0]),
        base_range_ref=np.zeros(2),
        weights=np.asarray([0.5, 0.7]),
        sat_ids=("G02", "G03"),
        ref_sat_ids=("G01", "G01"),
    )
    corrected = bias_correct_ddpr_epoch(
        obs,
        {("G01", "G02"): (3.0, 10), ("G01", "G03"): (5.0, 1)},
        epoch=12,
        max_age_epochs=5,
    )

    assert corrected is not None
    np.testing.assert_allclose(corrected.dd_pseudorange_m, [10.0])
    assert corrected.sat_ids == ("G02",)
    assert corrected.ref_sat_ids == ("G01",)


def test_estimate_arc_integers_median_rounds_and_rejects_outlier(monkeypatch) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    monkeypatch.setattr(
        module, "_dd_expected_and_jacobian_m", lambda *_args: (0.0, np.zeros(3))
    )
    trajectory = {epoch: np.zeros(3) for epoch in range(4)}
    values = [12.05, 11.96, 12.08, 12.9]
    rows = [
        CarrierRow(
            epoch,
            ("G01", "G02", 190000000),
            value,
            1.0,
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.0,
        )
        for epoch, value in enumerate(values)
    ]
    integers, retained = estimate_arc_integers(np.zeros(3), trajectory, rows)
    assert integers == {("G01", "G02", 190000000): 12}
    assert [row.epoch for row in retained] == [0, 1, 2]


def test_estimate_arc_integers_requires_distinct_epoch_support(monkeypatch) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    monkeypatch.setattr(
        module, "_dd_expected_and_jacobian_m", lambda *_args: (0.0, np.zeros(3))
    )
    trajectory = {0: np.zeros(3), 1: np.zeros(3)}
    rows = [
        CarrierRow(
            epoch,
            ("G01", "G02", 190000000),
            7.0,
            1.0,
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.0,
        )
        for epoch in (0, 0, 1)
    ]
    integers, retained = estimate_arc_integers(
        np.zeros(3), trajectory, rows, min_arc_epochs=3
    )
    assert integers == {}
    assert retained == []


def test_estimate_arc_integers_applies_affine_boundary_scales(monkeypatch) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    monkeypatch.setattr(
        module,
        "_dd_expected_and_jacobian_m",
        lambda position, *_args: (float(position[0]), np.asarray([1.0, 0.0, 0.0])),
    )
    trajectory = {0: np.zeros(3), 1: np.zeros(3)}
    rows = [
        CarrierRow(
            epoch,
            ("G01", "G02", 1_000_000_000),
            measured,
            1.0,
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.0,
        )
        for epoch, measured in ((0, 9.0), (1, 8.0))
    ]

    integers, retained = estimate_arc_integers(
        np.asarray([2.0, 0.0, 0.0]),
        trajectory,
        rows,
        min_arc_epochs=2,
        offset_scales={0: 1.0, 1: 0.5},
    )

    assert integers == {("G01", "G02", 1_000_000_000): 7}
    assert len(retained) == 2


def test_fixed_integer_optimizer_can_enforce_final_up_prior() -> None:
    solution = optimize_fixed_integers(
        np.zeros(3),
        {},
        [],
        {},
        {},
        up_prior=(np.asarray([1.0, 0.0, 0.0]), 2.0, 0.1),
    )

    assert abs(solution[0] - 2.0) < 1e-3
    np.testing.assert_allclose(solution[1:], 0.0, atol=1e-12)


def test_float_ambiguity_seeds_recovers_linear_position_and_integer(
    monkeypatch,
) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    jacobians = {
        1.0: np.asarray([1.0, 0.0, 0.0]),
        2.0: np.asarray([0.0, 1.0, 0.0]),
        3.0: np.asarray([0.0, 0.0, 1.0]),
    }
    monkeypatch.setattr(
        module,
        "_dd_expected_and_jacobian_m",
        lambda _position, sat, *_args: (
            float(np.dot(jacobians[float(sat[0])], _position)),
            jacobians[float(sat[0])],
        ),
    )
    truth = np.asarray([2.0, -3.0, 1.0])
    trajectory = {epoch: np.zeros(3) for epoch in range(3)}
    rows = []
    for epoch in range(3):
        for marker, integer in ((1.0, 7), (2.0, -4), (3.0, 11)):
            measured = float(np.dot(jacobians[marker], truth) + integer)
            rows.append(
                CarrierRow(
                    epoch,
                    ("G00", f"G{int(marker):02d}", 1000000000),
                    measured,
                    1.0,
                    np.asarray([marker, 0, 0]),
                    np.zeros(3),
                    0.0,
                    0.0,
                )
            )
    ddpr = {
        index: SimpleNamespace(
            n=1,
            sat_ecef_k=np.asarray([[marker, 0.0, 0.0]]),
            sat_ecef_ref=np.zeros((1, 3)),
            base_range_k=np.zeros(1),
            base_range_ref=np.zeros(1),
            dd_pseudorange_m=np.asarray([truth[index]]),
        )
        for index, marker in enumerate((1.0, 2.0, 3.0))
    }
    seeds, diagnostics = float_ambiguity_seeds(
        trajectory,
        rows,
        ddpr,
        ddpr_sigma_m=0.01,
        position_prior_sigma_m=20.0,
        up_prior_sigma_m=1.0e6,
        n_candidates=2,
    )
    assert diagnostics["float_integer_arcs"] == 3
    assert any(np.linalg.norm(seed - truth) < 1.0e-6 for seed in seeds)


def test_gsi_moving_up_prior_uses_cached_anchor_calibration(monkeypatch) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    monkeypatch.setattr(module, "_ecef_to_lla_py", lambda x, _y, _z: (0.0, 0.0, x))
    cache = {
        "schema": "wp50_gsi_moving_height_cache_v1",
        "production_input_truth": False,
        "runtime_network_required": False,
        "segment": [10, 12],
        "calibration_points": [
            {
                "dem_source": "5m（レーザ）",
                "geoid_model": "GSI",
                "antenna_position_ecef": [41.5, 0, 0],
                "elevation_m": 2.0,
                "geoid_height_m": 38.0,
            },
            {
                "dem_source": "5m（レーザ）",
                "geoid_model": "GSI",
                "antenna_position_ecef": [41.7, 0, 0],
                "elevation_m": 2.1,
                "geoid_height_m": 38.0,
            },
        ],
        "target_point": {
            "dem_source": "5m（レーザ）",
            "geoid_model": "GSI",
            "elevation_m": 2.2,
            "geoid_height_m": 38.0,
        },
    }
    result = gsi_moving_up_prior(
        cache,
        {10: np.asarray([44.0, 0, 0]), 11: np.asarray([44.2, 0, 0])},
        segment=(10, 12),
    )
    assert abs(result["calibrated_antenna_height_m"] - 1.55) < 1.0e-12
    assert abs(result["up_prior_center_m"] + 2.35) < 1.0e-12


def test_segment_carrier_arcs_splits_jump_and_gap(monkeypatch) -> None:
    import experiments.refine_wp31_moving_block_ambiguity as module

    monkeypatch.setattr(
        module, "_dd_expected_and_jacobian_m", lambda *_args: (0.0, np.zeros(3))
    )
    trajectory = {epoch: np.zeros(3) for epoch in (0, 5, 10, 30)}
    values = (4.0, 4.1, 5.2, 5.25)
    rows = [
        CarrierRow(
            epoch,
            ("G01", "G02", 1000000000),
            value,
            1.0,
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.0,
        )
        for epoch, value in zip(trajectory, values)
    ]
    segmented, splits = segment_carrier_arcs(trajectory, rows, max_epoch_gap=10)
    assert splits == 2
    assert [row.key[1] for row in segmented] == [
        "G02@arc0",
        "G02@arc0",
        "G02@arc1",
        "G02@arc2",
    ]


def test_rank_road_translation_seeds_supplies_shape_alignment() -> None:
    route = np.asarray([[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
    roads = STRtree([LineString([(-20.0, 8.0), (30.0, 8.0)])])
    seeds = rank_road_translation_seeds(
        route,
        roads,
        radius_m=5.0,
        coarse_step_m=1.0,
        fine_step_m=0.1,
        max_seeds=3,
    )
    assert seeds
    assert seeds[0]["road_p95_m"] < 1.0e-9
    assert abs(seeds[0]["translation_xy_m"][1] - 3.0) < 1.0e-9


def test_rank_road_translation_seeds_spatial_cells_cover_distinct_corridors() -> None:
    route = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    roads = STRtree([LineString([(-100.0, 0.0), (100.0, 0.0)])])
    seeds = rank_road_translation_seeds(
        route,
        roads,
        radius_m=20.0,
        coarse_step_m=2.0,
        fine_step_m=1.0,
        max_seeds=4,
        spatial_cell_m=10.0,
    )
    xs = [row["translation_xy_m"][0] for row in seeds]
    assert len(seeds) == 4
    assert max(xs) - min(xs) >= 30.0


def test_external_seed_offsets_reads_truth_free_candidate_pool(tmp_path) -> None:
    path = tmp_path / "pool.json"
    path.write_text(
        '{"candidates":[{"offset_ecef_m":[1,2,3]},'
        '{"offset_ecef_m":[4,5,6],"audit_median_error_m":0.1}]}',
        encoding="utf-8",
    )
    seeds = _external_seed_offsets(path)
    assert [seed.tolist() for seed in seeds] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_shared_road_seed_offsets_preserves_parent_translation(tmp_path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text('{"epsg":3857}', encoding="utf-8")
    trajectory = {0: np.asarray([6378137.0, 0.0, 0.0])}
    rows = [
        {
            "translation_xy_m": [2.0, 3.0],
            "road_p95_m": 0.4,
            "offset_ecef_m": [99, 99, 99],
        }
    ]
    offsets, diagnostics = shared_road_seed_offsets(trajectory, cache, rows)
    assert len(offsets) == 1
    assert diagnostics[0]["translation_xy_m"] == [2.0, 3.0]
    assert diagnostics[0]["offset_ecef_m"] != [99, 99, 99]


def test_covariance_guided_partial_ar_drops_worst_axis_loading() -> None:
    covariance = np.diag([1.0, 9.0, 4.0, 2.0, 3.0])

    subsets = covariance_guided_partial_ar_subsets(
        covariance, minimum_ambiguities=2, max_drop_steps=3, worst_axes=3
    )

    assert [subset.tolist() for subset in subsets] == [
        [0, 2, 3, 4],
        [0, 3, 4],
        [0, 3],
    ]


def test_cp_pr_consistency_rebases_pseudorange_reference() -> None:
    ddpr = {
        4: DDPseudorangeEpoch(
            dd_pseudorange_m=np.asarray([10.0, 16.0]),
            sat_ecef_k=np.zeros((2, 3)),
            sat_ecef_ref=np.zeros((2, 3)),
            base_range_k=np.zeros(2),
            base_range_ref=np.zeros(2),
            weights=np.ones(2),
            sat_ids=("G02", "G03"),
            ref_sat_ids=("G01", "G01"),
        )
    }
    row = CarrierRow(
        epoch=4,
        key=("G02@L1_E1_B1", "G03@L1_E1_B1@arc0", 200_000_000),
        measured_cycles=35.0,
        wavelength_m=0.2,
        sat_ecef_k=np.zeros(3),
        sat_ecef_ref=np.zeros(3),
        base_range_k=0.0,
        base_range_ref=0.0,
    )

    result = cp_pr_consistency([row], {row.key: 5}, ddpr)

    assert result["checked_pairs"] == 1
    assert result["bad_pairs"] == 0
    assert result["rms_innovation_m"] == 0.0


def test_gsi_moving_up_prior_accepts_consistent_one_meter_laser_source() -> None:
    cache = {
        "schema": "wp50_gsi_moving_height_cache_v1",
        "production_input_truth": False,
        "runtime_network_required": False,
        "segment": [10, 20],
        "calibration_points": [
            {
                "dem_source": "1m（レーザ）",
                "geoid_model": "GSIGEO2011_Ver2.2",
                "elevation_m": 1.0,
                "geoid_height_m": 35.0,
                "antenna_position_ecef": [6378138.5, 0.0, 0.0],
            },
            {
                "dem_source": "1m（レーザ）",
                "geoid_model": "GSIGEO2011_Ver2.2",
                "elevation_m": 1.2,
                "geoid_height_m": 35.0,
                "antenna_position_ecef": [6378138.7, 0.0, 0.0],
            },
        ],
        "target_point": {
            "dem_source": "1m（レーザ）",
            "geoid_model": "GSIGEO2011_Ver2.2",
            "elevation_m": 1.1,
            "geoid_height_m": 35.0,
        },
    }
    trajectory = {epoch: np.asarray([6378137.0, 0.0, 0.0]) for epoch in range(10, 20)}

    result = gsi_moving_up_prior(cache, trajectory, segment=(10, 20))

    assert result["dem_source"] == "1m（レーザ）"


def test_gsi_moving_up_prior_uses_source_matched_sample_consensus() -> None:
    cache = {
        "schema": "wp50_gsi_moving_height_cache_v1",
        "production_input_truth": False,
        "runtime_network_required": False,
        "segment": [10, 14],
        "calibration_points": [
            {
                "dem_source": "1m（レーザ）", "geoid_model": "GSI",
                "elevation_m": 1.0, "geoid_height_m": 35.0,
                "antenna_position_ecef": [6378138.5, 0.0, 0.0],
            },
            {
                "dem_source": "1m（レーザ）", "geoid_model": "GSI",
                "elevation_m": 1.2, "geoid_height_m": 35.0,
                "antenna_position_ecef": [6378138.7, 0.0, 0.0],
            },
        ],
        "target_point": {
            "dem_source": "10m", "geoid_model": "GSI",
            "elevation_m": 0.0, "geoid_height_m": 35.0,
        },
        "target_points": [
            {"epoch": 10, "dem_source": "10m", "geoid_model": "GSI",
             "elevation_m": 0.0, "geoid_height_m": 35.0},
            {"epoch": 11, "dem_source": "1m（レーザ）", "geoid_model": "GSI",
             "elevation_m": 1.0, "geoid_height_m": 35.0},
            {"epoch": 12, "dem_source": "1m（レーザ）", "geoid_model": "GSI",
             "elevation_m": 6.0, "geoid_height_m": 35.0},
            {"epoch": 13, "dem_source": "1m（レーザ）", "geoid_model": "GSI",
             "elevation_m": 1.1, "geoid_height_m": 35.0},
        ],
    }
    trajectory = {
        10: np.asarray([6378137.0, 0.0, 0.0]),
        11: np.asarray([6378137.5, 0.0, 0.0]),
        12: np.asarray([6378137.5, 0.0, 0.0]),
        13: np.asarray([6378137.6, 0.0, 0.0]),
    }
    result = gsi_moving_up_prior(cache, trajectory, segment=(10, 14))
    assert abs(result["up_prior_center_m"] - 1.0) < 1.0e-9
    assert result["target_sample_consensus"]["inlier_epochs"] == [11, 13]


def test_gsi_moving_up_prior_rejects_wide_two_point_consensus() -> None:
    cache = {
        "schema": "wp50_gsi_moving_height_cache_v1",
        "production_input_truth": False,
        "runtime_network_required": False,
        "segment": [10, 12],
        "calibration_points": [
            {"dem_source": "1m（レーザ）", "geoid_model": "GSI", "elevation_m": 0.0,
             "geoid_height_m": 0.0, "antenna_position_ecef": [6378138.0, 0.0, 0.0]},
            {"dem_source": "1m（レーザ）", "geoid_model": "GSI", "elevation_m": 0.0,
             "geoid_height_m": 0.0, "antenna_position_ecef": [6378138.0, 0.0, 0.0]},
        ],
        "target_point": {},
        "target_points": [
            {"epoch": 10, "dem_source": "1m（レーザ）", "geoid_model": "GSI",
             "elevation_m": 0.0, "geoid_height_m": 0.0},
            {"epoch": 11, "dem_source": "1m（レーザ）", "geoid_model": "GSI",
             "elevation_m": 0.8, "geoid_height_m": 0.0},
        ],
    }
    trajectory = {
        10: np.asarray([6378138.0, 0.0, 0.0]),
        11: np.asarray([6378138.0, 0.0, 0.0]),
    }
    with pytest.raises(ValueError, match="consensus"):
        gsi_moving_up_prior(cache, trajectory, segment=(10, 12))
