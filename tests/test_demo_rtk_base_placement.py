"""Tests for the pure logic in examples/demo_rtk_base_placement.py.

Covers candidate filtering/ranking on a synthetic
:class:`~gnss_gpu.coverage_map.CoverageMapResult` and common-view scoring on
synthetic :class:`~gnss_gpu.scenario.EpochRecord` lists with a known
satellite overlap. No GPU, PLATEAU mesh, or network access is used.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

from gnss_gpu.coverage_map import CoverageMapConfig, CoverageMapResult
from gnss_gpu.scenario import EpochRecord

DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_rtk_base_placement.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_rtk_base_placement", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coverage_result(mean_los, hdop, availability) -> CoverageMapResult:
    mean_los = np.asarray(mean_los, dtype=np.float64)
    hdop = np.asarray(hdop, dtype=np.float64)
    availability = np.asarray(availability, dtype=np.float64)
    n_north, n_east = mean_los.shape

    cell_lat_deg = np.array(
        [[35.6 + 0.0001 * r for _ in range(n_east)] for r in range(n_north)]
    )
    cell_lon_deg = np.array(
        [[139.7 + 0.0001 * c for c in range(n_east)] for _ in range(n_north)]
    )

    config = CoverageMapConfig(
        nav_file="unused.nav",
        center_lat_deg=35.6,
        center_lon_deg=139.7,
        epoch_times=["2024-01-15T00:00:00"],
    )

    zeros = np.zeros_like(mean_los)
    return CoverageMapResult(
        mean_visible=zeros,
        mean_los=mean_los,
        los_fraction=zeros,
        availability=availability,
        hdop=hdop,
        vdop=zeros,
        gdop=zeros,
        expected_hpe_m=zeros,
        cell_lat_deg=cell_lat_deg,
        cell_lon_deg=cell_lon_deg,
        epoch_times=["2024-01-15T00:00:00"],
        config=config,
    )


def _make_epoch(sat_ids, los_flags) -> EpochRecord:
    n = len(sat_ids)
    z = np.zeros(n, dtype=np.float64)
    return EpochRecord(
        time_gps_week=2300,
        time_sow=0.0,
        time_utc=datetime(2024, 1, 15),
        rx_ecef=np.zeros(3),
        sat_id=np.array(sat_ids, dtype="<U3"),
        elevation_rad=z,
        azimuth_rad=z,
        is_los=np.array(los_flags, dtype=bool),
        pseudorange_m=z,
        range_geometric_m=z,
        sat_clock_bias_m=z,
        iono_m=z,
        tropo_m=z,
        multipath_excess_m=z,
        cn0_dbhz=z,
        doppler_hz=z,
    )


# ---------------------------------------------------------------------------
# Candidate filtering / ranking
# ---------------------------------------------------------------------------


class TestFilterAndRankCandidates:
    def test_excludes_buildings_and_partial_availability(self):
        # (0, 2) is inside a building footprint (NaN); (1, 2) has <100% availability.
        mean_los = [[6.0, 5.0, np.nan], [4.0, 6.0, 5.5]]
        hdop = [[1.2, 1.5, np.nan], [2.0, 1.0, 1.3]]
        availability = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.8]]
        result = _make_coverage_result(mean_los, hdop, availability)

        candidates = demo.filter_and_rank_candidates(result, top_n=5)

        cells = {(c["row"], c["col"]) for c in candidates}
        assert (0, 2) not in cells
        assert (1, 2) not in cells
        assert len(candidates) == 4

    def test_ranks_by_mean_los_desc_then_hdop_asc(self):
        mean_los = [[6.0, 5.0, np.nan], [4.0, 6.0, 5.5]]
        hdop = [[1.2, 1.5, np.nan], [2.0, 1.0, 1.3]]
        availability = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.8]]
        result = _make_coverage_result(mean_los, hdop, availability)

        candidates = demo.filter_and_rank_candidates(result, top_n=5)

        # Both (0,0) and (1,1) have mean_los=6.0; (1,1) has the lower HDOP (1.0
        # vs 1.2) so it must rank first. Then mean_los=5.0 (0,1), then 4.0 (1,0).
        assert [(c["row"], c["col"]) for c in candidates] == [
            (1, 1), (0, 0), (0, 1), (1, 0),
        ]
        assert candidates[0]["mean_los"] == 6.0
        assert candidates[0]["hdop"] == 1.0
        # lat/lon are pulled straight from the result's per-cell grids.
        assert candidates[0]["lat"] == result.cell_lat_deg[1, 1]
        assert candidates[0]["lon"] == result.cell_lon_deg[1, 1]

    def test_top_n_truncates(self):
        mean_los = [[6.0, 5.0, np.nan], [4.0, 6.0, 5.5]]
        hdop = [[1.2, 1.5, np.nan], [2.0, 1.0, 1.3]]
        availability = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.8]]
        result = _make_coverage_result(mean_los, hdop, availability)

        candidates = demo.filter_and_rank_candidates(result, top_n=2)
        assert len(candidates) == 2
        assert [(c["row"], c["col"]) for c in candidates] == [(1, 1), (0, 0)]


# ---------------------------------------------------------------------------
# Common-view scoring
# ---------------------------------------------------------------------------


class TestCommonViewScore:
    def test_mean_intersection_of_los_satellites(self):
        # Epoch 0: base LOS = {G01, G02}, rover LOS = {G01, G04} -> overlap 1.
        base_ep0 = _make_epoch(["G01", "G02", "G03"], [True, True, False])
        rover_ep0 = _make_epoch(["G01", "G02", "G04"], [True, False, True])
        # Epoch 1: base LOS = {G01, G05}, rover LOS = {G01, G05} -> overlap 2.
        base_ep1 = _make_epoch(["G01", "G05"], [True, True])
        rover_ep1 = _make_epoch(["G01", "G05"], [True, True])

        score = demo.common_view_score([base_ep0, base_ep1], [rover_ep0, rover_ep1])
        assert score == 1.5

    def test_no_overlap_scores_zero(self):
        base_ep = _make_epoch(["G01", "G02"], [True, True])
        rover_ep = _make_epoch(["G03", "G04"], [True, True])
        score = demo.common_view_score([base_ep], [rover_ep])
        assert score == 0.0

    def test_empty_epochs_scores_zero(self):
        assert demo.common_view_score([], []) == 0.0

    def test_mismatched_lengths_use_shorter(self):
        base_ep0 = _make_epoch(["G01"], [True])
        base_ep1 = _make_epoch(["G01"], [True])
        rover_ep0 = _make_epoch(["G01"], [True])
        # Only one rover epoch -- the extra base epoch must be ignored.
        score = demo.common_view_score([base_ep0, base_ep1], [rover_ep0])
        assert score == 1.0
