"""Tests for the stats-computation helpers in examples/demo_urban_multipath.py.

Loads the demo module by path (matching the pattern used by
tests/test_demo_nlos_simulation.py and friends) and exercises
``compute_multipath_stats`` / ``build_satellite_tracks`` against hand-built
:class:`~gnss_gpu.scenario.EpochRecord` / :class:`~gnss_gpu.scenario.ScenarioResult`
objects -- no GPU, no PLATEAU mesh, no network access, no real NAV file
parsing required.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.scenario import EpochRecord, ScenarioConfig, ScenarioResult  # noqa: E402

DEMO_PATH = _REPO_ROOT / "examples" / "demo_urban_multipath.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_urban_multipath", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


# ---------------------------------------------------------------------------
# Hand-built synthetic ScenarioResult: two satellites over three epochs.
#
# G01: always LOS, zero multipath, high C/N0 (clean open-sky satellite).
# G02: NLOS at every epoch, non-zero multipath excess that grows over time,
#      and a lower C/N0 (blocked/diffracted urban-canyon satellite).
# ---------------------------------------------------------------------------


def _make_config() -> ScenarioConfig:
    # Only used as a placeholder container -- these tests never call
    # run_scenario / read the nav_file, so it does not need to exist.
    return ScenarioConfig(
        nav_file="unused.nav",
        lat_deg=35.6,
        lon_deg=139.7,
        alt_m=10.0,
        start_time="2024-01-15T00:00:00",
        duration_s=2.0,
        step_s=1.0,
    )


def _make_two_sat_result(n_epochs: int = 3) -> ScenarioResult:
    t0 = datetime(2024, 1, 15, 0, 0, 0)
    epochs = []
    for i in range(n_epochs):
        epochs.append(
            EpochRecord(
                time_gps_week=2295,
                time_sow=518400.0 + i,
                time_utc=t0 + timedelta(seconds=i),
                rx_ecef=np.array([1.0, 2.0, 3.0]),
                sat_id=np.array(["G01", "G02"], dtype="<U3"),
                elevation_rad=np.radians([60.0, 20.0]),
                azimuth_rad=np.radians([90.0, 200.0 + i]),
                is_los=np.array([True, False]),
                pseudorange_m=np.array([2.0e7, 2.1e7]),
                range_geometric_m=np.array([2.0e7, 2.1e7]),
                sat_clock_bias_m=np.array([0.0, 0.0]),
                iono_m=np.array([1.0, 1.5]),
                tropo_m=np.array([2.0, 3.0]),
                multipath_excess_m=np.array([0.0, 10.0 + 5.0 * i]),
                cn0_dbhz=np.array([45.0, 25.0]),
                doppler_hz=np.array([100.0, -50.0]),
            )
        )
    return ScenarioResult(epochs=epochs, config=_make_config())


def _make_empty_epoch_result() -> ScenarioResult:
    """A scenario with one satellite-bearing epoch and one fully-empty epoch
    (mirrors a real elevation-masked-out epoch, e.g. every satellite briefly
    below the mask)."""
    t0 = datetime(2024, 1, 15, 0, 0, 0)
    populated = EpochRecord(
        time_gps_week=2295,
        time_sow=518400.0,
        time_utc=t0,
        rx_ecef=np.array([1.0, 2.0, 3.0]),
        sat_id=np.array(["G01"], dtype="<U3"),
        elevation_rad=np.radians([45.0]),
        azimuth_rad=np.radians([10.0]),
        is_los=np.array([True]),
        pseudorange_m=np.array([2.0e7]),
        range_geometric_m=np.array([2.0e7]),
        sat_clock_bias_m=np.array([0.0]),
        iono_m=np.array([1.0]),
        tropo_m=np.array([2.0]),
        multipath_excess_m=np.array([0.0]),
        cn0_dbhz=np.array([44.0]),
        doppler_hz=np.array([10.0]),
    )
    empty = EpochRecord(
        time_gps_week=2295,
        time_sow=518401.0,
        time_utc=t0 + timedelta(seconds=1),
        rx_ecef=np.array([1.0, 2.0, 3.0]),
        sat_id=np.zeros(0, dtype="<U3"),
        elevation_rad=np.zeros(0),
        azimuth_rad=np.zeros(0),
        is_los=np.zeros(0, dtype=bool),
        pseudorange_m=np.zeros(0),
        range_geometric_m=np.zeros(0),
        sat_clock_bias_m=np.zeros(0),
        iono_m=np.zeros(0),
        tropo_m=np.zeros(0),
        multipath_excess_m=np.zeros(0),
        cn0_dbhz=np.zeros(0),
        doppler_hz=np.zeros(0),
    )
    return ScenarioResult(epochs=[populated, empty], config=_make_config())


class TestComputeMultipathStats:
    def test_overall_stats_split_los_vs_nlos(self, demo):
        result = _make_two_sat_result(n_epochs=3)
        stats = demo.compute_multipath_stats(result)

        assert stats["n_epochs"] == 3
        assert stats["n_rows"] == 6
        assert stats["n_los"] == 3
        assert stats["n_nlos"] == 3
        assert stats["nlos_fraction"] == pytest.approx(0.5)

        # G02's multipath excess is 10, 15, 20 -> mean 15, max 20.
        assert stats["mean_multipath_excess_nlos_m"] == pytest.approx(15.0)
        assert stats["max_multipath_excess_nlos_m"] == pytest.approx(20.0)

        assert stats["mean_cn0_los_dbhz"] == pytest.approx(45.0)
        assert stats["mean_cn0_nlos_dbhz"] == pytest.approx(25.0)

    def test_worst_offenders_ranks_by_mean_multipath_excess(self, demo):
        result = _make_two_sat_result(n_epochs=3)
        stats = demo.compute_multipath_stats(result, top_n=1)

        assert len(stats["per_satellite"]) == 2
        assert len(stats["worst_offenders"]) == 1
        top = stats["worst_offenders"][0]
        assert top["sat_id"] == "G02"
        assert top["mean_multipath_excess_m"] == pytest.approx(15.0)
        assert top["nlos_fraction"] == pytest.approx(1.0)

        # G01 (always LOS, zero multipath) should rank last.
        assert stats["per_satellite"][-1]["sat_id"] == "G01"
        assert stats["per_satellite"][-1]["mean_multipath_excess_m"] == pytest.approx(0.0)

    def test_handles_epochs_with_no_visible_satellites(self, demo):
        result = _make_empty_epoch_result()
        stats = demo.compute_multipath_stats(result)

        assert stats["n_epochs"] == 2
        assert stats["n_rows"] == 1
        assert stats["nlos_fraction"] == pytest.approx(0.0)
        # No NLOS rows at all -- NaN, not a crash.
        assert np.isnan(stats["mean_multipath_excess_nlos_m"])
        assert np.isnan(stats["mean_cn0_nlos_dbhz"])
        assert stats["mean_cn0_los_dbhz"] == pytest.approx(44.0)


class TestBuildSatelliteTracks:
    def test_groups_by_satellite_and_preserves_epoch_order(self, demo):
        result = _make_two_sat_result(n_epochs=3)
        tracks = demo.build_satellite_tracks(result)

        assert set(tracks.keys()) == {"G01", "G02"}
        g02 = tracks["G02"]
        assert g02["multipath_excess_m"].tolist() == pytest.approx([10.0, 15.0, 20.0])
        assert np.all(~g02["is_los"])
        assert np.all(tracks["G01"]["is_los"])

        # Azimuth for G02 advances by 1 deg per epoch in the fixture.
        assert g02["az_deg"].tolist() == pytest.approx([200.0, 201.0, 202.0])
