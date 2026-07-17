"""Tests for the route positioning-quality demo (examples/demo_route_accuracy.py).

Exercises only :func:`compute_route_metrics` (and, indirectly, its private
``_epoch_hdop`` helper) -- the pure, GPU/mesh-free part of the demo. One test
drives it through a real (tiny, synthetic) :func:`gnss_gpu.scenario.run_scenario`
call using the same minimal RINEX NAV fixture as ``tests/test_scenario.py``;
the others build :class:`~gnss_gpu.scenario.EpochRecord`/``ScenarioResult``
objects directly so the HDOP/availability arithmetic can be checked against
known geometry without needing ephemeris at all.
"""

from __future__ import annotations

import importlib.util
import math
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

from gnss_gpu.scenario import EpochRecord, ScenarioConfig, ScenarioResult, run_scenario

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_demo_module():
    demo_path = _REPO_ROOT / "examples" / "demo_route_accuracy.py"
    spec = importlib.util.spec_from_file_location("demo_route_accuracy", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# G01's toe (518400.0 s-of-week) falls on 2024-01-15T00:00:00 GPS time -- same
# single-satellite RINEX 3 NAV fixture as tests/test_scenario.py.
_NAV_GPS_ONLY = textwrap.dedent("""\
     3.04           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE
    test_program    test_agency         20240115 000000 UTC PGM / RUN BY / DATE
                                                                END OF HEADER
    G01 2024 01 15 00 00 00-3.930553793907E-05-1.023181539495E-12 0.000000000000E+00
         7.800000000000E+01-1.403125000000E+01 4.623016997497E-09 1.245932843990E+00
        -3.464519977570E-06 5.765914916992E-03 7.525086402893E-06 5.153637939453E+03
         5.184000000000E+05-1.303851604462E-07-2.495230285080E-01 5.587935447693E-08
         9.734965789940E-01 2.241562500000E+02 6.859404140730E-01-8.120689826012E-09
         1.132502065007E-10 1.000000000000E+00 2.295000000000E+03 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00-1.117587089539E-08 7.800000000000E+01
         5.184000000000E+05 4.000000000000E+00
""")
_START_TIME = "2024-01-15T00:00:00"


def _make_epoch(rx_ecef, elevation_deg, azimuth_deg, is_los, time_utc) -> EpochRecord:
    n = len(elevation_deg)
    z = np.zeros(n, dtype=np.float64)
    return EpochRecord(
        time_gps_week=2295,
        time_sow=518400.0,
        time_utc=time_utc,
        rx_ecef=np.asarray(rx_ecef, dtype=np.float64),
        sat_id=np.array([f"G{i:02d}" for i in range(n)], dtype="<U3"),
        elevation_rad=np.radians(np.asarray(elevation_deg, dtype=np.float64)),
        azimuth_rad=np.radians(np.asarray(azimuth_deg, dtype=np.float64)),
        is_los=np.asarray(is_los, dtype=bool),
        pseudorange_m=z, range_geometric_m=z, sat_clock_bias_m=z,
        iono_m=z, tropo_m=z, multipath_excess_m=z, cn0_dbhz=z, doppler_hz=z,
    )


def test_compute_route_metrics_shapes_via_run_scenario(tmp_path):
    """End-to-end through run_scenario with the shared NAV fixture (no plateau)."""
    demo = _load_demo_module()
    nav_file = tmp_path / "test.nav"
    nav_file.write_text(_NAV_GPS_ONLY)

    config = ScenarioConfig(
        nav_file=str(nav_file),
        lat_deg=35.6, lon_deg=139.7, alt_m=30.0,
        start_time=_START_TIME, duration_s=2.0, step_s=1.0,
        constellations=["G"], plateau_dir=None, elevation_mask_deg=0.0,
        pr_noise_sigma_zenith_m=0.0, pr_noise_sigma_horizon_m=0.0, seed=42,
    )
    result = run_scenario(config)
    assert result.n_epochs == 3

    metrics = demo.compute_route_metrics(result)
    for key in ("lat_deg", "lon_deg", "time_s", "n_visible", "n_los", "hdop", "expected_hpe_m", "available"):
        assert key in metrics
        assert metrics[key].shape == (3,)

    # No plateau mesh -> every visible satellite is LOS.
    np.testing.assert_array_equal(metrics["n_visible"], metrics["n_los"])
    # Only one satellite (G01) is ever visible -- below the 4-satellite HDOP floor.
    assert np.all(metrics["n_los"] <= 1)
    assert np.all(~metrics["available"])
    assert np.all(np.isnan(metrics["hdop"]))
    assert np.all(np.isnan(metrics["expected_hpe_m"]))
    np.testing.assert_allclose(metrics["lat_deg"], 35.6, atol=1e-6)
    np.testing.assert_allclose(metrics["lon_deg"], 139.7, atol=1e-6)


def test_hdop_finite_with_four_well_spread_satellites():
    """Four satellites at varied elevation/azimuth give a finite, sane HDOP.

    (Four satellites sharing one elevation makes the "up" and clock-bias
    columns of the design matrix collinear -- singular by construction --
    so the elevations must differ, as they would for any real constellation.)
    """
    demo = _load_demo_module()
    el = [30.0, 60.0, 45.0, 70.0]
    az = [10.0, 100.0, 200.0, 300.0]
    is_los = [True, True, True, True]
    hdop = demo._epoch_hdop(np.radians(el), np.radians(az), np.array(is_los))
    assert math.isfinite(hdop)
    assert hdop > 0.0

    # Dropping one satellite below the LOS threshold makes HDOP undefined again.
    is_los_dropped = [True, True, True, False]
    hdop_dropped = demo._epoch_hdop(np.radians(el), np.radians(az), np.array(is_los_dropped))
    assert math.isnan(hdop_dropped)


def test_availability_and_expected_hpe_from_synthetic_epochs():
    """availability flag and expected_hpe_m = hdop * uere_m on hand-built epochs."""
    demo = _load_demo_module()
    t0 = datetime(2024, 1, 15, 0, 0, 0)
    rx_ecef = np.array([1.0, 2.0, 3.0])

    epoch_good = _make_epoch(
        rx_ecef, [30.0, 60.0, 45.0, 70.0], [10.0, 100.0, 200.0, 300.0],
        [True, True, True, True], t0,
    )
    epoch_bad = _make_epoch(
        rx_ecef, [30.0, 60.0, 45.0], [10.0, 100.0, 200.0],
        [True, True, False], t0.replace(second=1),
    )

    config = ScenarioConfig(
        nav_file="unused.nav", lat_deg=0.0, lon_deg=0.0, alt_m=0.0,
        start_time=_START_TIME, duration_s=1.0,
    )
    result = ScenarioResult(epochs=[epoch_good, epoch_bad], config=config)

    metrics = demo.compute_route_metrics(result, uere_m=7.5)
    np.testing.assert_array_equal(metrics["available"], [True, False])
    assert math.isfinite(metrics["expected_hpe_m"][0])
    assert math.isnan(metrics["expected_hpe_m"][1])
    np.testing.assert_allclose(metrics["expected_hpe_m"][0], metrics["hdop"][0] * 7.5)
    np.testing.assert_allclose(metrics["time_s"], [0.0, 1.0])
