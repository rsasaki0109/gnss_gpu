"""Tests for skyplot / GNSS vulnerability map (requires CUDA GPU)."""

import json
import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_skyplot import compute_grid_quality
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from gnss_gpu.skyplot import VulnerabilityMap, _lla_to_ecef_py, _ecef_to_lla_py

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA skyplot module not available")


# Same satellite constellation used in test_positioning.py (Tokyo area)
SAT_ECEF = np.array([
    [-14985000.0,  -3988000.0,  21474000.0],  # G01
    [ -9575000.0,  15498000.0,  19457000.0],  # G03
    [  7624000.0, -16218000.0,  19843000.0],  # G06
    [ 16305000.0,  12037000.0,  17183000.0],  # G09
    [-20889000.0,  13759000.0,   8291000.0],  # G11
    [  5463000.0,  24413000.0,   8934000.0],  # G14
    [ 22169000.0,   3975000.0,  13781000.0],  # G17
    [-11527000.0, -19421000.0,  13682000.0],  # G22
], dtype=np.float64)


def test_dop_known_geometry():
    """Test DOP with the 8-satellite geometry at a known receiver position.

    With 8 well-distributed satellites, all DOPs should be reasonable (<5).
    """
    # Tokyo Station ECEF
    receiver_ecef = np.array([[-3957199.0, 3310205.0, 3737911.0]], dtype=np.float64)

    pdop, hdop, vdop, gdop, n_vis = compute_grid_quality(
        receiver_ecef.flatten(), SAT_ECEF.flatten(),
        1, 8, np.radians(10.0)
    )

    # 4 of 8 sats are above 10 deg from this location (sats 0,1,4,5)
    assert n_vis[0] >= 4, f"Expected at least 4 visible sats, got {n_vis[0]}"
    assert 0.5 < pdop[0] < 20.0, f"PDOP {pdop[0]} out of range"
    assert 0.5 < hdop[0] < 20.0, f"HDOP {hdop[0]} out of range"
    assert 0.5 < vdop[0] < 20.0, f"VDOP {vdop[0]} out of range"
    assert 0.5 < gdop[0] < 20.0, f"GDOP {gdop[0]} out of range"

    # GDOP >= PDOP >= HDOP (by definition)
    assert gdop[0] >= pdop[0] - 1e-6, "GDOP should be >= PDOP"
    assert pdop[0] >= hdop[0] - 1e-6, "PDOP should be >= HDOP"

    # PDOP^2 = HDOP^2 + VDOP^2
    pdop2 = hdop[0] ** 2 + vdop[0] ** 2
    assert abs(pdop[0] ** 2 - pdop2) < 0.01, \
        f"PDOP^2 ({pdop[0]**2:.4f}) != HDOP^2+VDOP^2 ({pdop2:.4f})"


def test_grid_generation_and_ecef():
    """Test that VulnerabilityMap generates a proper grid and converts to ECEF."""
    vm = VulnerabilityMap(
        origin_lla=(35.68, 139.77, 30.0),
        grid_size_m=100,
        resolution_m=10,
        height_m=1.5,
    )

    # Grid should be square
    assert vm.n_side == vm.n_side
    expected_n = len(np.arange(-50, 50 + 5, 10))
    assert vm.n_side == expected_n, f"Expected {expected_n} ticks, got {vm.n_side}"
    assert vm.n_grid == vm.n_side ** 2

    # All grid ECEF points should be near Earth surface
    norms = np.linalg.norm(vm.grid_ecef, axis=1)
    assert np.all(norms > 6.3e6), "Grid points too close to Earth center"
    assert np.all(norms < 6.4e6), "Grid points too far from Earth center"

    # Roundtrip: ECEF -> LLA -> ECEF
    x, y, z = vm.grid_ecef[:, 0], vm.grid_ecef[:, 1], vm.grid_ecef[:, 2]
    lat, lon, alt = _ecef_to_lla_py(x, y, z)
    x2, y2, z2 = _lla_to_ecef_py(lat, lon, alt)
    assert np.allclose(x, x2, atol=0.01)
    assert np.allclose(y, y2, atol=0.01)
    assert np.allclose(z, z2, atol=0.01)


def test_elevation_mask():
    """Verify that satellites below the mask are excluded."""
    receiver_ecef = np.array([[-3957199.0, 3310205.0, 3737911.0]], dtype=np.float64)

    # Very low mask: should see most satellites
    _, _, _, _, n_vis_low = compute_grid_quality(
        receiver_ecef.flatten(), SAT_ECEF.flatten(),
        1, 8, np.radians(5.0)
    )

    # Very high mask: should see fewer satellites
    _, _, _, _, n_vis_high = compute_grid_quality(
        receiver_ecef.flatten(), SAT_ECEF.flatten(),
        1, 8, np.radians(60.0)
    )

    assert n_vis_low[0] >= n_vis_high[0], \
        f"Low mask ({n_vis_low[0]} sats) should see >= high mask ({n_vis_high[0]} sats)"

    # At 60 deg mask, very few sats should be visible
    assert n_vis_high[0] < n_vis_low[0], \
        "60 deg mask should block some satellites"


def test_elevation_mask_extreme():
    """With 90 deg mask, no satellite should be visible -> DOP = 999."""
    receiver_ecef = np.array([[-3957199.0, 3310205.0, 3737911.0]], dtype=np.float64)

    pdop, hdop, vdop, gdop, n_vis = compute_grid_quality(
        receiver_ecef.flatten(), SAT_ECEF.flatten(),
        1, 8, np.radians(89.0)
    )

    # Very few (likely 0) sats at 89 deg elevation
    if n_vis[0] < 4:
        assert pdop[0] >= 999.0, "PDOP should be 999 when < 4 sats visible"


def test_vulnerability_map_evaluate():
    """End-to-end: create map, evaluate, check output shapes."""
    vm = VulnerabilityMap(
        origin_lla=(35.68, 139.77, 30.0),
        grid_size_m=50,
        resolution_m=25,
        height_m=1.5,
    )

    result = vm.evaluate(SAT_ECEF, elevation_mask_deg=10.0)

    assert "pdop" in result
    assert "hdop" in result
    assert "vdop" in result
    assert "gdop" in result
    assert "n_visible" in result

    ns = vm.n_side
    for key in ["pdop", "hdop", "vdop", "gdop"]:
        assert result[key].shape == (ns, ns), f"{key} shape mismatch"
    assert result["n_visible"].shape == (ns, ns)


def test_to_geojson():
    """Test GeoJSON export."""
    vm = VulnerabilityMap(
        origin_lla=(35.68, 139.77, 30.0),
        grid_size_m=50,
        resolution_m=25,
        height_m=1.5,
    )
    vm.evaluate(SAT_ECEF, elevation_mask_deg=10.0)

    geojson = vm.to_geojson(metric="hdop")
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == vm.n_side ** 2

    feat = geojson["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert "hdop" in feat["properties"]
    assert "fill" in feat["properties"]

    # Verify JSON serialisable
    json_str = json.dumps(geojson)
    assert len(json_str) > 0
