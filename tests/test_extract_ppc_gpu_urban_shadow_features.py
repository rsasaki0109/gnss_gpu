import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "extract_ppc_gpu_urban_shadow_features.py"
_SPEC = importlib.util.spec_from_file_location("extract_ppc_gpu_urban_shadow_features", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EXTRACT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EXTRACT
_SPEC.loader.exec_module(_EXTRACT)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_route_csv_accepts_local_xyz(tmp_path):
    route = tmp_path / "route.csv"
    _write_csv(
        route,
        [
            {"tow": "10.0", "x": "0", "y": "0", "z": "1.6"},
            {"tow": "11.0", "x": "1", "y": "2", "z": "1.7"},
        ],
    )

    tow, local, ecef = _EXTRACT.load_route_csv(route)

    np.testing.assert_allclose(tow, [10.0, 11.0])
    np.testing.assert_allclose(local[:, :2], [[0.0, 0.0], [1.0, 2.0]])
    assert ecef is None


def test_load_route_csv_accepts_ppc_reference_ecef(tmp_path):
    route = tmp_path / "reference.csv"
    _write_csv(
        route,
        [
            {
                "GPS TOW (s)": "187470.0",
                "ECEF X (m)": "-3961767.630",
                "ECEF Y (m)": "3349008.712",
                "ECEF Z (m)": "3698309.780",
            },
            {
                "GPS TOW (s)": "187470.2",
                "ECEF X (m)": "-3961767.500",
                "ECEF Y (m)": "3349008.850",
                "ECEF Z (m)": "3698309.900",
            },
        ],
    )

    tow, local, ecef = _EXTRACT.load_route_csv(route)

    assert ecef is not None
    np.testing.assert_allclose(tow, [187470.0, 187470.2])
    np.testing.assert_allclose(local[0], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.linalg.norm(local[1]) > 0.01


def test_load_route_csv_skips_rows_without_coordinates(tmp_path):
    route = tmp_path / "route.csv"
    _write_csv(
        route,
        [
            {"tow": "9.0", "x": "", "y": "", "z": ""},
            {"tow": "10.0", "x": "0", "y": "0", "z": "1.6"},
            {"tow": "11.0", "x": "1", "y": "2", "z": "1.7"},
        ],
    )

    tow, local, ecef = _EXTRACT.load_route_csv(route)

    assert ecef is None
    np.testing.assert_allclose(tow, [10.0, 11.0])
    np.testing.assert_allclose(local[:, 2], [1.6, 1.7])


def test_extract_features_cpu_only_writes_selector_ready_rows(tmp_path):
    route = tmp_path / "route.csv"
    _write_csv(
        route,
        [
            {"tow": str(i), "x": "0", "y": str(i * 18.0), "z": "1.6"}
            for i in range(8)
        ],
    )
    out = tmp_path / "features.csv"
    args = argparse.Namespace(
        route_csv=route,
        satellite_json=Path("/does/not/exist.json"),
        out_csv=out,
        run_id="unit_run",
        max_epochs=8,
        cpu_only=True,
        seed=123,
        particles_per_epoch=3,
        rx_height_m=1.6,
        sat_range_m=10000.0,
        building_height_scale=1.0,
        min_mesh_length_m=180.0,
        mesh_margin_m=20.0,
        block_depth_m=32.0,
        road_half_width_m=10.0,
        building_width_m=22.0,
        base_height_m=30.0,
        height_wave_m=18.0,
        n_blocks_per_side=3,
    )

    rows = _EXTRACT.extract_features(args)
    _EXTRACT._write_csv(out, rows)

    assert len(rows) == 8
    assert rows[0]["run_id"] == "unit_run"
    assert "gpu_urban_mean_blocked_ratio" in rows[0]
    assert "gpu_urban_particle_shadow_contrast" in rows[0]
    assert all(0.0 <= row["gpu_urban_mean_blocked_ratio"] <= 1.0 for row in rows)
    assert out.exists()


def test_path_alignment_rotates_enu_directions_into_route_frame():
    local = np.array(
        [
            [0.0, 0.0, 1.6],
            [10.0, 0.0, 1.6],
            [20.0, 0.0, 1.6],
        ],
        dtype=np.float64,
    )

    alignment = _EXTRACT._build_path_alignment(local)
    rotated = _EXTRACT._rotate_enu_vectors_to_path_frame(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        alignment,
    )

    np.testing.assert_allclose(rotated[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotated[1], [1.0, 0.0, 0.0], atol=1e-12)


def test_particle_sat_positions_keep_epoch_specific_vectors():
    route = np.array([[0.0, 0.0, 1.6], [0.0, 10.0, 1.6]], dtype=np.float64)
    particles = np.array(
        [
            [1.0, 0.0, 1.6],
            [2.0, 0.0, 1.6],
            [0.0, 11.0, 1.6],
            [0.0, 12.0, 1.6],
        ],
        dtype=np.float64,
    )
    sat_route = np.array(
        [
            [[0.0, 100.0, 1.6], [50.0, 0.0, 1.6]],
            [[0.0, 20.0, 1.6], [20.0, 10.0, 1.6]],
        ],
        dtype=np.float64,
    )

    sat_particles = _EXTRACT._sat_positions_for_particles(
        particles,
        route,
        sat_route,
        particles_per_epoch=2,
    )

    assert sat_particles.shape == (4, 2, 3)
    np.testing.assert_allclose(sat_particles[0, 0], [1.0, 100.0, 1.6])
    np.testing.assert_allclose(sat_particles[2, 0], [0.0, 21.0, 1.6])


def test_load_nav_satellite_route_uses_ephemeris_batch(monkeypatch, tmp_path):
    origin = np.array([_EXTRACT.WGS84_A, 0.0, 0.0], dtype=np.float64)
    route_ecef = np.vstack([origin, origin + np.array([0.0, 0.0, 1.0])])
    local = np.array([[0.0, 0.0, 1.6], [0.0, 1.0, 1.6]], dtype=np.float64)
    route = _EXTRACT._path_aligned_route(local, 1.6)
    alignment = _EXTRACT._build_path_alignment(local)
    high_el_delta = np.array([20_200_000.0, 0.0, 20_200_000.0])
    low_el_delta = np.array([2_000_000.0, 20_000_000.0, 0.0])
    fake_sat_ecef = np.stack(
        [
            np.vstack([route_ecef[0] + high_el_delta, route_ecef[0] + low_el_delta]),
            np.vstack([route_ecef[1] + high_el_delta, route_ecef[1] + low_el_delta]),
        ],
        axis=0,
    )

    class FakeEphemeris:
        def __init__(self, nav_messages):
            self.available_prns = list(nav_messages)

        def compute_batch(self, gps_times, prn_list=None):
            assert list(gps_times) == [10.0, 11.0]
            assert prn_list == ["G01", "G02"]
            return fake_sat_ecef, np.zeros((2, 2), dtype=np.float64), ["G01", "G02"]

    import gnss_gpu.ephemeris as eph_mod
    import gnss_gpu.io.nav_rinex as nav_mod

    monkeypatch.setattr(eph_mod, "Ephemeris", FakeEphemeris)
    monkeypatch.setattr(nav_mod, "read_nav_rinex_multi", lambda *_args, **_kwargs: {"G01": [], "G02": []})

    specs, sat_route = _EXTRACT._load_nav_satellite_route(
        nav_rinex=tmp_path / "base.nav",
        tow=np.array([10.0, 11.0], dtype=np.float64),
        route_ecef=route_ecef,
        route=route,
        alignment=alignment,
        nav_systems=("G",),
        nav_prns=["G01", "G02"],
        min_elevation_deg=20.0,
        sat_range_m=1000.0,
    )

    assert [spec.prn for spec in specs] == ["G01"]
    assert sat_route.shape == (2, 1, 3)
    np.testing.assert_allclose(np.linalg.norm(sat_route - route[:, None, :], axis=2), 1000.0)
