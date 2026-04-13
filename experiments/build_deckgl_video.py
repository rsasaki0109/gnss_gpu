#!/usr/bin/env python3
"""Generate LOS/NLOS geometry video using deck.gl + Playwright.

Uses deck.gl over an OpenStreetMap basemap (no API key needed).
Lighter than CesiumJS and suitable for headless Chrome recording.

The receiver trajectory comes from UrbanNav, while satellite rays use a
synthetic Earth-centered orbital surrogates so the output stays a geometry
sanity check rather than a real-sky validation artifact. To avoid the
misleading impression that satellites sit on the street grid, the visible
satellites are projected onto a virtual sky ceiling above the receiver.
"""

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# NOTE: Do NOT add python/ to sys.path — C extensions (.so) are only
# available via pip install, not from the source tree directly.

from gnss_gpu.io.citygml import parse_citygml
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla


SATELLITE_MANIFEST = (
    {"label": "G05", "system": "GPS", "code": "G", "slot": 5},
    {"label": "G12", "system": "GPS", "code": "G", "slot": 12},
    {"label": "G19", "system": "GPS", "code": "G", "slot": 19},
    {"label": "E11", "system": "Galileo", "code": "E", "slot": 11},
    {"label": "E24", "system": "Galileo", "code": "E", "slot": 24},
    {"label": "E31", "system": "Galileo", "code": "E", "slot": 31},
    {"label": "R04", "system": "GLONASS", "code": "R", "slot": 4},
    {"label": "R11", "system": "GLONASS", "code": "R", "slot": 11},
    {"label": "J02", "system": "QZSS", "code": "J", "slot": 2},
    {"label": "J07", "system": "QZSS", "code": "J", "slot": 7},
)

SYSTEM_ORBIT_CONFIG = {
    "GPS": {
        "radius_m": 26_560_000.0,
        "period_s": 11.966 * 3600.0,
        "axis_seed": np.array([0.24, 0.18, 1.0], dtype=np.float64),
    },
    "Galileo": {
        "radius_m": 29_600_000.0,
        "period_s": 14.08 * 3600.0,
        "axis_seed": np.array([-0.12, 0.32, 1.0], dtype=np.float64),
    },
    "GLONASS": {
        "radius_m": 25_510_000.0,
        "period_s": 11.25 * 3600.0,
        "axis_seed": np.array([0.28, -0.22, 1.0], dtype=np.float64),
    },
    "QZSS": {
        "radius_m": 42_164_000.0,
        "period_s": 23.934 * 3600.0,
        "axis_seed": np.array([-0.34, 0.06, 1.0], dtype=np.float64),
    },
}
INITIAL_ELEVATION_DEG = (18.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 34.0, 58.0, 74.0)
SATELLITE_ORBIT_CACHE = None


def load_trajectory(csv_path, step=200):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]), float(r[" ECEF Y (m)"]), float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("cannot normalize near-zero vector")
    return vec / norm


def _rotation_matrix(axis, angle_rad):
    axis = _normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c + x * x * (1.0 - c), x * y * (1.0 - c) - z * s, x * z * (1.0 - c) + y * s],
        [y * x * (1.0 - c) + z * s, c + y * y * (1.0 - c), y * z * (1.0 - c) - x * s],
        [z * x * (1.0 - c) - y * s, z * y * (1.0 - c) + x * s, c + z * z * (1.0 - c)],
    ], dtype=np.float64)


def _rotate_vector(vec, axis, angle_rad):
    return _rotation_matrix(axis, angle_rad) @ np.asarray(vec, dtype=np.float64)


def _az_el_to_ecef_unit(rx_ecef, az_deg, el_deg):
    east, north, up = _enu_basis(rx_ecef)
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return _normalize(
        math.sin(az) * math.cos(el) * east
        + math.cos(az) * math.cos(el) * north
        + math.sin(el) * up
    )


def _line_sphere_intersection(rx_ecef, los_dir_ecef, shell_radius_m):
    rx = np.asarray(rx_ecef, dtype=np.float64)
    los_dir = _normalize(los_dir_ecef)
    rx_dot = float(np.dot(rx, los_dir))
    radius_sq = shell_radius_m * shell_radius_m
    rx_sq = float(np.dot(rx, rx))
    discriminant = rx_dot * rx_dot - (rx_sq - radius_sq)
    if discriminant <= 0.0:
        raise ValueError("line of sight misses orbital shell")
    range_m = -rx_dot + math.sqrt(discriminant)
    return rx + range_m * los_dir


def _unwrap_angle_deg(previous_deg, current_deg):
    delta = (current_deg - previous_deg + 180.0) % 360.0 - 180.0
    return previous_deg + delta


def _build_satellite_orbits(reference_rx_ecef, sat_manifest):
    orbits = []
    for idx, sat_meta in enumerate(sat_manifest):
        cfg = SYSTEM_ORBIT_CONFIG[sat_meta["system"]]
        az_deg = (26.0 + idx * 37.0 + (sat_meta["slot"] % 5) * 4.0) % 360.0
        el_deg = INITIAL_ELEVATION_DEG[idx % len(INITIAL_ELEVATION_DEG)]
        los_dir = _az_el_to_ecef_unit(reference_rx_ecef, az_deg=az_deg, el_deg=el_deg)
        reference_sat_ecef = _line_sphere_intersection(
            reference_rx_ecef,
            los_dir,
            shell_radius_m=cfg["radius_m"],
        )

        axis_seed = _rotate_vector(
            cfg["axis_seed"],
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            angle_rad=math.radians((sat_meta["slot"] * 19) % 360),
        )
        tangent = np.cross(axis_seed, reference_sat_ecef)
        if float(np.linalg.norm(tangent)) < 1e-6:
            tangent = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), reference_sat_ecef)
        orbit_axis = _normalize(np.cross(reference_sat_ecef, tangent))
        slot_rate_scale = 1.0 + 0.02 * ((sat_meta["slot"] % 5) - 2)
        angular_rate_rad_s = 2.0 * math.pi / cfg["period_s"] * slot_rate_scale
        orbits.append({
            "reference_ecef": reference_sat_ecef,
            "orbit_axis": orbit_axis,
            "angular_rate_rad_s": angular_rate_rad_s,
        })
    return orbits


def generate_sats(rx_ecef, sat_manifest, time_s):
    """Generate synthetic Earth-centered satellite positions for visualization."""
    global SATELLITE_ORBIT_CACHE
    if SATELLITE_ORBIT_CACHE is None:
        SATELLITE_ORBIT_CACHE = {
            "reference_time_s": float(time_s),
            "orbits": _build_satellite_orbits(rx_ecef, sat_manifest),
        }

    delta_t = float(time_s) - SATELLITE_ORBIT_CACHE["reference_time_s"]
    sat_ecef = np.zeros((len(sat_manifest), 3), dtype=np.float64)
    for idx, orbit in enumerate(SATELLITE_ORBIT_CACHE["orbits"]):
        sat_ecef[idx] = _rotate_vector(
            orbit["reference_ecef"],
            orbit["orbit_axis"],
            orbit["angular_rate_rad_s"] * delta_t,
        )
    return sat_ecef


def _enu_basis(rx_ecef):
    lat, lon, _ = ecef_to_lla(*rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array(
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        dtype=np.float64,
    )
    up = np.array(
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        dtype=np.float64,
    )
    return east, north, up


def _ecef_to_enu(rx_ecef, target_ecef):
    east, north, up = _enu_basis(rx_ecef)
    diff = np.asarray(target_ecef, dtype=np.float64) - np.asarray(rx_ecef, dtype=np.float64)
    return np.array(
        [diff.dot(east), diff.dot(north), diff.dot(up)],
        dtype=np.float64,
    )


def _enu_to_lla(rx_ecef, east_m, north_m, up_m):
    east, north, up = _enu_basis(rx_ecef)
    ecef = (
        np.asarray(rx_ecef, dtype=np.float64)
        + east_m * east
        + north_m * north
        + up_m * up
    )
    lat, lon, alt = ecef_to_lla(*ecef)
    return [math.degrees(lat), math.degrees(lon), float(alt)]


def _convex_hull_xy(points_xy):
    pts = np.unique(np.asarray(points_xy, dtype=np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0_deg))
    return np.array(
        [
            (lon_deg - lon0_deg) * meters_per_deg_lon,
            (lat_deg - lat0_deg) * meters_per_deg_lat,
        ],
        dtype=np.float64,
    )


def _latlon_from_local_xy(x_m, y_m, lat0_deg, lon0_deg):
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0_deg))
    return [
        lat0_deg + y_m / meters_per_deg_lat,
        lon0_deg + x_m / meters_per_deg_lon,
    ]


def _polygon_area_xy(points_xy):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _load_visual_buildings(loader, plateau_dir, positions, margin_m=260.0, max_buildings=600):
    traj_lla = np.array(
        [[math.degrees(lat), math.degrees(lon)] for lat, lon, _ in (ecef_to_lla(*p) for p in positions)],
        dtype=np.float64,
    )
    lat0 = float(np.mean(traj_lla[:, 0]))
    lon0 = float(np.mean(traj_lla[:, 1]))
    traj_xy = np.array([_local_xy(lat, lon, lat0, lon0) for lat, lon in traj_lla], dtype=np.float64)

    lat_margin = margin_m / 111_320.0
    lon_margin = margin_m / (111_320.0 * math.cos(math.radians(lat0)))
    lat_min = float(np.min(traj_lla[:, 0]) - lat_margin)
    lat_max = float(np.max(traj_lla[:, 0]) + lat_margin)
    lon_min = float(np.min(traj_lla[:, 1]) - lon_margin)
    lon_max = float(np.max(traj_lla[:, 1]) + lon_margin)

    visual_buildings = []
    for filepath in sorted(Path(plateau_dir).rglob("*.gml")):
        for building in parse_citygml(filepath):
            latlon_vertices = []
            alt_min = float("inf")
            alt_max = float("-inf")
            for polygon in building.polygons:
                if polygon.shape[0] < 3:
                    continue
                if loader._looks_geodetic_degrees(polygon):
                    latlon_vertices.extend((float(lat), float(lon)) for lat, lon in polygon[:, :2])
                    alt_min = min(alt_min, float(np.min(polygon[:, 2])))
                    alt_max = max(alt_max, float(np.max(polygon[:, 2])))
                    continue
                for y_north, x_east, z_up in polygon:
                    lat, lon = loader._gauss_kruger_inverse(y_north, x_east, loader._lat0, loader._lon0)
                    latlon_vertices.append((math.degrees(lat), math.degrees(lon)))
                    alt_min = min(alt_min, float(z_up))
                    alt_max = max(alt_max, float(z_up))

            if not latlon_vertices:
                continue

            lats = np.array([lat for lat, _ in latlon_vertices], dtype=np.float64)
            lons = np.array([lon for _, lon in latlon_vertices], dtype=np.float64)
            centroid_lat = float(np.mean(lats))
            centroid_lon = float(np.mean(lons))
            if not (lat_min <= centroid_lat <= lat_max and lon_min <= centroid_lon <= lon_max):
                continue

            local_points = np.array(
                [_local_xy(lat, lon, centroid_lat, centroid_lon) for lat, lon in latlon_vertices],
                dtype=np.float64,
            )
            hull_xy = _convex_hull_xy(local_points)
            if hull_xy.shape[0] < 3:
                continue

            area_m2 = _polygon_area_xy(hull_xy)
            height_m = alt_max - alt_min
            if area_m2 < 25.0 or height_m < 4.0:
                continue

            hull_latlon = [
                _latlon_from_local_xy(x_m, y_m, centroid_lat, centroid_lon)
                for x_m, y_m in hull_xy
            ]
            centroid_xy = _local_xy(centroid_lat, centroid_lon, lat0, lon0)
            dist2 = float(np.min(np.sum((traj_xy - centroid_xy) ** 2, axis=1)))
            visual_buildings.append(
                {
                    "polygon": [[lon, lat] for lat, lon in hull_latlon],
                    "height_m": round(float(height_m), 1),
                    "area_m2": round(float(area_m2), 1),
                    "_dist2": dist2,
                }
            )

    visual_buildings.sort(key=lambda item: (item["_dist2"], -item["area_m2"]))
    trimmed = visual_buildings[:max_buildings]
    for item in trimmed:
        item.pop("_dist2", None)
    return trimmed


def _sky_disk(rx_ecef, sky_height_m, sky_radius_m, n_points=40):
    points = []
    for theta in np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False):
        east_m = sky_radius_m * math.cos(theta)
        north_m = sky_radius_m * math.sin(theta)
        points.append(_enu_to_lla(rx_ecef, east_m, north_m, sky_height_m))
    return points


def _sky_radial_norm(el_deg):
    elevation_01 = min(max(el_deg / 90.0, 0.0), 1.0)
    return 0.22 + 0.78 * math.sqrt(max(0.0, 1.0 - elevation_01))


def _project_to_sky_ceiling(rx_ecef, sat_ecef, sky_height_m, sky_radius_m):
    east_m, north_m, up_m = _ecef_to_enu(rx_ecef, sat_ecef)
    az = math.atan2(east_m, north_m)
    horiz = math.hypot(east_m, north_m)
    el = math.atan2(up_m, horiz)
    # Keep azimuth/elevation ordering monotonic but avoid collapsing the
    # readable marker layout toward the receiver footprint.
    radial_norm = _sky_radial_norm(math.degrees(el))
    radial_m = sky_radius_m * radial_norm
    sky_east_m = radial_m * math.sin(az)
    sky_north_m = radial_m * math.cos(az)
    sky = _enu_to_lla(rx_ecef, sky_east_m, sky_north_m, sky_height_m)
    return sky, math.degrees(az) % 360.0, math.degrees(el)


def compute_epochs(
    area_name,
    plateau_dir,
    traj_csv,
    n_epochs=14,
    step=240,
    ray_length_m=1000.0,
    sky_height_m=220.0,
    sky_radius_m=240.0,
    building_margin_m=260.0,
    max_buildings=600,
):
    print(f"[{area_name}] Loading PLATEAU + BVH...")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory(plateau_dir)
    bvh = BVHAccelerator.from_building_model(building)

    trajectory_step = max(1, min(step, 10))
    traj_positions, traj_times = load_trajectory(traj_csv, step=trajectory_step)
    positions, times = load_trajectory(traj_csv, step=step)
    indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)
    visual_buildings = _load_visual_buildings(
        loader,
        plateau_dir,
        traj_positions,
        margin_m=building_margin_m,
        max_buildings=max_buildings,
    )

    sat_manifest = list(SATELLITE_MANIFEST)
    n_sat = len(sat_manifest)
    prn_list = list(range(1, n_sat + 1))
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    epochs = []
    traj_lla = []
    traj_t = [float(t - traj_times[0]) for t in traj_times]
    prev_az_cont = {}
    for p in traj_positions:
        la, lo, al = ecef_to_lla(*p)
        traj_lla.append([math.degrees(la), math.degrees(lo), 4.0])

    for fi, ei in enumerate(indices):
        rx = positions[ei]
        t = times[ei] - times[0]
        sats = generate_sats(rx, sat_manifest, time_s=times[ei])
        result = usim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)

        la, lo, al = ecef_to_lla(*rx)
        rx_ll = [math.degrees(la), math.degrees(lo), 4.0]

        rays = []
        current_az_cont = {}
        for i in range(n_sat):
            sat_meta = sat_manifest[i]
            sky_point, az_deg, el_deg = _project_to_sky_ceiling(
                rx,
                sats[i],
                sky_height_m=sky_height_m,
                sky_radius_m=sky_radius_m,
            )
            prn = prn_list[i]
            az_cont = _unwrap_angle_deg(prev_az_cont[prn], az_deg) if prn in prev_az_cont else az_deg
            current_az_cont[prn] = az_cont
            if not result["visible"][i]:
                continue
            rays.append({
                "prn": prn,
                "label": sat_meta["label"],
                "system": sat_meta["system"],
                "code": sat_meta["code"],
                "slot": sat_meta["slot"],
                "los": bool(result["is_los"][i]),
                "el": float(el_deg),
                "az": float(az_deg),
                "az_cont": float(az_cont),
                "sky": sky_point,
            })
        prev_az_cont = current_az_cont

        epochs.append({
            "rx": rx_ll, "rays": rays,
            "n_los": result["n_los"], "n_nlos": result["n_nlos"],
            "sky_disk": _sky_disk(rx, sky_height_m=sky_height_m, sky_radius_m=sky_radius_m),
            "t": t,
        })
        print(f"  [{fi+1}/{n_epochs}] t={t:.0f}s LOS={result['n_los']} NLOS={result['n_nlos']}")

    return {
        "epochs": epochs,
        "trajectory": traj_lla,
        "trajectory_t": traj_t,
        "buildings": visual_buildings,
        "area": area_name,
        "sky": {"height_m": sky_height_m, "radius_m": sky_radius_m},
    }


def generate_html(
    datasets,
    output_path,
    *,
    zoom=15.0,
    pitch=55.0,
    bearing=-20.0,
    rotation_step=6.0,
    hold_ms=2200,
    initial_delay_ms=1000,
):
    data_json = json.dumps(datasets)

    # Center on first epoch of first dataset
    center = datasets[0]["epochs"][0]["rx"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPU Urban GNSS Signal Simulator — LOS/NLOS</title>
<link
  rel="stylesheet"
  href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"
/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@9.1.12/dist.min.js"></script>
<script src="https://unpkg.com/@deck.gl/mapbox@9.1.12/dist.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#0a0f1e; overflow:hidden; font-family:monospace; }}
  #map {{ width:100vw; height:100vh; }}
  #hud {{
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(255,255,255,0.93); color:#222; padding:14px 18px;
    border-radius:10px; border:1px solid #ccc; min-width:270px; box-shadow:0 2px 8px rgba(0,0,0,0.12);
  }}
  #hud h2 {{ margin:0 0 6px 0; font-size:15px; color:#111; }}
  #hud .area {{ font-size:18px; color:#c07800; margin-bottom:4px; }}
  #meta {{ color:#9cb0c8; font-size:11px; margin-top:6px; line-height:1.45; }}
  .los {{ color:#008866; font-weight:bold; }}
  .nlos {{ color:#cc2222; font-weight:bold; }}
  #epoch {{ color:#666; font-size:11px; margin-top:4px; }}
  #legend {{ margin-top:8px; font-size:11px; }}
  #legend span {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
  .maplibregl-control-container {{ font-family: monospace; }}
  .maplibregl-ctrl-bottom-right {{ opacity: 0.9; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="hud">
  <h2>GPU Urban GNSS Signal Sim</h2>
  <div class="area" id="area"></div>
  <div id="stats"></div>
  <div id="epoch"></div>
  <div id="legend">
    <span style="background:#00d4aa"></span>LOS &nbsp;
    <span style="background:#ff6b6b"></span>NLOS &nbsp;
    <span style="background:#ffd93d"></span>Receiver
  </div>
</div>
<script>
const datasets = {data_json};
const HOLD_MS = {int(hold_ms)};
const INITIAL_DELAY_MS = {int(initial_delay_ms)};
const CAMERA_BEARING = {float(bearing)};
const ROTATION_STEP = {float(rotation_step)};
const CAMERA_LEAD_M = 92.0;
const INTRO_MS = 4400.0;
const INTRO_LEAD_M = 34.0;
const INTRO_ZOOM_OFFSET = -0.75;
const INTRO_PITCH_OFFSET = -13.0;
const INTRO_BEARING_OFFSET = -10.0;

const INITIAL_VIEW = {{
  longitude: {center[1]},
  latitude: {center[0]},
  zoom: {float(zoom)},
  pitch: {float(pitch)},
  bearing: CAMERA_BEARING,
}};

const MAP_STYLE = {{
  version: 8,
  sources: {{
    osm: {{
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
      tileSize: 256,
      maxzoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }},
  }},
  layers: [
    {{
      id: 'bg',
      type: 'background',
      paint: {{ 'background-color': '#e8e4da' }},
    }},
    {{
      id: 'osm',
      type: 'raster',
      source: 'osm',
      paint: {{ 'raster-opacity': 1.0 }},
    }},
  ],
}};

const map = new maplibregl.Map({{
  container: 'map',
  style: MAP_STYLE,
  center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
  zoom: INITIAL_VIEW.zoom,
  pitch: INITIAL_VIEW.pitch,
  bearing: INITIAL_VIEW.bearing,
  antialias: true,
  attributionControl: true,
  renderWorldCopies: false,
  dragPan: false,
  scrollZoom: false,
  boxZoom: false,
  dragRotate: false,
  keyboard: false,
  doubleClickZoom: false,
  touchZoomRotate: false,
}});

const overlay = new deck.MapboxOverlay({{
  interleaved: false,
  layers: [],
  getTooltip: ({{object}}) => object && object.label ? `${{object.system}} ${{object.label}} ${{object.los ? 'LOS' : 'NLOS'}} el=${{object.el.toFixed(0)}}°` : null,
}});
map.addControl(overlay);

const CAPTURE_FPS = 60;
const CAPTURE_WIDTH = 1280;
const CAPTURE_HEIGHT = 720;
const HUD_META_LINES = [
  'Map: OpenStreetMap',
  'LOS/NLOS geometry: PLATEAU BVH',
  'Synthetic sky: Earth-centered orbital surrogates',
  'Labels: G=GPS R=GLONASS E=Galileo J=QZSS',
];
const captureCanvas = document.createElement('canvas');
captureCanvas.width = CAPTURE_WIDTH;
captureCanvas.height = CAPTURE_HEIGHT;
const captureCtx = captureCanvas.getContext('2d', {{ alpha: false }});
let captureActive = false;
let captureState = null;

function roundedRectPath(ctx, x, y, w, h, r) {{
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}}

function drawLegendItem(ctx, x, y, color, label) {{
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 9, 12, 12);
  ctx.fillStyle = '#e0e0e0';
  ctx.fillText(label, x + 18, y + 1);
}}

function drawHudToCapture(ctx) {{
  if (!captureState) {{
    return;
  }}
  const panelX = 12;
  const panelY = 12;
  const panelW = 420;
  const panelH = 188;

  ctx.save();
  roundedRectPath(ctx, panelX, panelY, panelW, panelH, 12);
  ctx.fillStyle = 'rgba(10,15,30,0.92)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(51,68,85,0.95)';
  ctx.lineWidth = 1;
  ctx.stroke();

  let y = panelY + 24;
  ctx.textBaseline = 'alphabetic';
  ctx.font = 'bold 15px monospace';
  ctx.fillStyle = '#ffffff';
  ctx.fillText('GPU Urban GNSS Signal Sim', panelX + 18, y);

  y += 30;
  ctx.font = 'bold 18px monospace';
  ctx.fillStyle = '#ffd93d';
  ctx.fillText(captureState.area, panelX + 18, y);

  y += 26;
  ctx.font = 'bold 14px monospace';
  const losText = `LOS: ${{captureState.nLos}}`;
  ctx.fillStyle = '#00d4aa';
  ctx.fillText(losText, panelX + 18, y);
  ctx.fillStyle = '#ff6b6b';
  ctx.fillText(`NLOS: ${{captureState.nNlos}}`, panelX + 18 + ctx.measureText(losText).width + 18, y);

  y += 21;
  ctx.font = '12px monospace';
  ctx.fillStyle = '#8f99ad';
  ctx.fillText(captureState.epochText, panelX + 18, y);

  y += 17;
  ctx.fillStyle = '#9cb0c8';
  for (const line of HUD_META_LINES) {{
    ctx.fillText(line, panelX + 18, y);
    y += 16;
  }}

  y += 5;
  ctx.font = '12px monospace';
  drawLegendItem(ctx, panelX + 18, y, '#00d4aa', 'LOS');
  drawLegendItem(ctx, panelX + 88, y, '#ff6b6b', 'NLOS');
  drawLegendItem(ctx, panelX + 166, y, '#ffd93d', 'Receiver');
  ctx.restore();
}}

function compositeCaptureFrame() {{
  const mapEl = document.getElementById('map');
  const mapRect = mapEl.getBoundingClientRect();
  if (mapRect.width <= 0 || mapRect.height <= 0) {{
    return;
  }}
  const scaleX = CAPTURE_WIDTH / mapRect.width;
  const scaleY = CAPTURE_HEIGHT / mapRect.height;

  captureCtx.save();
  captureCtx.fillStyle = '#e9e6dc';
  captureCtx.fillRect(0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
  for (const canvas of mapEl.querySelectorAll('canvas')) {{
    if (!canvas.width || !canvas.height) {{
      continue;
    }}
    const rect = canvas.getBoundingClientRect();
    const dx = (rect.left - mapRect.left) * scaleX;
    const dy = (rect.top - mapRect.top) * scaleY;
    const dw = rect.width * scaleX;
    const dh = rect.height * scaleY;
    try {{
      captureCtx.drawImage(canvas, dx, dy, dw, dh);
    }} catch (err) {{
      // Ignore transient cross-canvas draw failures during tile uploads.
    }}
  }}
  drawHudToCapture(captureCtx);
  captureCtx.restore();
}}

function clamp01(v) {{
  return Math.max(0, Math.min(1, v));
}}

function smoothstep(t) {{
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}}

function lerp(a, b, t) {{
  return a + (b - a) * t;
}}

function lerpVec(a, b, t) {{
  return a.map((value, idx) => lerp(value, b[idx], t));
}}

function catmullRom(a, b, c, d, t) {{
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * b) +
    (-a + c) * t +
    (2 * a - 5 * b + 4 * c - d) * t2 +
    (-a + 3 * b - 3 * c + d) * t3
  );
}}

function catmullRomVec(a, b, c, d, t) {{
  return b.map((_, idx) => catmullRom(a[idx], b[idx], c[idx], d[idx], t));
}}

function wrapAngleDeltaDeg(deltaDeg) {{
  let delta = deltaDeg % 360.0;
  if (delta > 180.0) {{
    delta -= 360.0;
  }}
  if (delta < -180.0) {{
    delta += 360.0;
  }}
  return delta;
}}

function unwrapAngleDeg(referenceDeg, angleDeg) {{
  return referenceDeg + wrapAngleDeltaDeg(angleDeg - referenceDeg);
}}

function interpolateAzimuthDeg(rayPrev, ray0, ray1, rayNextNext, t) {{
  const az0 = ray0.az_cont ?? ray0.az;
  const az1 = ray1.az_cont ?? unwrapAngleDeg(az0, ray1.az);
  const azPrev = rayPrev?.az_cont ?? unwrapAngleDeg(az0, rayPrev ? rayPrev.az : ray0.az);
  const azNextNext = rayNextNext?.az_cont ?? unwrapAngleDeg(az1, rayNextNext ? rayNextNext.az : ray1.az);
  return catmullRom(azPrev, az0, az1, azNextNext, t);
}}

function mixColor(rgb, alpha) {{
  return [rgb[0], rgb[1], rgb[2], Math.round(alpha)];
}}

function metersToLatDeg(m) {{
  return m / 111320.0;
}}

function metersToLonDeg(m, latDeg) {{
  return m / (111320.0 * Math.max(Math.cos(latDeg * Math.PI / 180.0), 1e-4));
}}

function offsetLatLon(latDeg, lonDeg, bearingDeg, forwardM) {{
  const br = bearingDeg * Math.PI / 180.0;
  const northM = Math.cos(br) * forwardM;
  const eastM = Math.sin(br) * forwardM;
  return [
    latDeg + metersToLatDeg(northM),
    lonDeg + metersToLonDeg(eastM, latDeg),
  ];
}}

function skyRadialNorm(elDeg) {{
  const elevation01 = clamp01(elDeg / 90.0);
  return 0.22 + 0.78 * Math.sqrt(Math.max(0.0, 1.0 - elevation01));
}}

function projectToSkyCeiling(rxLla, skySpec, azDeg, elDeg) {{
  const radialM = skySpec.radius_m * skyRadialNorm(elDeg);
  const [latDeg, lonDeg] = offsetLatLon(rxLla[0], rxLla[1], azDeg, radialM);
  return [latDeg, lonDeg, (rxLla[2] || 0) + skySpec.height_m];
}}

function buildSkyDisk(rxLla, skySpec, nPoints = 40) {{
  const out = [];
  for (let idx = 0; idx < nPoints; idx += 1) {{
    const azDeg = (idx * 360.0) / nPoints;
    const [latDeg, lonDeg] = offsetLatLon(rxLla[0], rxLla[1], azDeg, skySpec.radius_m);
    out.push([lonDeg, latDeg, (rxLla[2] || 0) + skySpec.height_m]);
  }}
  return out;
}}

function rayMap(ep) {{
  const out = new Map();
  for (const ray of ep.rays) {{
    out.set(ray.prn, ray);
  }}
  return out;
}}

function lowerBound(values, target) {{
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {{
    const mid = Math.floor((lo + hi) / 2);
    if (values[mid] < target) {{
      lo = mid + 1;
    }} else {{
      hi = mid;
    }}
  }}
  return lo;
}}

function sampleTrajectoryAt(ds, tSec) {{
  const traj = ds.trajectory || [];
  const times = ds.trajectory_t || [];
  if (!traj.length) {{
    return ds.epochs[0].rx;
  }}
  if (!times.length || times.length !== traj.length) {{
    return traj[0];
  }}
  if (tSec <= times[0]) {{
    return traj[0];
  }}
  const lastIdx = times.length - 1;
  if (tSec >= times[lastIdx]) {{
    return traj[lastIdx];
  }}
  const hi = Math.max(1, lowerBound(times, tSec));
  const lo = hi - 1;
  const denom = Math.max(times[hi] - times[lo], 1e-6);
  return lerpVec(traj[lo], traj[hi], clamp01((tSec - times[lo]) / denom));
}}

function prefixTrajectory(ds, tSec, rxNow) {{
  const traj = ds.trajectory || [];
  const times = ds.trajectory_t || [];
  if (!traj.length) {{
    return [{{ path: [[rxNow[1], rxNow[0], rxNow[2]]] }}];
  }}
  let cutIdx = traj.length - 1;
  if (times.length === traj.length) {{
    cutIdx = Math.min(traj.length - 1, Math.max(0, lowerBound(times, tSec)));
    if (times[cutIdx] > tSec && cutIdx > 0) {{
      cutIdx -= 1;
    }}
  }}
  const prefix = traj
    .slice(0, cutIdx + 1)
    .map(p => [p[1], p[0], p[2]]);
  prefix.push([rxNow[1], rxNow[0], rxNow[2]]);
  return [{{ path: prefix }}];
}}

function interpolateEpoch(ds, epIdx, tRaw) {{
  const prevEp = epIdx > 0 ? ds.epochs[epIdx - 1] : ds.epochs[epIdx];
  const ep = ds.epochs[epIdx];
  const atDatasetEnd = epIdx >= ds.epochs.length - 1;
  const nextEp = atDatasetEnd ? ep : ds.epochs[epIdx + 1];
  const nextNextEp = epIdx + 2 < ds.epochs.length ? ds.epochs[epIdx + 2] : nextEp;
  const tLinear = atDatasetEnd ? 0 : clamp01(tRaw);
  const t = atDatasetEnd ? 0 : smoothstep(tRaw);
  const tSec = atDatasetEnd ? ep.t : lerp(ep.t, nextEp.t, tLinear);
  const rxNow = sampleTrajectoryAt(ds, tSec);
  const ceilingPolygon = buildSkyDisk(rxNow, ds.sky);

  const prevRays = rayMap(prevEp);
  const rays0 = rayMap(ep);
  const rays1 = rayMap(nextEp);
  const nextNextRays = rayMap(nextNextEp);
  const prns = new Set([...rays0.keys(), ...rays1.keys()]);
  const rayData = [];
  const satData = [];

  for (const prn of prns) {{
    const ray0 = rays0.get(prn);
    const ray1 = rays1.get(prn);
    const fadeIn = !ray0 && ray1;
    const fadeOut = ray0 && !ray1;
    const base = ray1 || ray0;
    const sourceAlpha = fadeIn ? 255 * t : fadeOut ? 255 * (1 - t) : 255;
    if (sourceAlpha < 8 || !base) {{
      continue;
    }}

    let azNow = ray0 ? (ray0.az_cont ?? ray0.az) : (ray1.az_cont ?? ray1.az);
    let elNow = ray0 ? ray0.el : ray1.el;
    if (ray0 && ray1) {{
      const rayPrev = prevRays.get(prn) || ray0;
      const rayNextNext = nextNextRays.get(prn) || ray1;
      azNow = interpolateAzimuthDeg(rayPrev, ray0, ray1, rayNextNext, t);
      elNow = catmullRom(rayPrev.el, ray0.el, ray1.el, rayNextNext.el, t);
    }} else if (ray1 && fadeIn) {{
      azNow = ray1.az_cont ?? ray1.az;
      elNow = ray1.el;
    }} else if (ray0 && fadeOut) {{
      azNow = ray0.az_cont ?? ray0.az;
      elNow = ray0.el;
    }}
    const skyNow = projectToSkyCeiling(rxNow, ds.sky, azNow, elNow);

    const losNow = t >= 0.5 ? (ray1 ? ray1.los : ray0.los) : (ray0 ? ray0.los : ray1.los);
    const color = losNow ? [0, 212, 170] : [255, 107, 107];
    const satAlpha = fadeIn ? 230 * t : fadeOut ? 230 * (1 - t) : 230;
    const targetAlpha = fadeIn ? 120 * t : fadeOut ? 120 * (1 - t) : 120;

    rayData.push({{
      source: [rxNow[1], rxNow[0], rxNow[2]],
      target: [skyNow[1], skyNow[0], skyNow[2]],
      los: losNow,
      prn,
      label: base.label,
      system: base.system,
      el: elNow,
      sourceColor: mixColor(color, sourceAlpha),
      targetColor: mixColor(color, targetAlpha),
      width: losNow ? 3 : 5,
    }});

    satData.push({{
      position: [skyNow[1], skyNow[0], skyNow[2]],
      color: mixColor(color, satAlpha),
      prn,
      label: base.label,
      system: base.system,
      los: losNow,
      el: elNow,
    }});
  }}

  return {{
    ep,
    rxNow,
    rayData,
    satData,
    ceilingData: [{{ polygon: ceilingPolygon }}],
    pathData: prefixTrajectory(
      ds,
      tSec,
      rxNow,
    ),
    bearing: CAMERA_BEARING + (epIdx + tLinear) * ROTATION_STEP,
  }};
}}

for (const ds of datasets) {{
  ds.fullPathData = [{{
    path: (ds.trajectory || []).map(p => [p[1], p[0], p[2]]),
  }}];
}}

const datasetDurations = datasets.map(ds => ds.epochs.length * HOLD_MS);
const totalDurationMs = datasetDurations.reduce((acc, value) => acc + value, 0);
let startTs = null;
let mapReady = false;
let firstMeaningfulMs = null;

function initialCaptureState() {{
  const ds = datasets[0];
  const ep = ds.epochs[0];
  return {{
    area: ds.area,
    nLos: ep.n_los,
    nNlos: ep.n_nlos,
    epochText: 'Epoch 1/' + ds.epochs.length + '  t=' + ep.t.toFixed(0) + 's',
  }};
}}

function resetAnimation() {{
  startTs = null;
  firstMeaningfulMs = null;
  captureState = initialCaptureState();
}}

window.__losReset = resetAnimation;

window.__losCapture = {{
  ready: false,
  fps: CAPTURE_FPS,
  async download(durationMs, fps = CAPTURE_FPS) {{
    if (!mapReady) {{
      throw new Error('map not ready');
    }}
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
        ? 'video/webm;codecs=vp8'
        : 'video/webm';
    const stream = captureCanvas.captureStream(fps);
    const recorder = new MediaRecorder(stream, {{
      mimeType,
      videoBitsPerSecond: 12_000_000,
    }});
    const chunks = [];
    const stopped = new Promise((resolve, reject) => {{
      recorder.ondataavailable = event => {{
        if (event.data && event.data.size) {{
          chunks.push(event.data);
        }}
      }};
      recorder.onerror = event => reject(event.error || new Error('MediaRecorder failed'));
      recorder.onstop = () => resolve(new Blob(chunks, {{ type: mimeType }}));
    }});

    captureActive = true;
    recorder.start(250);
    resetAnimation();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    compositeCaptureFrame();
    await new Promise(resolve => setTimeout(resolve, durationMs));
    recorder.stop();
    const blob = await stopped;
    captureActive = false;
    stream.getTracks().forEach(track => track.stop());

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'los_nlos_deckgl.webm';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }},
}};

map.on('load', () => {{
  mapReady = true;
  window.__losCapture.ready = true;
  requestAnimationFrame(renderFrame);
}});

function renderFrame(nowTs) {{
  if (!mapReady) {{
    return;
  }}
  if (startTs === null) {{
    startTs = nowTs;
  }}
  const elapsed = nowTs - startTs;
  if (elapsed < INITIAL_DELAY_MS) {{
    if (captureActive) {{
      compositeCaptureFrame();
    }}
    requestAnimationFrame(renderFrame);
    return;
  }}

  let cycleMs = (elapsed - INITIAL_DELAY_MS) % totalDurationMs;
  let dsIdx = 0;
  while (cycleMs >= datasetDurations[dsIdx]) {{
    cycleMs -= datasetDurations[dsIdx];
    dsIdx += 1;
  }}

  const ds = datasets[dsIdx];
  const epIdx = Math.min(Math.floor(cycleMs / HOLD_MS), ds.epochs.length - 1);
  const segMs = cycleMs - epIdx * HOLD_MS;
  const interp = interpolateEpoch(ds, epIdx, segMs / HOLD_MS);
  const ep = interp.ep;
  if (firstMeaningfulMs === null && interp.satData.length > 0) {{
    firstMeaningfulMs = elapsed;
  }}
  window.__losFrameState = {{
    area: ds.area,
    epochIdx: epIdx,
    satCount: interp.satData.length,
    buildingCount: ds.buildings.length,
    elapsedMs: elapsed,
    meaningfulMs: firstMeaningfulMs,
  }};
  const introMix = dsIdx === 0 ? 1.0 - smoothstep(clamp01(cycleMs / INTRO_MS)) : 0.0;
  const cameraBearing = lerp(interp.bearing, CAMERA_BEARING + INTRO_BEARING_OFFSET, introMix);
  const cameraLeadM = lerp(CAMERA_LEAD_M, INTRO_LEAD_M, introMix);
  const cameraZoom = lerp(INITIAL_VIEW.zoom, INITIAL_VIEW.zoom + INTRO_ZOOM_OFFSET, introMix);
  const cameraPitch = lerp(INITIAL_VIEW.pitch, Math.max(18.0, INITIAL_VIEW.pitch + INTRO_PITCH_OFFSET), introMix);
  const viewCenter = offsetLatLon(interp.rxNow[0], interp.rxNow[1], cameraBearing, cameraLeadM);

  map.easeTo({{
    center: [viewCenter[1], viewCenter[0]],
    zoom: cameraZoom,
    pitch: cameraPitch,
    bearing: cameraBearing,
    duration: 800,
    easing: t => t,
  }});

  overlay.setProps({{
    layers: [
      new deck.PolygonLayer({{
        id: 'buildings',
        data: ds.buildings,
        getPolygon: d => d.polygon,
        extruded: true,
        filled: true,
        stroked: true,
        wireframe: false,
        getElevation: d => d.height_m,
        getFillColor: d => d.height_m > 45 ? [160, 170, 185, 220] : [185, 195, 205, 200],
        getLineColor: [120, 130, 145, 220],
        lineWidthMinPixels: 1,
      }}),
      new deck.PolygonLayer({{
        id: 'sky-ceiling',
        data: interp.ceilingData,
        getPolygon: d => d.polygon,
        filled: true,
        stroked: true,
        getFillColor: [72, 119, 196, 26],
        getLineColor: [132, 176, 255, 140],
        lineWidthMinPixels: 2,
      }}),
      new deck.PathLayer({{
        id: 'traj-context',
        data: ds.fullPathData,
        getPath: d => d.path,
        getColor: [230, 236, 245, 72],
        widthMinPixels: 2,
        widthScale: 1,
      }}),
      new deck.PathLayer({{
        id: 'traj', data: interp.pathData,
        getPath: d => d.path, getColor: [255, 217, 61, 180],
        widthMinPixels: 4, widthScale: 1,
      }}),
      new deck.LineLayer({{
        id: 'rays', data: interp.rayData,
        getSourcePosition: d => d.source,
        getTargetPosition: d => d.target,
        getSourceColor: d => d.sourceColor,
        getTargetColor: d => d.targetColor,
        getWidth: d => d.width,
      }}),
      new deck.ScatterplotLayer({{
        id: 'rx', data: [{{ position: [interp.rxNow[1], interp.rxNow[0], interp.rxNow[2]], color: [255, 217, 61] }}],
        getPosition: d => d.position,
        getFillColor: d => d.color,
        getLineColor: [20, 24, 34, 255],
        lineWidthMinPixels: 2,
        stroked: true,
        getRadius: 36, radiusMinPixels: 9,
      }}),
      new deck.ScatterplotLayer({{
        id: 'sky-nodes', data: interp.satData,
        getPosition: d => d.position,
        getFillColor: d => d.color,
        getLineColor: [10, 15, 30, 255],
        lineWidthMinPixels: 1,
        stroked: true,
        getRadius: d => d.los ? 22 : 26,
        radiusMinPixels: 6,
      }}),
      new deck.TextLayer({{
        id: 'labels', data: interp.satData,
        getPosition: d => d.position,
        getText: d => d.label,
        getColor: d => d.color,
        getSize: 14, getAngle: 0,
        getTextAnchor: 'middle', getAlignmentBaseline: 'bottom',
        fontFamily: 'monospace', fontWeight: 'bold',
      }}),
    ],
  }});

  document.getElementById('area').textContent = ds.area;
  document.getElementById('stats').innerHTML =
    '<span class="los">LOS: ' + ep.n_los + '</span>  <span class="nlos">NLOS: ' + ep.n_nlos + '</span>';
  const epochText = 'Epoch ' + (epIdx+1) + '/' + ds.epochs.length + '  t=' + ep.t.toFixed(0) + 's';
  document.getElementById('epoch').textContent = epochText;
  captureState = {{
    area: ds.area,
    nLos: ep.n_los,
    nNlos: ep.n_nlos,
    epochText,
  }};
  if (captureActive) {{
    compositeCaptureFrame();
  }}
  requestAnimationFrame(renderFrame);
}}

map.on('error', event => {{
  console.error('map-error', event?.error?.message || event);
}});

</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML: {output_path}")


def _start_local_http_server(root_dir):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "-d", root_dir],
        cwd=root_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    return server, f"http://127.0.0.1:{port}"


def capture_still(html_path, output_path, delay_ms=6000, width=1440, height=1100):
    """Capture a representative still frame for PRs and docs."""
    print(f"Capturing still ({delay_ms/1000:.1f}s delay)...")
    output_abs = str(Path(output_path).resolve())
    html_name = Path(html_path).name
    root_dir = str(Path(html_path).resolve().parent)
    server, base_url = _start_local_http_server(root_dir)
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{
    headless: true,
    args: ['--use-angle=swiftshader','--enable-webgl','--ignore-gpu-blocklist']
  }});
  const page = await browser.newPage({{
    viewport: {{ width: {int(width)}, height: {int(height)} }},
  }});
  await page.goto({json.dumps(base_url + "/" + html_name)}, {{ waitUntil: 'domcontentloaded' }});
  await page.waitForTimeout({int(delay_ms)});
  await page.screenshot({{
    path: {json.dumps(output_abs)},
    type: 'png',
  }});
  await browser.close();
}})();
"""
    script_path = os.path.join(os.path.dirname(output_abs), "_still.js")
    with open(script_path, "w") as f:
        f.write(script)
    try:
        subprocess.run(
            ["node", script_path],
            timeout=delay_ms / 1000 + 90,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
            check=True,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        if os.path.exists(script_path):
            os.unlink(script_path)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"Still: {output_path} ({size_mb:.1f}MB)")


def _record_video_playwright_fallback(html_path, output_dir, duration_ms, target_webm, warmup_ms=7000):
    """Fallback recorder when in-page MediaRecorder is unavailable."""
    html_name = Path(html_path).name
    root_dir = str(Path(html_path).resolve().parent)
    server, base_url = _start_local_http_server(root_dir)
    preexisting = {
        path.name
        for path in Path(output_dir).glob("*.webm")
        if path.name != Path(target_webm).name
    }
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{
    headless: true,
    args: ['--use-angle=swiftshader','--enable-webgl','--ignore-gpu-blocklist']
  }});
  const ctx = await browser.newContext({{
    viewport: {{ width: 1280, height: 720 }},
    recordVideo: {{ dir: {json.dumps(output_dir)}, size: {{ width: 1280, height: 720 }} }},
  }});
  const page = await ctx.newPage();
  await page.goto({json.dumps(base_url + "/" + html_name)}, {{ waitUntil: 'domcontentloaded' }});
  await page.waitForFunction(() => window.__losCapture?.ready === true, {{ timeout: 30000 }});
  await page.waitForTimeout({int(warmup_ms)});
  await page.evaluate(() => window.__losReset && window.__losReset());
  await page.waitForTimeout(250);
  await page.waitForTimeout({int(duration_ms)});
  const frameState = await page.evaluate(() => window.__losFrameState || null);
  console.log(JSON.stringify({{ frameState }}));
  await ctx.close();
  await browser.close();
}})().catch(err => {{
  console.error(err);
  process.exit(1);
}});
"""
    script_path = os.path.join(output_dir, "_rec_fallback.js")
    with open(script_path, "w") as f:
        f.write(script)
    capture_meta = None
    try:
        proc = subprocess.run(
            ["node", script_path],
            timeout=(duration_ms + warmup_ms) / 1000 + 60,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
            check=True,
            capture_output=True,
            text=True,
        )
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            capture_meta = payload.get("frameState")
            if capture_meta is not None:
                break
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        if os.path.exists(script_path):
            os.unlink(script_path)

    new_webms = [
        path for path in Path(output_dir).glob("*.webm")
        if path.name != Path(target_webm).name and path.name not in preexisting
    ]
    if not new_webms:
        return None

    src = max(new_webms, key=lambda path: path.stat().st_mtime)
    dst = Path(target_webm)
    os.replace(src, dst)
    return {
        "webm": str(dst),
        "frame_state": capture_meta,
    }


def record_video(
    html_path,
    output_dir,
    duration_ms=25000,
    gif_duration_s=30,
    capture_fps=60,
    trim_start_ms=7000,
):
    """Record the interactive map and export smoother 60 fps masters."""
    print(f"Recording video ({duration_ms/1000:.0f}s @ {capture_fps}fps)...")
    target_webm = os.path.join(output_dir, "los_nlos_deckgl.webm")
    raw_webm = os.path.join(output_dir, "_los_nlos_deckgl_raw.webm")
    trimmed_webm = os.path.join(output_dir, "_los_nlos_deckgl_trimmed.webm")
    if os.path.exists(raw_webm):
        os.unlink(raw_webm)
    if os.path.exists(trimmed_webm):
        os.unlink(trimmed_webm)

    capture_result = _record_video_playwright_fallback(
        html_path,
        output_dir,
        duration_ms,
        raw_webm,
        warmup_ms=trim_start_ms,
    )
    recorded_webm = capture_result["webm"] if capture_result else None
    if not recorded_webm or not os.path.exists(recorded_webm):
        return None

    dst = Path(recorded_webm)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"Raw video: {dst} ({size_mb:.1f}MB)")

    auto_trim_ms = 0
    frame_state = capture_result.get("frame_state") if capture_result else None
    if frame_state and frame_state.get("meaningfulMs") is not None:
        auto_trim_ms = max(0, int(round(frame_state["meaningfulMs"])) - 250)
        print(
            "First meaningful frame:"
            f" t={frame_state['meaningfulMs']:.0f}ms"
            f" sat={frame_state.get('satCount', 0)}"
            f" buildings={frame_state.get('buildingCount', 0)}"
        )

    effective_trim_ms = max(int(trim_start_ms), auto_trim_ms)
    source_webm = str(dst)
    if effective_trim_ms > 0:
        trim_start_s = effective_trim_ms / 1000.0
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(trim_start_s),
            "-i", str(dst),
            "-avoid_negative_ts", "make_zero",
            "-c", "copy",
            trimmed_webm,
        ], capture_output=True, check=True)
        if os.path.exists(trimmed_webm):
            source_webm = trimmed_webm
            size_mb = os.path.getsize(trimmed_webm) / 1e6
            print(f"Trimmed WebM: {trimmed_webm} ({size_mb:.1f}MB, start +{trim_start_s:.1f}s)")

    mp4_path = os.path.join(output_dir, "los_nlos_deckgl.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", source_webm,
        "-vf", f"framerate=fps={int(capture_fps)}:interp_start=0:interp_end=255:scene=100,format=yuv420p",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-movflags", "+faststart",
        mp4_path,
    ], capture_output=True, check=True)
    if os.path.exists(mp4_path):
        size_mb = os.path.getsize(mp4_path) / 1e6
        print(f"MP4: {mp4_path} ({size_mb:.1f}MB)")

    os.replace(source_webm, target_webm)
    if os.path.exists(target_webm):
        size_mb = os.path.getsize(target_webm) / 1e6
        print(f"WebM: {target_webm} ({size_mb:.1f}MB)")

    gif_path = os.path.join(output_dir, "los_nlos_deckgl.gif")
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", "fps=10,scale=800:-1:flags=lanczos",
        "-t", str(gif_duration_s), gif_path
    ], capture_output=True, check=True)
    if os.path.exists(gif_path):
        size_mb = os.path.getsize(gif_path) / 1e6
        print(f"GIF: {gif_path} ({size_mb:.1f}MB)")
    if os.path.exists(raw_webm):
        os.unlink(raw_webm)
    if os.path.exists(trimmed_webm):
        os.unlink(trimmed_webm)
    return gif_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-epochs", type=int, default=14,
                        help="Epochs to render per area.")
    parser.add_argument("--step", type=int, default=240,
                        help="Trajectory subsampling step.")
    parser.add_argument("--ray-length-m", type=float, default=1000.0,
                        help="Deprecated compatibility flag; no longer used.")
    parser.add_argument("--sky-height-m", type=float, default=220.0,
                        help="Virtual sky ceiling height above the receiver [m].")
    parser.add_argument("--sky-radius-m", type=float, default=240.0,
                        help="Virtual sky ceiling radius [m].")
    parser.add_argument("--building-margin-m", type=float, default=260.0,
                        help="Margin around the trajectory for 3D building extraction [m].")
    parser.add_argument("--max-buildings", type=int, default=600,
                        help="Maximum number of nearby buildings to draw per area.")
    parser.add_argument("--hold-ms", type=int, default=2200,
                        help="Milliseconds to hold each epoch on screen.")
    parser.add_argument("--initial-delay-ms", type=int, default=1000,
                        help="Delay before animation starts.")
    parser.add_argument("--duration-ms", type=int, default=0,
                        help="Recording duration in ms. 0 means auto.")
    parser.add_argument("--gif-duration-s", type=int, default=30,
                        help="GIF clip length in seconds.")
    parser.add_argument("--capture-fps", type=int, default=60,
                        help="Target video frame rate for recorder/compositor output.")
    parser.add_argument("--trim-start-ms", type=int, default=7000,
                        help="Force-trim this many milliseconds from the start of exported video files.")
    parser.add_argument("--still-delay-ms", type=int, default=6000,
                        help="Delay before capturing the still preview image.")
    parser.add_argument("--zoom", type=float, default=14.2,
                        help="Map zoom level.")
    parser.add_argument("--pitch", type=float, default=41.0,
                        help="Camera pitch in degrees.")
    parser.add_argument("--bearing", type=float, default=-20.0,
                        help="Initial camera bearing in degrees.")
    parser.add_argument("--rotation-step", type=float, default=1.8,
                        help="Bearing increment per epoch.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")

    shinjuku = compute_epochs(
        "Shinjuku",
        "experiments/data/plateau_shinjuku",
        "experiments/data/urbannav/Shinjuku/reference.csv",
        n_epochs=args.n_epochs,
        step=args.step,
        ray_length_m=args.ray_length_m,
        sky_height_m=args.sky_height_m,
        sky_radius_m=args.sky_radius_m,
        building_margin_m=args.building_margin_m,
        max_buildings=args.max_buildings,
    )
    odaiba = compute_epochs(
        "Odaiba",
        "experiments/data/plateau_odaiba",
        "experiments/data/urbannav/Odaiba/reference.csv",
        n_epochs=args.n_epochs,
        step=args.step,
        ray_length_m=args.ray_length_m,
        sky_height_m=args.sky_height_m,
        sky_radius_m=args.sky_radius_m,
        building_margin_m=args.building_margin_m,
        max_buildings=args.max_buildings,
    )

    html_path = os.path.join(out_dir, "los_nlos_deckgl.html")
    datasets = [shinjuku, odaiba]
    generate_html(
        datasets,
        html_path,
        zoom=args.zoom,
        pitch=args.pitch,
        bearing=args.bearing,
        rotation_step=args.rotation_step,
        hold_ms=args.hold_ms,
        initial_delay_ms=args.initial_delay_ms,
    )

    capture_still(
        html_path,
        os.path.join(out_dir, "los_nlos_deckgl_still.png"),
        delay_ms=args.still_delay_ms,
    )

    duration_ms = args.duration_ms
    if duration_ms <= 0:
        duration_ms = args.initial_delay_ms + len(datasets) * args.n_epochs * args.hold_ms

    record_video(
        html_path,
        out_dir,
        duration_ms=duration_ms,
        gif_duration_s=args.gif_duration_s,
        capture_fps=args.capture_fps,
        trim_start_ms=args.trim_start_ms,
    )


if __name__ == "__main__":
    main()
