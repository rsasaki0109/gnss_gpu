#!/usr/bin/env python3
"""Extract GPU urban-shadow features for a real route.

The input route can be a PPC/UrbanNav-style `reference.csv` with ECEF or LLA
columns, or a small generic CSV with `tow,x,y,z` local coordinates.  The route is
converted into a local path-aligned frame, a synthetic canyon mesh is generated
around it, and CUDA BVH batch LOS is used when available.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
DEFAULT_ROUTE = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run1/reference.csv")
DEFAULT_OUT = REPO / "experiments/results/ppc_gpu_urban_shadow_features_tokyo_run1_smoke.csv"
DEFAULT_SAT_JSON = REPO / "data/sample_satellites.json"

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


@dataclass(frozen=True)
class PathAlignment:
    mean_xy: np.ndarray
    axis_xy: np.ndarray
    cross_xy: np.ndarray
    z_median: float
    fallback_linear: bool = False


def _load_urban_module():
    path = HERE / "exp_urban_shadow_lab.py"
    spec = importlib.util.spec_from_file_location("exp_urban_shadow_lab", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


URBAN = _load_urban_module()


def _norm_key(key: str) -> str:
    return (
        key.strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _row_get(row: dict[str, str], *names: str) -> str | None:
    norm = {_norm_key(k): v for k, v in row.items()}
    for name in names:
        key = _norm_key(name)
        if key in norm:
            return norm[key]
    return None


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt_m) * cos_lat * cos_lon,
            (n + alt_m) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def _ecef_to_lla(ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = map(float, ecef)
    b = WGS84_A * (1.0 - WGS84_F)
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * WGS84_A, p * b)
    lat = math.atan2(
        z + WGS84_E2 / (1.0 - WGS84_E2) * b * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    lon = math.atan2(y, x)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def _ecef_to_enu(ecef: np.ndarray, origin_ecef: np.ndarray) -> np.ndarray:
    lat0_deg, lon0_deg, _ = _ecef_to_lla(origin_ecef)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return (np.asarray(ecef, dtype=np.float64) - origin_ecef) @ rot.T


def load_route_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return `(tow, route_local_xyz, route_ecef_or_none)`."""
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"route CSV is empty: {path}")

    tow_vals = []
    local_rows = []
    ecef_rows = []
    for idx, row in enumerate(rows):
        tow = _to_float(_row_get(row, "GPS TOW (s)", "tow", "gps_tow", "time"))
        if not np.isfinite(tow):
            tow = float(idx)

        x = _to_float(_row_get(row, "x_m", "x", "local_x_m", "enu_e_m"))
        y = _to_float(_row_get(row, "y_m", "y", "local_y_m", "enu_n_m"))
        z = _to_float(_row_get(row, "z_m", "z", "local_z_m", "enu_u_m"))
        ex = _to_float(_row_get(row, "ECEF X (m)", "ecef_x_m", "ecef_x"))
        ey = _to_float(_row_get(row, "ECEF Y (m)", "ecef_y_m", "ecef_y"))
        ez = _to_float(_row_get(row, "ECEF Z (m)", "ecef_z_m", "ecef_z"))
        lat = _to_float(_row_get(row, "Latitude (deg)", "LatitudeDegrees", "lat", "latitude_deg"))
        lon = _to_float(_row_get(row, "Longitude (deg)", "LongitudeDegrees", "lon", "longitude_deg"))
        alt = _to_float(_row_get(row, "Ellipsoid Height (m)", "Altitude (m)", "height_m", "alt"))

        if np.all(np.isfinite([x, y, z])):
            tow_vals.append(tow)
            local_rows.append([x, y, z])
            ecef_rows.append([np.nan, np.nan, np.nan])
        elif np.all(np.isfinite([ex, ey, ez])):
            tow_vals.append(tow)
            local_rows.append([np.nan, np.nan, np.nan])
            ecef_rows.append([ex, ey, ez])
        elif np.all(np.isfinite([lat, lon, alt])):
            tow_vals.append(tow)
            local_rows.append([np.nan, np.nan, np.nan])
            ecef_rows.append(_lla_to_ecef(lat, lon, alt).tolist())
        else:
            continue

    tow = np.asarray(tow_vals, dtype=np.float64)
    local = np.asarray(local_rows, dtype=np.float64)
    ecef = np.asarray(ecef_rows, dtype=np.float64)
    finite_local = np.all(np.isfinite(local), axis=1)
    finite_ecef = np.all(np.isfinite(ecef), axis=1)

    keep = finite_local | finite_ecef
    tow = tow[keep]
    local = local[keep]
    ecef = ecef[keep]
    finite_local = finite_local[keep]
    finite_ecef = finite_ecef[keep]

    if np.any(finite_ecef):
        ecef_valid = ecef[finite_ecef]
        origin = ecef_valid[0].copy()
        enu = _ecef_to_enu(ecef_valid, origin)
        local[finite_ecef] = enu
        full_ecef = np.full_like(local, np.nan)
        full_ecef[finite_ecef] = ecef_valid
        return tow, local.astype(np.float64), full_ecef
    return tow, local.astype(np.float64), None


def _select_epochs(
    tow: np.ndarray,
    local: np.ndarray,
    ecef: np.ndarray | None,
    *,
    max_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_epochs <= 0 or len(tow) <= max_epochs:
        return tow, local, ecef
    idx = np.unique(np.linspace(0, len(tow) - 1, max_epochs).round().astype(int))
    return tow[idx], local[idx], None if ecef is None else ecef[idx]


def _build_path_alignment(local: np.ndarray) -> PathAlignment:
    xy = np.asarray(local[:, :2], dtype=np.float64)
    z = np.asarray(local[:, 2], dtype=np.float64)
    mean_xy = np.mean(xy, axis=0)
    centered = xy - mean_xy
    if np.max(np.linalg.norm(centered, axis=1)) < 1.0:
        return PathAlignment(
            mean_xy=mean_xy,
            axis_xy=np.array([0.0, 1.0], dtype=np.float64),
            cross_xy=np.array([-1.0, 0.0], dtype=np.float64),
            z_median=float(np.nanmedian(z)),
            fallback_linear=True,
        )
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[1] < 0:
        axis = -axis
    cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    return PathAlignment(
        mean_xy=mean_xy,
        axis_xy=axis.astype(np.float64),
        cross_xy=cross.astype(np.float64),
        z_median=float(np.nanmedian(z)),
    )


def _path_aligned_route(local: np.ndarray, rx_height_m: float) -> np.ndarray:
    alignment = _build_path_alignment(local)
    xy = np.asarray(local[:, :2], dtype=np.float64)
    z = np.asarray(local[:, 2], dtype=np.float64)
    centered = xy - alignment.mean_xy
    if alignment.fallback_linear:
        y = np.arange(len(local), dtype=np.float64)
        y -= float(np.mean(y))
        x = np.zeros_like(y)
    else:
        y = centered @ alignment.axis_xy
        x = centered @ alignment.cross_xy
    z_rel = z - alignment.z_median
    out = np.column_stack([x, y, np.maximum(0.8, rx_height_m + z_rel)])
    return out.astype(np.float64)


def _rotate_enu_vectors_to_path_frame(vectors_enu: np.ndarray, alignment: PathAlignment) -> np.ndarray:
    vectors = np.asarray(vectors_enu, dtype=np.float64)
    original_shape = vectors.shape
    flat = vectors.reshape(-1, 3)
    xy = flat[:, :2]
    rotated = np.column_stack(
        [
            xy @ alignment.cross_xy,
            xy @ alignment.axis_xy,
            flat[:, 2],
        ]
    )
    norm = np.linalg.norm(rotated, axis=1, keepdims=True)
    rotated = rotated / np.maximum(norm, 1e-12)
    return rotated.reshape(original_shape).astype(np.float64)


def _sat_positions_for_particles(
    particles: np.ndarray,
    route: np.ndarray,
    sat_route: np.ndarray,
    *,
    particles_per_epoch: int,
) -> np.ndarray:
    n_epoch = route.shape[0]
    n_sat = sat_route.shape[1]
    particle_epoch = np.asarray(particles, dtype=np.float64).reshape(n_epoch, particles_per_epoch, 3)
    sat_vectors = np.asarray(sat_route, dtype=np.float64) - route[:, None, :]
    return (particle_epoch[:, :, None, :] + sat_vectors[:, None, :, :]).reshape(
        n_epoch * particles_per_epoch,
        n_sat,
        3,
    )


def _satellite_specs_from_path_directions(prns: list[str], directions_path: np.ndarray) -> list:
    dirs = np.asarray(directions_path, dtype=np.float64).reshape(-1, 3)
    specs = []
    for prn, d in zip(prns, dirs):
        az = math.degrees(math.atan2(float(d[0]), float(d[1]))) % 360.0
        el = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
        specs.append(URBAN.SatelliteSpec(str(prn), az, el))
    return specs


def _ensure_path_alignment(value: PathAlignment | np.ndarray) -> PathAlignment:
    if isinstance(value, PathAlignment):
        return value
    return _build_path_alignment(np.asarray(value, dtype=np.float64))


def _load_satellite_json(
    path: Path,
    route_ecef: np.ndarray | None,
    alignment: PathAlignment | np.ndarray,
) -> tuple[list, np.ndarray]:
    alignment = _ensure_path_alignment(alignment)
    if not path.exists():
        specs = URBAN._satellite_specs()
        return specs, URBAN._satellite_directions(specs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("satellites", payload if isinstance(payload, list) else [])
    if not items:
        specs = URBAN._satellite_specs()
        return specs, URBAN._satellite_directions(specs)

    if "azimuth_deg" in items[0] or "az_deg" in items[0]:
        specs = []
        for item in items:
            prn = str(item.get("prn", item.get("label", f"G{len(specs)+1:02d}")))
            az = float(item.get("azimuth_deg", item.get("az_deg")))
            el = float(item.get("elevation_deg", item.get("el_deg")))
            specs.append(URBAN.SatelliteSpec(prn, az, el))
        return specs, URBAN._satellite_directions(specs)

    if "ecef" in items[0] and route_ecef is not None and np.any(np.all(np.isfinite(route_ecef), axis=1)):
        origin_ecef = route_ecef[np.where(np.all(np.isfinite(route_ecef), axis=1))[0][0]]
        sat_ecef = np.asarray([item["ecef"] for item in items], dtype=np.float64)
        sat_local = _ecef_to_enu(sat_ecef, origin_ecef)
        dirs = sat_local / np.maximum(np.linalg.norm(sat_local, axis=1, keepdims=True), 1.0)
        dirs_path = _rotate_enu_vectors_to_path_frame(dirs, alignment)
        prns = [str(item.get("prn", f"G{idx+1:02d}")) for idx, item in enumerate(items)]
        return _satellite_specs_from_path_directions(prns, dirs_path), dirs_path

    specs = URBAN._satellite_specs()
    return specs, URBAN._satellite_directions(specs)


def _parse_constellation_list(value: str) -> tuple[str, ...]:
    systems = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    return systems or ("G",)


def _parse_prn_list(value: str, *, default_system: str) -> list[str] | None:
    if not value.strip():
        return None
    prns = []
    for raw in value.split(","):
        item = raw.strip().upper()
        if not item:
            continue
        if item[0].isalpha():
            system = item[0]
            number = item[1:]
        else:
            system = default_system
            number = item
        prns.append(f"{system}{int(number):02d}" if number.isdigit() else item)
    return prns or None


def _load_nav_satellite_route(
    *,
    nav_rinex: Path,
    tow: np.ndarray,
    route_ecef: np.ndarray | None,
    route: np.ndarray,
    alignment: PathAlignment,
    nav_systems: tuple[str, ...],
    nav_prns: list[str] | None,
    min_elevation_deg: float,
    sat_range_m: float,
) -> tuple[list, np.ndarray]:
    if route_ecef is None:
        raise ValueError("--nav-rinex requires route CSV rows with ECEF or LLA coordinates")
    route_ecef_arr = np.asarray(route_ecef, dtype=np.float64)
    finite_route = np.all(np.isfinite(route_ecef_arr), axis=1)
    if not np.all(finite_route):
        raise ValueError("--nav-rinex requires finite ECEF/LLA coordinates for every selected epoch")

    from gnss_gpu.ephemeris import Ephemeris
    from gnss_gpu.io.nav_rinex import read_nav_rinex_multi

    nav_messages = read_nav_rinex_multi(nav_rinex, systems=nav_systems)
    if not nav_messages:
        raise ValueError(f"no usable navigation messages found in {nav_rinex}")

    eph = Ephemeris(nav_messages)
    prn_list = nav_prns if nav_prns is not None else list(eph.available_prns)
    sat_ecef, _sat_clk, used_prns = eph.compute_batch(tow, prn_list=prn_list)
    if sat_ecef.shape[1] == 0 or not used_prns:
        raise ValueError(f"no satellite positions could be computed from {nav_rinex}")

    dirs_enu = np.zeros_like(sat_ecef, dtype=np.float64)
    elevations = np.zeros((sat_ecef.shape[0], sat_ecef.shape[1]), dtype=np.float64)
    for epoch in range(sat_ecef.shape[0]):
        enu = _ecef_to_enu(sat_ecef[epoch], route_ecef_arr[epoch])
        enu_norm = np.linalg.norm(enu, axis=1, keepdims=True)
        dirs = enu / np.maximum(enu_norm, 1.0)
        dirs_enu[epoch] = dirs
        elevations[epoch] = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))

    mean_elevations = np.mean(elevations, axis=0)
    keep = mean_elevations >= float(min_elevation_deg)
    if not np.any(keep):
        raise ValueError(
            "all computed satellites are below --min-sat-elevation-deg "
            f"({float(min_elevation_deg):.1f} deg)"
        )

    dirs_path = _rotate_enu_vectors_to_path_frame(dirs_enu[:, keep, :], alignment)
    sat_route = route[:, None, :] + dirs_path * float(sat_range_m)
    kept_prns = [str(prn) for prn, ok in zip(used_prns, keep) if bool(ok)]
    mean_dirs = np.mean(dirs_path, axis=0)
    mean_dirs /= np.maximum(np.linalg.norm(mean_dirs, axis=1, keepdims=True), 1e-12)
    specs = _satellite_specs_from_path_directions(kept_prns, mean_dirs)
    return specs, sat_route.astype(np.float64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_id", "tow"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_features(args: argparse.Namespace) -> list[dict]:
    tow, local, route_ecef = load_route_csv(args.route_csv)
    tow, local, route_ecef = _select_epochs(tow, local, route_ecef, max_epochs=args.max_epochs)
    alignment = _build_path_alignment(local)
    route = _path_aligned_route(local, args.rx_height_m)
    nav_rinex = getattr(args, "nav_rinex", None)
    if nav_rinex is not None:
        nav_systems = _parse_constellation_list(getattr(args, "nav_systems", "G"))
        specs, sat_route = _load_nav_satellite_route(
            nav_rinex=nav_rinex,
            tow=tow,
            route_ecef=route_ecef,
            route=route,
            alignment=alignment,
            nav_systems=nav_systems,
            nav_prns=_parse_prn_list(
                getattr(args, "nav_prns", ""),
                default_system=nav_systems[0],
            ),
            min_elevation_deg=getattr(args, "min_sat_elevation_deg", 5.0),
            sat_range_m=args.sat_range_m,
        )
        satellite_source = "nav_rinex"
    else:
        specs, directions = _load_satellite_json(args.satellite_json, route_ecef, alignment)
        sat_route = URBAN._sat_positions_for_receivers(route, directions, range_m=args.sat_range_m)
        satellite_source = "satellite_json"

    y_min = float(np.min(route[:, 1]))
    y_max = float(np.max(route[:, 1]))
    mesh_length = max(args.min_mesh_length_m, (y_max - y_min) + 2.0 * args.mesh_margin_m)
    triangles, _buildings = URBAN._build_canyon_mesh(
        length_m=mesh_length,
        block_depth_m=args.block_depth_m,
        road_half_width_m=args.road_half_width_m,
        building_width_m=args.building_width_m,
        base_height_m=args.base_height_m * args.building_height_scale,
        height_wave_m=args.height_wave_m * args.building_height_scale,
        n_blocks_per_side=args.n_blocks_per_side,
    )
    route_backend, _route_ms, route_los = URBAN._run_los_batch(
        route, sat_route, triangles, cpu_only=args.cpu_only
    )

    particles = URBAN._make_particles(
        route,
        particles_per_epoch=args.particles_per_epoch,
        seed=args.seed,
    )
    sat_particles = _sat_positions_for_particles(
        particles,
        route,
        sat_route,
        particles_per_epoch=args.particles_per_epoch,
    )
    particle_backend, _particle_ms, particle_los = URBAN._run_los_batch(
        particles, sat_particles, triangles, cpu_only=args.cpu_only
    )
    epoch_rows = URBAN._epoch_metrics(
        route=route,
        route_los=route_los,
        particle_los=particle_los,
        specs=specs,
        particles_per_epoch=args.particles_per_epoch,
        route_backend=route_backend,
        particle_backend=particle_backend,
    )

    out = []
    for metric, tow_val in zip(epoch_rows, tow):
        row = {
            "run_id": args.run_id,
            "tow": round(float(tow_val), 1),
            "gpu_urban_backend": route_backend,
            "gpu_urban_particle_backend": particle_backend,
            "gpu_urban_satellite_source": satellite_source,
            "gpu_urban_n_sat": float(metric.n_sat),
            "gpu_urban_n_los": float(metric.n_los),
            "gpu_urban_n_nlos": float(metric.n_nlos),
            "gpu_urban_mean_blocked_ratio": metric.blocked_ratio,
            "gpu_urban_max_blocked_ratio": metric.blocked_ratio,
            "gpu_urban_low_elev_blocked_ratio": metric.low_elevation_blocked_ratio,
            "gpu_urban_mean_elevation_los_deg": metric.mean_elevation_los_deg,
            "gpu_urban_mean_elevation_nlos_deg": metric.mean_elevation_nlos_deg,
            "gpu_urban_expected_nlos_bias_m": metric.expected_nlos_bias_m,
            "gpu_urban_route_weight_delta_log": metric.route_weight_delta_log,
            "gpu_urban_particle_blocked_mean": metric.particle_blocked_mean,
            "gpu_urban_particle_blocked_std": metric.particle_blocked_std,
            "gpu_urban_particle_shadow_contrast": metric.particle_shadow_contrast,
            "gpu_urban_particles_per_epoch": float(args.particles_per_epoch),
            "gpu_urban_building_height_scale": float(args.building_height_scale),
        }
        out.append(row)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--satellite-json", type=Path, default=DEFAULT_SAT_JSON)
    parser.add_argument(
        "--nav-rinex",
        type=Path,
        default=None,
        help="RINEX navigation file; when set, epoch satellite ECEF replaces --satellite-json.",
    )
    parser.add_argument(
        "--nav-systems",
        default="G",
        help="Comma-separated constellation IDs to parse from --nav-rinex.",
    )
    parser.add_argument(
        "--nav-prns",
        default="",
        help="Optional comma-separated PRN list such as G01,G03 or 1,3.",
    )
    parser.add_argument("--min-sat-elevation-deg", type=float, default=5.0)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default="tokyo_run1")
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260526)

    parser.add_argument("--particles-per-epoch", type=int, default=32)
    parser.add_argument("--rx-height-m", type=float, default=1.6)
    parser.add_argument("--sat-range-m", type=float, default=20_200_000.0)
    parser.add_argument("--building-height-scale", type=float, default=1.0)
    parser.add_argument("--min-mesh-length-m", type=float, default=360.0)
    parser.add_argument("--mesh-margin-m", type=float, default=80.0)
    parser.add_argument("--block-depth-m", type=float, default=44.0)
    parser.add_argument("--road-half-width-m", type=float, default=13.0)
    parser.add_argument("--building-width-m", type=float, default=26.0)
    parser.add_argument("--base-height-m", type=float, default=34.0)
    parser.add_argument("--height-wave-m", type=float, default=42.0)
    parser.add_argument("--n-blocks-per-side", type=int, default=9)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rows = extract_features(args)
    _write_csv(args.out_csv, rows)
    if rows:
        mean_blocked = float(np.mean([r["gpu_urban_mean_blocked_ratio"] for r in rows]))
        max_blocked = float(np.max([r["gpu_urban_max_blocked_ratio"] for r in rows]))
    else:
        mean_blocked = 0.0
        max_blocked = 0.0
    print(
        f"[ppc-gpu-shadow] wrote {args.out_csv} rows={len(rows)} "
        f"mean_blocked={mean_blocked:.3f} max_blocked={max_blocked:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
