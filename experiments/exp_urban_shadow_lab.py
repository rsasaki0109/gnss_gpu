#!/usr/bin/env python3
"""Urban GNSS shadow lab.

Phase 2 MVP for gnss_gpu: build a synthetic urban canyon, cast route and
particle rays to GPS-like satellites, and summarize LOS/NLOS shadow structure.
CUDA BVH batch traversal is used when available; a NumPy Moller-Trumbore fallback
keeps the report reproducible without compiled bindings.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_SATELLITES = (
    ("G03", 80.0, 18.0),
    ("G07", 105.0, 28.0),
    ("G11", 135.0, 48.0),
    ("G14", 190.0, 62.0),
    ("G17", 250.0, 24.0),
    ("G19", 285.0, 36.0),
    ("G22", 320.0, 54.0),
    ("G27", 25.0, 16.0),
    ("G30", 350.0, 72.0),
    ("G31", 215.0, 12.0),
)


@dataclass(frozen=True)
class SatelliteSpec:
    prn: str
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class ShadowEpochMetrics:
    epoch: int
    y_m: float
    x_m: float
    n_sat: int
    n_los: int
    n_nlos: int
    blocked_ratio: float
    low_elevation_blocked_ratio: float
    mean_elevation_los_deg: float
    mean_elevation_nlos_deg: float
    expected_nlos_bias_m: float
    route_weight_delta_log: float
    particle_blocked_mean: float
    particle_blocked_std: float
    particle_shadow_contrast: float
    route_backend: str
    particle_backend: str


@dataclass(frozen=True)
class ShadowRunSummary:
    route_backend: str
    particle_backend: str
    n_epochs: int
    n_sat: int
    n_triangles: int
    n_buildings: int
    route_rays: int
    particle_rays: int
    route_los_ms: float
    particle_los_ms: float
    mean_blocked_ratio: float
    max_blocked_ratio: float
    mean_particle_shadow_contrast: float
    worst_epoch: int
    worst_epoch_blocked_ratio: float


def _box_triangles(center, width, depth, height) -> np.ndarray:
    cx, cy, cz = map(float, center)
    hw = float(width) / 2.0
    hd = float(depth) / 2.0
    hh = float(height) / 2.0
    v = np.array(
        [
            [cx - hw, cy - hd, cz - hh],
            [cx + hw, cy - hd, cz - hh],
            [cx + hw, cy + hd, cz - hh],
            [cx - hw, cy + hd, cz - hh],
            [cx - hw, cy - hd, cz + hh],
            [cx + hw, cy - hd, cz + hh],
            [cx + hw, cy + hd, cz + hh],
            [cx - hw, cy + hd, cz + hh],
        ],
        dtype=np.float64,
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 5, 1),
        (0, 4, 5),
        (2, 7, 3),
        (2, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
    )
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float64)


def _build_canyon_mesh(
    *,
    length_m: float,
    block_depth_m: float,
    road_half_width_m: float,
    building_width_m: float,
    base_height_m: float,
    height_wave_m: float,
    n_blocks_per_side: int,
) -> tuple[np.ndarray, list[dict]]:
    ys = np.linspace(
        -length_m / 2.0 + block_depth_m / 2.0,
        length_m / 2.0 - block_depth_m / 2.0,
        n_blocks_per_side,
    )
    centers_x = (
        -(road_half_width_m + building_width_m / 2.0),
        road_half_width_m + building_width_m / 2.0,
    )
    all_tris = []
    buildings: list[dict] = []
    for side_idx, cx in enumerate(centers_x):
        for block_idx, cy in enumerate(ys):
            height = base_height_m + height_wave_m * (
                0.5 + 0.5 * math.sin(block_idx * 1.7 + side_idx * 0.9)
            )
            center = (cx, float(cy), height / 2.0)
            tris = _box_triangles(center, building_width_m, block_depth_m, height)
            all_tris.append(tris)
            buildings.append(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "width": float(building_width_m),
                    "depth": float(block_depth_m),
                    "height": float(height),
                    "side": "west" if cx < 0 else "east",
                }
            )
    return np.concatenate(all_tris, axis=0), buildings


def _satellite_specs() -> list[SatelliteSpec]:
    return [SatelliteSpec(prn, az, el) for prn, az, el in DEFAULT_SATELLITES]


def _direction_from_az_el(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    return np.array(
        [
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )


def _satellite_directions(specs: list[SatelliteSpec]) -> np.ndarray:
    return np.vstack([_direction_from_az_el(s.azimuth_deg, s.elevation_deg) for s in specs])


def _make_route(n_epochs: int, length_m: float, rx_height_m: float) -> np.ndarray:
    y = np.linspace(-length_m / 2.0, length_m / 2.0, int(n_epochs))
    x = 1.8 * np.sin(np.linspace(0.0, 4.0 * np.pi, int(n_epochs)))
    z = np.full_like(y, float(rx_height_m))
    return np.column_stack([x, y, z]).astype(np.float64)


def _sat_positions_for_receivers(
    rx: np.ndarray,
    directions: np.ndarray,
    *,
    range_m: float,
) -> np.ndarray:
    rx_arr = np.asarray(rx, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    return rx_arr[:, None, :] + dirs[None, :, :] * float(range_m)


def _ray_intersects_any_triangle(origin: np.ndarray, end: np.ndarray, triangles: np.ndarray) -> bool:
    direction = end - origin
    max_t = float(np.linalg.norm(direction))
    if not np.isfinite(max_t) or max_t <= 0.0:
        return True
    direction = direction / max_t

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    mask = np.abs(det) > 1e-9
    if not np.any(mask):
        return False

    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hit = mask & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1e-6) & (t < max_t)
    return bool(np.any(hit))


def _cpu_los_batch(rx: np.ndarray, sat: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    rx_arr = np.asarray(rx, dtype=np.float64).reshape(-1, 3)
    sat_arr = np.asarray(sat, dtype=np.float64)
    if sat_arr.ndim != 3 or sat_arr.shape[0] != rx_arr.shape[0] or sat_arr.shape[2] != 3:
        raise ValueError("sat must have shape (N, n_sat, 3) and match rx")

    out = np.ones((rx_arr.shape[0], sat_arr.shape[1]), dtype=bool)
    for epoch in range(rx_arr.shape[0]):
        for sat_idx in range(sat_arr.shape[1]):
            if not np.all(np.isfinite(sat_arr[epoch, sat_idx])):
                out[epoch, sat_idx] = False
                continue
            out[epoch, sat_idx] = not _ray_intersects_any_triangle(
                rx_arr[epoch], sat_arr[epoch, sat_idx], triangles
            )
    return out


def _cuda_bvh_los_batch(rx: np.ndarray, sat: np.ndarray, triangles: np.ndarray) -> tuple[float, np.ndarray]:
    from gnss_gpu.bvh import BVHAccelerator

    bvh = BVHAccelerator(triangles)
    bvh.check_los_batch(rx[:1], sat[:1])
    start = time.perf_counter()
    los = np.asarray(bvh.check_los_batch(rx, sat), dtype=bool)
    return (time.perf_counter() - start) * 1000.0, los


def _run_los_batch(
    rx: np.ndarray,
    sat: np.ndarray,
    triangles: np.ndarray,
    *,
    cpu_only: bool,
) -> tuple[str, float, np.ndarray]:
    start = time.perf_counter()
    if not cpu_only:
        try:
            elapsed_ms, los = _cuda_bvh_los_batch(rx, sat, triangles)
            return "cuda_bvh_batch", elapsed_ms, los
        except Exception as exc:
            print(f"[urban-shadow-lab] CUDA BVH unavailable: {exc}")

    los = _cpu_los_batch(rx, sat, triangles)
    return "numpy_moller_trumbore", (time.perf_counter() - start) * 1000.0, los


def _make_particles(route: np.ndarray, *, particles_per_epoch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_epoch = route.shape[0]
    offsets = rng.normal(loc=0.0, scale=[5.0, 7.5, 0.0], size=(n_epoch, particles_per_epoch, 3))
    offsets[:, :, 2] = rng.normal(loc=0.0, scale=0.15, size=(n_epoch, particles_per_epoch))
    particles = route[:, None, :] + offsets
    particles[:, :, 2] = np.maximum(0.8, particles[:, :, 2])
    return particles.reshape(-1, 3).astype(np.float64)


def _weight_delta_from_los(los: np.ndarray, elevations_deg: np.ndarray) -> np.ndarray:
    blocked = ~np.asarray(los, dtype=bool)
    low_weight = np.clip(1.0 - elevations_deg / 80.0, 0.15, 1.0)
    return -np.sum(blocked * low_weight[None, :] * 0.65, axis=1)


def _epoch_metrics(
    *,
    route: np.ndarray,
    route_los: np.ndarray,
    particle_los: np.ndarray,
    specs: list[SatelliteSpec],
    particles_per_epoch: int,
    route_backend: str,
    particle_backend: str,
) -> list[ShadowEpochMetrics]:
    elevations = np.array([s.elevation_deg for s in specs], dtype=np.float64)
    low_mask = elevations < 30.0
    particle_los_epoch = particle_los.reshape(route.shape[0], particles_per_epoch, len(specs))
    particle_blocked = np.mean(~particle_los_epoch, axis=2)
    route_delta = _weight_delta_from_los(route_los, elevations)

    rows: list[ShadowEpochMetrics] = []
    for epoch in range(route.shape[0]):
        los = route_los[epoch]
        blocked = ~los
        los_elev = elevations[los]
        nlos_elev = elevations[blocked]
        low_blocked = float(np.mean(blocked[low_mask])) if np.any(low_mask) else 0.0
        blocked_ratio = float(np.mean(blocked))
        particle_mean = float(np.mean(particle_blocked[epoch]))
        particle_std = float(np.std(particle_blocked[epoch]))
        rows.append(
            ShadowEpochMetrics(
                epoch=int(epoch),
                y_m=float(route[epoch, 1]),
                x_m=float(route[epoch, 0]),
                n_sat=int(len(specs)),
                n_los=int(np.sum(los)),
                n_nlos=int(np.sum(blocked)),
                blocked_ratio=blocked_ratio,
                low_elevation_blocked_ratio=low_blocked,
                mean_elevation_los_deg=float(np.mean(los_elev)) if los_elev.size else 0.0,
                mean_elevation_nlos_deg=float(np.mean(nlos_elev)) if nlos_elev.size else 0.0,
                expected_nlos_bias_m=float(np.sum(blocked * np.clip(32.0 - elevations, 0.0, None) * 0.55)),
                route_weight_delta_log=float(route_delta[epoch]),
                particle_blocked_mean=particle_mean,
                particle_blocked_std=particle_std,
                particle_shadow_contrast=float(particle_std / max(particle_mean, 1e-9)),
                route_backend=route_backend,
                particle_backend=particle_backend,
            )
        )
    return rows


def _summary(
    rows: list[ShadowEpochMetrics],
    *,
    route_backend: str,
    particle_backend: str,
    n_sat: int,
    n_triangles: int,
    n_buildings: int,
    route_los_ms: float,
    particle_los_ms: float,
    particles_per_epoch: int,
) -> ShadowRunSummary:
    worst = max(rows, key=lambda row: row.blocked_ratio)
    return ShadowRunSummary(
        route_backend=route_backend,
        particle_backend=particle_backend,
        n_epochs=len(rows),
        n_sat=int(n_sat),
        n_triangles=int(n_triangles),
        n_buildings=int(n_buildings),
        route_rays=int(len(rows) * n_sat),
        particle_rays=int(len(rows) * particles_per_epoch * n_sat),
        route_los_ms=float(route_los_ms),
        particle_los_ms=float(particle_los_ms),
        mean_blocked_ratio=float(np.mean([row.blocked_ratio for row in rows])),
        max_blocked_ratio=float(worst.blocked_ratio),
        mean_particle_shadow_contrast=float(np.mean([row.particle_shadow_contrast for row in rows])),
        worst_epoch=int(worst.epoch),
        worst_epoch_blocked_ratio=float(worst.blocked_ratio),
    )


def _write_csv(path: Path, rows: list[ShadowEpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(
    path: Path,
    *,
    summary: ShadowRunSummary,
    rows: list[ShadowEpochMetrics],
    buildings: list[dict],
    satellites: list[SatelliteSpec],
) -> None:
    payload = {
        "experiment": "urban_shadow_lab_phase2_mvp",
        "summary": asdict(summary),
        "satellites": [asdict(sat) for sat in satellites],
        "buildings": buildings,
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bar(width: float, label: str, value: str, color: str) -> str:
    return (
        '<div class="bar-row">'
        f"<span>{html.escape(label)}</span>"
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{max(0.0, min(100.0, width)):.1f}%;background:{color}"></div>'
        "</div>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _write_html(
    path: Path,
    *,
    summary: ShadowRunSummary,
    rows: list[ShadowEpochMetrics],
    buildings: list[dict],
    satellites: list[SatelliteSpec],
    route: np.ndarray,
) -> None:
    worst_rows = sorted(rows, key=lambda row: row.blocked_ratio, reverse=True)[:8]
    max_bias = max(row.expected_nlos_bias_m for row in rows) or 1.0
    max_contrast = max(row.particle_shadow_contrast for row in rows) or 1.0

    y_min = min([b["y"] - b["depth"] / 2.0 for b in buildings] + [float(route[:, 1].min())])
    y_max = max([b["y"] + b["depth"] / 2.0 for b in buildings] + [float(route[:, 1].max())])
    x_min = min([b["x"] - b["width"] / 2.0 for b in buildings] + [float(route[:, 0].min())])
    x_max = max([b["x"] + b["width"] / 2.0 for b in buildings] + [float(route[:, 0].max())])
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)

    building_divs = []
    for building in buildings:
        left = (building["x"] - building["width"] / 2.0 - x_min) / span_x * 100.0
        bottom = (building["y"] - building["depth"] / 2.0 - y_min) / span_y * 100.0
        width = building["width"] / span_x * 100.0
        height = building["depth"] / span_y * 100.0
        shade = int(74 - min(34, building["height"] * 0.35))
        building_divs.append(
            f'<div class="building" style="left:{left:.2f}%;bottom:{bottom:.2f}%;'
            f'width:{width:.2f}%;height:{height:.2f}%;background:hsl(18 18% {shade}%);"></div>'
        )

    route_points = []
    for row in rows:
        left = (row.x_m - x_min) / span_x * 100.0
        bottom = (row.y_m - y_min) / span_y * 100.0
        red = int(80 + row.blocked_ratio * 150)
        green = int(150 - row.blocked_ratio * 95)
        route_points.append(
            f'<span class="route-dot" title="epoch {row.epoch}: {row.blocked_ratio:.2f}" '
            f'style="left:{left:.2f}%;bottom:{bottom:.2f}%;'
            f'background:rgb({red},{green},55);"></span>'
        )

    worst_table = "".join(
        "<tr>"
        f"<td>{row.epoch}</td>"
        f"<td>{row.y_m:.1f}</td>"
        f"<td>{row.n_nlos}/{row.n_sat}</td>"
        f"<td>{row.blocked_ratio:.2f}</td>"
        f"<td>{row.expected_nlos_bias_m:.1f}</td>"
        f"<td>{row.particle_shadow_contrast:.2f}</td>"
        "</tr>"
        for row in worst_rows
    )
    satellite_rows = "".join(
        "<tr>"
        f"<td>{html.escape(sat.prn)}</td>"
        f"<td>{sat.azimuth_deg:.0f}</td>"
        f"<td>{sat.elevation_deg:.0f}</td>"
        "</tr>"
        for sat in satellites
    )
    bias_bars = "".join(
        _bar(
            row.expected_nlos_bias_m / max_bias * 100.0,
            f"epoch {row.epoch}",
            f"{row.expected_nlos_bias_m:.1f} m",
            "#8b4b2f",
        )
        for row in worst_rows[:5]
    )
    contrast_bars = "".join(
        _bar(
            row.particle_shadow_contrast / max_contrast * 100.0,
            f"epoch {row.epoch}",
            f"{row.particle_shadow_contrast:.2f}",
            "#2f6673",
        )
        for row in sorted(rows, key=lambda row: row.particle_shadow_contrast, reverse=True)[:5]
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Urban GNSS Shadow Lab</title>
  <style>
    :root {{
      --ink: #172126;
      --muted: #5b676d;
      --line: #d8dee1;
      --bg: #f4f6f2;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 32px 6vw 20px;
      background: #eaf0ed;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    .sub {{
      max-width: 980px;
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    main {{
      padding: 24px 6vw 42px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .stat strong {{
      display: block;
      margin-top: 6px;
      font-size: 24px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 1.2fr) minmax(260px, 0.8fr);
      gap: 14px;
      align-items: start;
    }}
    @media (max-width: 860px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
    .map {{
      position: relative;
      height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(27,35,38,.05) 1px, transparent 1px),
        linear-gradient(0deg, rgba(27,35,38,.05) 1px, transparent 1px),
        #f7f8f5;
      background-size: 32px 32px;
    }}
    .building {{
      position: absolute;
      border: 1px solid rgba(40, 44, 46, 0.25);
    }}
    .route-dot {{
      position: absolute;
      width: 8px;
      height: 8px;
      margin-left: -4px;
      margin-bottom: -4px;
      border-radius: 50%;
      box-shadow: 0 0 0 1px rgba(0,0,0,.24);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 9px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      background: #edf1ee;
      color: #4e5a60;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .stack {{
      display: grid;
      gap: 14px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 82px 1fr 70px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      margin: 9px 0;
    }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: #e8ecee;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Urban GNSS Shadow Lab</h1>
    <p class="sub">
      Phase 2 MVP: route and particle LOS/NLOS rays through a synthetic urban canyon.
      Route backend <code>{html.escape(summary.route_backend)}</code>, particle backend
      <code>{html.escape(summary.particle_backend)}</code>.
    </p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><span>Mean Blocked</span><strong>{summary.mean_blocked_ratio:.2f}</strong></div>
      <div class="stat"><span>Max Blocked</span><strong>{summary.max_blocked_ratio:.2f}</strong></div>
      <div class="stat"><span>Route Rays</span><strong>{summary.route_rays}</strong></div>
      <div class="stat"><span>Particle Rays</span><strong>{summary.particle_rays}</strong></div>
      <div class="stat"><span>Route Time</span><strong>{summary.route_los_ms:.1f} ms</strong></div>
      <div class="stat"><span>Particle Time</span><strong>{summary.particle_los_ms:.1f} ms</strong></div>
    </section>
    <section class="layout">
      <div class="map">
        {''.join(building_divs)}
        {''.join(route_points)}
      </div>
      <div class="stack">
        <article class="panel">
          <h2>Worst Shadow Epochs</h2>
          <table>
            <thead><tr><th>Epoch</th><th>Y m</th><th>NLOS</th><th>Blocked</th><th>Bias</th><th>Contrast</th></tr></thead>
            <tbody>{worst_table}</tbody>
          </table>
        </article>
        <article class="panel">
          <h2>NLOS Bias Pressure</h2>
          {bias_bars}
        </article>
        <article class="panel">
          <h2>Particle Shadow Contrast</h2>
          {contrast_bars}
        </article>
        <article class="panel">
          <h2>Satellite Geometry</h2>
          <table>
            <thead><tr><th>PRN</th><th>Az</th><th>El</th></tr></thead>
            <tbody>{satellite_rows}</tbody>
          </table>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/urban_shadow_lab"))
    parser.add_argument("--n-epochs", type=int, default=72)
    parser.add_argument("--particles-per-epoch", type=int, default=48)
    parser.add_argument("--length-m", type=float, default=480.0)
    parser.add_argument("--block-depth-m", type=float, default=48.0)
    parser.add_argument("--road-half-width-m", type=float, default=13.0)
    parser.add_argument("--building-width-m", type=float, default=26.0)
    parser.add_argument("--base-height-m", type=float, default=34.0)
    parser.add_argument("--height-wave-m", type=float, default=42.0)
    parser.add_argument("--n-blocks-per-side", type=int, default=8)
    parser.add_argument("--rx-height-m", type=float, default=1.6)
    parser.add_argument("--sat-range-m", type=float, default=20_200_000.0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--cpu-only", action="store_true", help="Skip CUDA BVH and use NumPy fallback.")
    return parser


def run(args: argparse.Namespace) -> tuple[ShadowRunSummary, list[ShadowEpochMetrics]]:
    triangles, buildings = _build_canyon_mesh(
        length_m=args.length_m,
        block_depth_m=args.block_depth_m,
        road_half_width_m=args.road_half_width_m,
        building_width_m=args.building_width_m,
        base_height_m=args.base_height_m,
        height_wave_m=args.height_wave_m,
        n_blocks_per_side=args.n_blocks_per_side,
    )
    specs = _satellite_specs()
    directions = _satellite_directions(specs)
    route = _make_route(args.n_epochs, args.length_m, args.rx_height_m)
    sat_route = _sat_positions_for_receivers(route, directions, range_m=args.sat_range_m)
    route_backend, route_ms, route_los = _run_los_batch(
        route, sat_route, triangles, cpu_only=args.cpu_only
    )

    particles = _make_particles(route, particles_per_epoch=args.particles_per_epoch, seed=args.seed)
    sat_particles = _sat_positions_for_receivers(particles, directions, range_m=args.sat_range_m)
    particle_backend, particle_ms, particle_los = _run_los_batch(
        particles, sat_particles, triangles, cpu_only=args.cpu_only
    )

    rows = _epoch_metrics(
        route=route,
        route_los=route_los,
        particle_los=particle_los,
        specs=specs,
        particles_per_epoch=args.particles_per_epoch,
        route_backend=route_backend,
        particle_backend=particle_backend,
    )
    summary = _summary(
        rows,
        route_backend=route_backend,
        particle_backend=particle_backend,
        n_sat=len(specs),
        n_triangles=triangles.shape[0],
        n_buildings=len(buildings),
        route_los_ms=route_ms,
        particle_los_ms=particle_ms,
        particles_per_epoch=args.particles_per_epoch,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "urban_shadow_epoch_summary.csv", rows)
    _write_json(
        args.out_dir / "urban_shadow_summary.json",
        summary=summary,
        rows=rows,
        buildings=buildings,
        satellites=specs,
    )
    _write_html(
        args.out_dir / "urban_shadow_report.html",
        summary=summary,
        rows=rows,
        buildings=buildings,
        satellites=specs,
        route=route,
    )
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary, _rows = run(args)
    print(f"[urban-shadow-lab] wrote {args.out_dir / 'urban_shadow_epoch_summary.csv'}")
    print(f"[urban-shadow-lab] wrote {args.out_dir / 'urban_shadow_summary.json'}")
    print(f"[urban-shadow-lab] wrote {args.out_dir / 'urban_shadow_report.html'}")
    print(
        "[urban-shadow-lab] "
        f"route={summary.route_backend} {summary.route_los_ms:.2f}ms, "
        f"particles={summary.particle_backend} {summary.particle_los_ms:.2f}ms, "
        f"mean_blocked={summary.mean_blocked_ratio:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
