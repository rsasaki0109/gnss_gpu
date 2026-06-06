#!/usr/bin/env python3
"""Generate a standalone HTML visualization for the PLATEAU NLOS demo.

Run from the repo root:

    PYTHONPATH=python python3 examples/demo_plateau_nlos_visualization.py

The default output is ``/tmp/gnss_gpu_plateau_nlos_viz.html``. The file is
self-contained and uses inline SVG, so it can be opened directly in a browser.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_OUTPUT = Path("/tmp/gnss_gpu_plateau_nlos_viz.html")


def _load_plateau_demo_module():
    module_path = Path(__file__).resolve().parent / "demo_plateau_nlos_simulation.py"
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    return (
        f'<polyline points="{_points_attr(points)}" fill="none" '
        f'stroke="{color}" stroke-width="{width:.1f}" stroke-linejoin="round" '
        'stroke-linecap="round"/>'
    )


def _scale_xy(
    xy: np.ndarray,
    width: int,
    height: int,
    margin: int,
) -> tuple[Callable[[float, float], tuple[float, float]], tuple[float, float, float, float]]:
    xmin = float(np.min(xy[:, 0]))
    xmax = float(np.max(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    ymax = float(np.max(xy[:, 1]))
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)
    xpad = 0.5 * ((width - 2 * margin) / scale - dx)
    ypad = 0.5 * ((height - 2 * margin) / scale - dy)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def _map(x: float, y: float) -> tuple[float, float]:
        sx = margin + (x - xmin) * scale
        sy = height - margin - (y - ymin) * scale
        return sx, sy

    return _map, (xmin, xmax, ymin, ymax)


def _scene_svg(details: dict[str, object]) -> str:
    triangles = np.asarray(details["triangles_enu"], dtype=np.float64)
    rx = np.asarray(details["rx_enu"], dtype=np.float64)
    naive = np.asarray(details["naive_error_m"], dtype=np.float64)
    plateau = np.asarray(details["plateau_error_m"], dtype=np.float64)
    worst_idx = int(np.argmax(naive))

    width, height, margin = 880, 500, 34
    all_xy = np.vstack([triangles.reshape(-1, 3)[:, :2], rx[:, :2]])
    map_xy, bounds = _scale_xy(all_xy, width, height, margin)

    polygons = []
    for tri in triangles:
        pts = [map_xy(float(p[0]), float(p[1])) for p in tri]
        polygons.append(
            f'<polygon points="{_points_attr(pts)}" fill="#d8dee6" '
            'stroke="#798694" stroke-width="0.8" opacity="0.82"/>'
        )

    traj = [map_xy(float(p[0]), float(p[1])) for p in rx]
    start = traj[0]
    end = traj[-1]
    worst = traj[worst_idx]
    xmin, xmax, ymin, ymax = bounds
    label = (
        f"ENU bounds: E {xmin:.0f}..{xmax:.0f} m, "
        f"N {ymin:.0f}..{ymax:.0f} m"
    )
    gain = naive[worst_idx] - plateau[worst_idx]

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="PLATEAU mesh and receiver trajectory">
  <rect width="{width}" height="{height}" fill="#f7f9fb"/>
  <g>{''.join(polygons)}</g>
  {_polyline(traj, "#1f5eff", 3.0)}
  <circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="5" fill="#168a47"/>
  <circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="5" fill="#111827"/>
  <circle cx="{worst[0]:.1f}" cy="{worst[1]:.1f}" r="7" fill="#d62828" opacity="0.88"/>
  <text x="22" y="28" class="svg-title">PLATEAU sample mesh + receiver trajectory</text>
  <text x="22" y="52" class="svg-note">{html.escape(label)}</text>
  <text x="22" y="76" class="svg-note">Red point: worst naive epoch, PLATEAU-aware gain {gain:.1f} m</text>
  <g class="legend">
    <rect x="{width - 190}" y="22" width="16" height="10" fill="#d8dee6" stroke="#798694"/>
    <text x="{width - 166}" y="32">PLATEAU triangles</text>
    <line x1="{width - 190}" y1="52" x2="{width - 172}" y2="52" stroke="#1f5eff" stroke-width="3"/>
    <text x="{width - 166}" y="56">trajectory</text>
  </g>
</svg>"""


def _sky_svg(details: dict[str, object]) -> str:
    azel = np.asarray(details["satellite_az_el_deg"], dtype=np.float64)
    los_mask = np.asarray(details["los_mask"], dtype=bool)
    naive = np.asarray(details["naive_error_m"], dtype=np.float64)
    worst_idx = int(np.argmax(naive))
    is_los = los_mask[worst_idx]
    nlos_count = int(np.sum(~is_los))

    width, height = 420, 420
    cx, cy, radius = 210.0, 218.0, 154.0
    rings = []
    for elev in (0, 30, 60):
        r = radius * (90.0 - elev) / 90.0
        rings.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="#d5dde8"/>')
        rings.append(f'<text x="{cx + r + 5:.1f}" y="{cy - 4:.1f}" class="svg-note">{elev} deg</text>')

    sats = []
    for i, (az_deg, el_deg) in enumerate(azel):
        az = np.radians(az_deg)
        r = radius * (90.0 - el_deg) / 90.0
        x = cx + r * np.sin(az)
        y = cy - r * np.cos(az)
        color = "#168a47" if is_los[i] else "#d62828"
        sats.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>')
        sats.append(
            f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" class="sat-label">S{i + 1}</text>'
        )

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Sky plot at worst naive epoch">
  <rect width="{width}" height="{height}" fill="#f7f9fb"/>
  <text x="22" y="30" class="svg-title">Sky plot at worst naive epoch</text>
  <text x="22" y="54" class="svg-note">Epoch {worst_idx}: {nlos_count}/{len(is_los)} satellites NLOS</text>
  <line x1="{cx - radius}" y1="{cy}" x2="{cx + radius}" y2="{cy}" stroke="#d5dde8"/>
  <line x1="{cx}" y1="{cy - radius}" x2="{cx}" y2="{cy + radius}" stroke="#d5dde8"/>
  {''.join(rings)}
  {''.join(sats)}
  <g class="legend">
    <circle cx="24" cy="386" r="7" fill="#168a47"/><text x="38" y="391">LOS</text>
    <circle cx="92" cy="386" r="7" fill="#d62828"/><text x="106" y="391">NLOS</text>
  </g>
</svg>"""


def _error_svg(details: dict[str, object]) -> str:
    naive = np.asarray(details["naive_error_m"], dtype=np.float64)
    robust = np.asarray(details["robust_error_m"], dtype=np.float64)
    plateau = np.asarray(details["plateau_error_m"], dtype=np.float64)
    nlos_count = np.asarray(details["nlos_count"], dtype=np.float64)
    epochs = np.asarray(details["epoch_index"], dtype=np.float64)

    width, height, margin = 880, 330, 44
    ymax = max(float(np.max(naive)), float(np.max(robust)), float(np.max(plateau)), 1.0) * 1.12
    xmin = float(np.min(epochs))
    xmax = float(np.max(epochs))
    dx = max(xmax - xmin, 1.0)

    def map_xy(epoch: float, value: float) -> tuple[float, float]:
        x = margin + (epoch - xmin) / dx * (width - 2 * margin)
        y = height - margin - value / ymax * (height - 2 * margin)
        return x, y

    bars = []
    max_nlos = max(float(np.max(nlos_count)), 1.0)
    bar_w = (width - 2 * margin) / max(len(epochs), 1)
    for epoch, count in zip(epochs, nlos_count):
        x, _ = map_xy(float(epoch), 0.0)
        h = (count / max_nlos) * 46.0
        bars.append(
            f'<rect x="{x - 0.45 * bar_w:.1f}" y="{height - margin - h:.1f}" '
            f'width="{0.9 * bar_w:.1f}" height="{h:.1f}" fill="#dbe4ee"/>'
        )

    naive_line = _polyline([map_xy(e, v) for e, v in zip(epochs, naive)], "#c65a00", 2.4)
    robust_line = _polyline([map_xy(e, v) for e, v in zip(epochs, robust)], "#7c3aed", 2.0)
    plateau_line = _polyline([map_xy(e, v) for e, v in zip(epochs, plateau)], "#1f5eff", 2.6)

    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = height - margin - frac * (height - 2 * margin)
        grid.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width - margin}" y2="{y:.1f}" stroke="#e3e9f0"/>')
        grid.append(f'<text x="8" y="{y + 4:.1f}" class="svg-note">{frac * ymax:.0f}m</text>')

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Positioning error timeline">
  <rect width="{width}" height="{height}" fill="#f7f9fb"/>
  <text x="22" y="30" class="svg-title">Positioning error timeline</text>
  {''.join(grid)}
  {''.join(bars)}
  {naive_line}
  {robust_line}
  {plateau_line}
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#9aa7b5"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#9aa7b5"/>
  <text x="{width - margin - 58}" y="{height - 12}" class="svg-note">epoch</text>
  <g class="legend">
    <line x1="560" y1="28" x2="584" y2="28" stroke="#c65a00" stroke-width="3"/><text x="592" y="33">naive</text>
    <line x1="650" y1="28" x2="674" y2="28" stroke="#7c3aed" stroke-width="3"/><text x="682" y="33">robust</text>
    <line x1="744" y1="28" x2="768" y2="28" stroke="#1f5eff" stroke-width="3"/><text x="776" y="33">PLATEAU-aware</text>
  </g>
</svg>"""


def build_html(result: dict[str, object]) -> str:
    details = result["details"]
    scene_svg = _scene_svg(details)
    sky_svg = _sky_svg(details)
    error_svg = _error_svg(details)
    gain = 100.0 * (1.0 - result["plateau_rms_m"] / result["naive_rms_m"])

    cards = [
        ("Ray Source", str(result["ray_source"])),
        ("NLOS Signals", f"{100.0 * result['nlos_fraction']:.1f}%"),
        ("Naive RMS", f"{result['naive_rms_m']:.2f} m"),
        ("PLATEAU-aware SPP RMS", f"{result['plateau_rms_m']:.2f} m"),
        ("RMS Gain", f"{gain:.0f}%"),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLATEAU NLOS Visualization</title>
<style>
body {{
  margin: 0;
  background: #eef2f6;
  color: #17202a;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 28px;
  letter-spacing: 0;
}}
p {{
  margin: 0 0 18px;
  color: #52606d;
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin: 18px 0;
}}
.card {{
  background: #ffffff;
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  padding: 12px 14px;
}}
.label {{
  color: #607080;
  font-size: 13px;
}}
.value {{
  font-size: 22px;
  font-weight: 700;
  margin-top: 4px;
}}
.grid {{
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
  gap: 14px;
  align-items: stretch;
}}
.panel {{
  background: #ffffff;
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  padding: 10px;
}}
.panel svg {{
  width: 100%;
  height: auto;
  display: block;
}}
.wide {{
  margin-top: 14px;
}}
.svg-title {{
  font-size: 18px;
  font-weight: 700;
  fill: #17202a;
}}
.svg-note, .legend text {{
  font-size: 12px;
  fill: #52606d;
}}
.sat-label {{
  font-size: 12px;
  font-weight: 700;
  fill: #17202a;
}}
@media (max-width: 820px) {{
  main {{ padding: 16px; }}
  .grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <h1>PLATEAU NLOS Visualization</h1>
  <p>Standalone view of the sample CityGML mesh, BVH/triangle LOS mask, simulated NLOS measurements, and SPP error recovery.</p>
  <section class="cards">{card_html}</section>
  <section class="grid">
    <div class="panel">{scene_svg}</div>
    <div class="panel">{sky_svg}</div>
  </section>
  <section class="panel wide">{error_svg}</section>
</main>
</body>
</html>
"""


def render_visualization(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    gml_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    demo = _load_plateau_demo_module()
    result = demo.run_demo(gml_path=gml_path, include_details=True)
    html_text = build_html(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path, result


def main() -> tuple[Path, dict[str, object]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gml", type=Path, default=None)
    args = parser.parse_args()
    output_path, result = render_visualization(args.output, gml_path=args.gml)
    print(f"Wrote {output_path}")
    print(
        f"PLATEAU-aware RMS {result['plateau_rms_m']:.2f} m vs "
        f"naive {result['naive_rms_m']:.2f} m"
    )
    return output_path, result


if __name__ == "__main__":
    main()
