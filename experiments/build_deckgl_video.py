#!/usr/bin/env python3
"""Generate LOS/NLOS verification video using deck.gl + Playwright.

Uses deck.gl's ArcLayer over OpenStreetMap tiles (no API key needed).
Lighter than CesiumJS — works in headless Chrome with swiftshader.
"""

import csv
import json
import math
import os
import subprocess
import time

import numpy as np

from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla


def load_trajectory(csv_path, step=200):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]), float(r[" ECEF Y (m)"]), float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


def generate_sats(rx_ecef, n_sat=10, time_offset=0.0):
    lat, lon, _ = ecef_to_lla(*rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    sat_ecef = np.zeros((n_sat, 3))
    for i in range(n_sat):
        el_deg = 10 + 70 * (i / max(n_sat - 1, 1))
        az_deg = (i * 36 + time_offset * 2) % 360
        el, az = math.radians(el_deg), math.radians(az_deg)
        r = 26600e3
        e = math.sin(az) * math.cos(el)
        n = math.cos(az) * math.cos(el)
        u = math.sin(el)
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u
        sat_ecef[i] = rx_ecef + r * np.array([dx, dy, dz])
    return sat_ecef


def compute_epochs(area_name, plateau_dir, traj_csv, n_epochs=10, step=300):
    print(f"[{area_name}] Loading PLATEAU + BVH...")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory(plateau_dir)
    bvh = BVHAccelerator.from_building_model(building)

    positions, times = load_trajectory(traj_csv, step=step)
    indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)

    n_sat = 10
    prn_list = list(range(1, n_sat + 1))
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    epochs = []
    traj_lla = []
    for p in positions:
        la, lo, al = ecef_to_lla(*p)
        traj_lla.append([math.degrees(la), math.degrees(lo)])

    for fi, ei in enumerate(indices):
        rx = positions[ei]
        t = times[ei] - times[0]
        sats = generate_sats(rx, n_sat, time_offset=t)
        result = usim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)

        la, lo, al = ecef_to_lla(*rx)
        rx_ll = [math.degrees(la), math.degrees(lo)]

        rays = []
        for i in range(n_sat):
            if not result["visible"][i]:
                continue
            direction = sats[i] - rx
            dist = np.linalg.norm(direction)
            ray_end = rx + direction / dist * 800  # 800m for map view
            re_la, re_lo, _ = ecef_to_lla(*ray_end)
            rays.append({
                "prn": prn_list[i],
                "los": bool(result["is_los"][i]),
                "el": float(np.degrees(result["elevations"][i])),
                "end": [math.degrees(re_la), math.degrees(re_lo)],
            })

        epochs.append({
            "rx": rx_ll, "rays": rays,
            "n_los": result["n_los"], "n_nlos": result["n_nlos"],
            "t": t,
        })
        print(f"  [{fi+1}/{n_epochs}] t={t:.0f}s LOS={result['n_los']} NLOS={result['n_nlos']}")

    return {"epochs": epochs, "trajectory": traj_lla, "area": area_name}


def generate_html(datasets, output_path):
    data_json = json.dumps(datasets)

    # Center on first epoch of first dataset
    center = datasets[0]["epochs"][0]["rx"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPU Urban GNSS Signal Simulator — LOS/NLOS</title>
<script src="https://unpkg.com/deck.gl@9.1.12/dist.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#0a0f1e; overflow:hidden; font-family:monospace; }}
  #map {{ width:100vw; height:100vh; }}
  #hud {{
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(10,15,30,0.92); color:#e0e0e0; padding:14px 18px;
    border-radius:10px; border:1px solid #334; min-width:240px;
  }}
  #hud h2 {{ margin:0 0 6px 0; font-size:15px; color:#fff; }}
  #hud .area {{ font-size:18px; color:#ffd93d; margin-bottom:4px; }}
  .los {{ color:#00d4aa; font-weight:bold; }}
  .nlos {{ color:#ff6b6b; font-weight:bold; }}
  #epoch {{ color:#888; font-size:11px; margin-top:4px; }}
  #legend {{ margin-top:8px; font-size:11px; }}
  #legend span {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
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
let dsIdx = 0, epIdx = 0;

const INITIAL_VIEW = {{
  longitude: {center[1]},
  latitude: {center[0]},
  zoom: 14.5,
  pitch: 55,
  bearing: -20,
}};

const deckgl = new deck.DeckGL({{
  container: 'map',
  initialViewState: INITIAL_VIEW,
  controller: false,
  layers: [],
  getTooltip: ({{object}}) => object && object.prn ? `PRN ${{object.prn}} ${{object.los?'LOS':'NLOS'}} el=${{object.el.toFixed(0)}}°` : null,
}});

function update() {{
  const ds = datasets[dsIdx];
  const ep = ds.epochs[epIdx];

  // Trajectory path
  const pathData = [{{ path: ds.trajectory.map(p => [p[1], p[0]]) }}];

  // Rays as arcs
  const arcData = ep.rays.map(r => ({{
    source: [ep.rx[1], ep.rx[0]],
    target: [r.end[1], r.end[0]],
    los: r.los, prn: r.prn, el: r.el,
  }}));

  // Receiver point
  const pointData = [{{ position: [ep.rx[1], ep.rx[0]], color: [255, 217, 61] }}];

  // Satellite endpoints
  const satData = ep.rays.map(r => ({{
    position: [r.end[1], r.end[0]],
    color: r.los ? [0, 212, 170] : [255, 107, 107],
    prn: r.prn, los: r.los, el: r.el,
  }}));

  deckgl.setProps({{
    viewState: {{
      ...INITIAL_VIEW,
      longitude: ep.rx[1],
      latitude: ep.rx[0],
      bearing: -20 + epIdx * 8,
      transitionDuration: 1200,
    }},
    layers: [
      new deck.PathLayer({{
        id: 'traj', data: pathData,
        getPath: d => d.path, getColor: [255, 217, 61, 100],
        widthMinPixels: 3, widthScale: 1,
      }}),
      new deck.ArcLayer({{
        id: 'rays', data: arcData,
        getSourcePosition: d => d.source,
        getTargetPosition: d => d.target,
        getSourceColor: d => d.los ? [0, 212, 170, 200] : [255, 107, 107, 200],
        getTargetColor: d => d.los ? [0, 212, 170, 120] : [255, 107, 107, 120],
        getWidth: d => d.los ? 2 : 4,
        getHeight: 0.3,
      }}),
      new deck.ScatterplotLayer({{
        id: 'rx', data: pointData,
        getPosition: d => d.position,
        getFillColor: d => d.color,
        getRadius: 30, radiusMinPixels: 8,
      }}),
      new deck.TextLayer({{
        id: 'labels', data: satData,
        getPosition: d => d.position,
        getText: d => 'PRN' + d.prn,
        getColor: d => d.color,
        getSize: 13, getAngle: 0,
        getTextAnchor: 'middle', getAlignmentBaseline: 'bottom',
        fontFamily: 'monospace', fontWeight: 'bold',
      }}),
    ],
  }});

  document.getElementById('area').textContent = ds.area;
  document.getElementById('stats').innerHTML =
    '<span class="los">LOS: ' + ep.n_los + '</span>  <span class="nlos">NLOS: ' + ep.n_nlos + '</span>';
  document.getElementById('epoch').textContent =
    'Epoch ' + (epIdx+1) + '/' + ds.epochs.length + '  t=' + ep.t.toFixed(0) + 's';

  epIdx++;
  if (epIdx >= ds.epochs.length) {{
    epIdx = 0;
    dsIdx = (dsIdx + 1) % datasets.length;
  }}
  setTimeout(update, 2000);
}}

setTimeout(update, 1000);
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML: {output_path}")


def record_video(html_path, output_dir, duration_ms=25000):
    """Record with Playwright."""
    print(f"Recording video ({duration_ms/1000:.0f}s)...")
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{
    headless: true,
    args: ['--use-angle=swiftshader','--enable-webgl','--ignore-gpu-blocklist']
  }});
  const ctx = await browser.newContext({{
    viewport: {{ width: 1280, height: 720 }},
    recordVideo: {{ dir: '{output_dir}', size: {{ width: 1280, height: 720 }} }},
  }});
  const page = await ctx.newPage();
  await page.goto('file://{os.path.abspath(html_path)}');
  await page.waitForTimeout({duration_ms});
  await ctx.close();
  await browser.close();
}})();
"""
    script_path = os.path.join(output_dir, "_rec.js")
    with open(script_path, "w") as f:
        f.write(script)
    subprocess.run(["node", script_path], timeout=duration_ms/1000+30,
                   cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")
    os.unlink(script_path)

    # Find and rename webm
    webms = sorted([f for f in os.listdir(output_dir) if f.endswith(".webm")])
    if webms:
        src = os.path.join(output_dir, webms[-1])
        dst = os.path.join(output_dir, "los_nlos_deckgl.webm")
        os.rename(src, dst)
        size_mb = os.path.getsize(dst) / 1e6
        print(f"Video: {dst} ({size_mb:.1f}MB)")

        # Convert to gif
        gif_path = os.path.join(output_dir, "los_nlos_deckgl.gif")
        subprocess.run([
            "ffmpeg", "-y", "-i", dst,
            "-vf", "fps=8,scale=800:-1:flags=lanczos",
            "-t", "24", gif_path
        ], capture_output=True)
        if os.path.exists(gif_path):
            size_mb = os.path.getsize(gif_path) / 1e6
            print(f"GIF: {gif_path} ({size_mb:.1f}MB)")
        return gif_path
    return None


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")

    shinjuku = compute_epochs("Shinjuku", "experiments/data/plateau_shinjuku",
                              "experiments/data/urbannav/Shinjuku/reference.csv", n_epochs=10)
    odaiba = compute_epochs("Odaiba", "experiments/data/plateau_odaiba",
                            "experiments/data/urbannav/Odaiba/reference.csv", n_epochs=10)

    html_path = os.path.join(out_dir, "los_nlos_deckgl.html")
    generate_html([shinjuku, odaiba], html_path)

    record_video(html_path, out_dir, duration_ms=25000)


if __name__ == "__main__":
    main()
