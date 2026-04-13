#!/usr/bin/env python3
"""Generate 3D LOS/NLOS geometry visualization + recording helper.

Creates a standalone HTML file using CesiumJS (free tier) that shows:
  - 3D globe with terrain + buildings (Cesium OSM Buildings)
  - Receiver trajectory (yellow line)
  - Per-epoch satellite rays (green=LOS, red=NLOS)
  - Animated flythrough

Then uses Playwright to record a video of the visualization.

The receiver trajectory is sourced from UrbanNav, while satellite rays use
a synthetic sky geometry to sanity-check LOS/NLOS behavior against buildings.
"""

import csv
import json
import math
import os
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
    """Generate synthetic satellite positions for visualization."""
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


def compute_all_epochs(area_name, plateau_dir, traj_csv, n_epochs=15, step=200):
    """Compute LOS/NLOS for all epochs and return visualization data."""
    print(f"[{area_name}] Loading PLATEAU...")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory(plateau_dir)
    print(f"  {len(building.triangles)} triangles")

    print(f"[{area_name}] Building BVH...")
    bvh = BVHAccelerator.from_building_model(building)

    positions, times = load_trajectory(traj_csv, step=step)
    indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)

    n_sat = 10
    prn_list = list(range(1, n_sat + 1))
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    epochs_data = []
    for fi, ei in enumerate(indices):
        rx = positions[ei]
        t = times[ei] - times[0]
        sats = generate_sats(rx, n_sat, time_offset=t)
        result = usim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)

        lat, lon, alt = ecef_to_lla(*rx)
        epoch = {
            "rx": [math.degrees(lat), math.degrees(lon), alt + 2],
            "rays": [],
            "n_los": result["n_los"],
            "n_nlos": result["n_nlos"],
            "t": t,
        }
        for i in range(n_sat):
            if not result["visible"][i]:
                continue
            s_lat, s_lon, s_alt = ecef_to_lla(*sats[i])
            # Shorten rays to ~5km for visibility
            direction = sats[i] - rx
            dist = np.linalg.norm(direction)
            ray_end = rx + direction / dist * 5000
            re_lat, re_lon, re_alt = ecef_to_lla(*ray_end)
            epoch["rays"].append({
                "prn": prn_list[i],
                "los": bool(result["is_los"][i]),
                "el": float(np.degrees(result["elevations"][i])),
                "end": [math.degrees(re_lat), math.degrees(re_lon), re_alt],
            })
        epochs_data.append(epoch)
        print(f"  [{fi+1}/{n_epochs}] t={t:.0f}s LOS={result['n_los']} NLOS={result['n_nlos']}")

    # Trajectory line
    traj = []
    for p in positions:
        lat, lon, alt = ecef_to_lla(*p)
        traj.append([math.degrees(lat), math.degrees(lon), alt + 1])

    return {"epochs": epochs_data, "trajectory": traj, "area": area_name}


def generate_html(datasets, output_path):
    """Generate standalone HTML with CesiumJS visualization."""
    data_json = json.dumps(datasets)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPU Urban GNSS Signal Simulator — LOS/NLOS 3D Verification</title>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }}
  #cesiumContainer {{ width: 100%; height: 100%; }}
  #overlay {{
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: rgba(10, 15, 30, 0.85); color: #e0e0e0; padding: 12px 16px;
    border-radius: 8px; font-family: monospace; font-size: 13px;
    border: 1px solid #334; min-width: 220px;
  }}
  #overlay h3 {{ margin: 0 0 8px 0; color: #fff; font-size: 14px; }}
  .los {{ color: #00d4aa; }}
  .nlos {{ color: #ff6b6b; }}
  #progress {{ margin-top: 6px; font-size: 11px; color: #888; }}
</style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="overlay">
  <h3>GPU Urban GNSS Signal Sim</h3>
  <div id="area">Loading...</div>
  <div id="stats"></div>
  <div id="progress"></div>
</div>
<script>
const viewer = new Cesium.Viewer('cesiumContainer', {{
  timeline: false,
  animation: false,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
}});

const ionToken = window.CESIUM_ION_TOKEN || '';
if (ionToken) {{
  Cesium.Ion.defaultAccessToken = ionToken;
  viewer.scene.setTerrain(Cesium.Terrain.fromWorldTerrain());
  Cesium.createOsmBuildingsAsync().then(tileset => {{
    viewer.scene.primitives.add(tileset);
  }});
}}

const datasets = {data_json};
let currentDataset = 0;
let currentEpoch = 0;
let rayEntities = [];

function clearRays() {{
  rayEntities.forEach(e => viewer.entities.remove(e));
  rayEntities = [];
}}

function showEpoch(ds, epochIdx) {{
  clearRays();
  const epoch = ds.epochs[epochIdx];
  const rx = epoch.rx;

  // Receiver marker
  rayEntities.push(viewer.entities.add({{
    position: Cesium.Cartesian3.fromDegrees(rx[1], rx[0], rx[2]),
    point: {{ pixelSize: 10, color: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 }},
    label: {{ text: 'RX', font: '12px monospace', fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE, outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM, pixelOffset: new Cesium.Cartesian2(0, -12) }},
  }}));

  // Satellite rays
  epoch.rays.forEach(ray => {{
    const color = ray.los ? Cesium.Color.fromCssColorString('#00d4aa').withAlpha(0.8)
                          : Cesium.Color.fromCssColorString('#ff6b6b').withAlpha(0.8);
    const width = ray.los ? 2 : 3;

    rayEntities.push(viewer.entities.add({{
      polyline: {{
        positions: Cesium.Cartesian3.fromDegreesArrayHeights([
          rx[1], rx[0], rx[2],
          ray.end[1], ray.end[0], ray.end[2]
        ]),
        width: width,
        material: color,
      }},
    }}));

    // Satellite label at ray end
    rayEntities.push(viewer.entities.add({{
      position: Cesium.Cartesian3.fromDegrees(ray.end[1], ray.end[0], ray.end[2]),
      label: {{
        text: 'PRN' + ray.prn + (ray.los ? ' LOS' : ' NLOS'),
        font: '11px monospace',
        fillColor: color,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        outlineWidth: 1,
        scale: 0.8,
      }},
    }}));
  }});

  // Update overlay
  document.getElementById('area').textContent = ds.area;
  const losCount = epoch.rays.filter(r => r.los).length;
  const nlosCount = epoch.rays.filter(r => !r.los).length;
  document.getElementById('stats').innerHTML =
    '<span class="los">LOS: ' + losCount + '</span> | <span class="nlos">NLOS: ' + nlosCount + '</span>';
  document.getElementById('progress').textContent =
    'Epoch ' + (epochIdx+1) + '/' + ds.epochs.length + '  t=' + epoch.t.toFixed(0) + 's';
}}

// Draw trajectory
function drawTrajectory(ds) {{
  const coords = [];
  ds.trajectory.forEach(p => {{ coords.push(p[1], p[0], p[2]); }});
  viewer.entities.add({{
    polyline: {{
      positions: Cesium.Cartesian3.fromDegreesArrayHeights(coords),
      width: 3,
      material: Cesium.Color.YELLOW.withAlpha(0.7),
      clampToGround: true,
    }},
  }});
}}

// Animation
function animate() {{
  const ds = datasets[currentDataset];
  showEpoch(ds, currentEpoch);
  currentEpoch = (currentEpoch + 1) % ds.epochs.length;

  // Switch dataset after full loop
  if (currentEpoch === 0 && datasets.length > 1) {{
    // Clear trajectory
    viewer.entities.removeAll();
    currentDataset = (currentDataset + 1) % datasets.length;
    drawTrajectory(datasets[currentDataset]);

    // Fly to new area
    const ep = datasets[currentDataset].epochs[0];
    viewer.camera.flyTo({{
      destination: Cesium.Cartesian3.fromDegrees(ep.rx[1], ep.rx[0], 800),
      orientation: {{ heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 }},
      duration: 2,
    }});
    setTimeout(animate, 3000);
    return;
  }}
  setTimeout(animate, 1500);
}}

// Initial camera
const firstEpoch = datasets[0].epochs[0];
viewer.camera.setView({{
  destination: Cesium.Cartesian3.fromDegrees(firstEpoch.rx[1], firstEpoch.rx[0], 800),
  orientation: {{ heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 }},
}});
drawTrajectory(datasets[0]);

// Start after buildings load
setTimeout(animate, 3000);
</script>
</body>
</html>"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML: {output_path}")


def record_video(html_path, video_path, duration_ms=30000):
    """Record video of the visualization using Playwright."""
    import subprocess
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{ headless: true }});
  const context = await browser.newContext({{
    viewport: {{ width: 1280, height: 720 }},
    recordVideo: {{ dir: '{os.path.dirname(video_path)}', size: {{ width: 1280, height: 720 }} }},
  }});
  const page = await context.newPage();
  await page.goto('file://{os.path.abspath(html_path)}');
  await page.waitForTimeout({duration_ms});
  await context.close();
  await browser.close();
  // Rename video
  const fs = require('fs');
  const files = fs.readdirSync('{os.path.dirname(video_path)}')
    .filter(f => f.endsWith('.webm'));
  if (files.length > 0) {{
    fs.renameSync(
      '{os.path.dirname(video_path)}/' + files[files.length - 1],
      '{video_path}'
    );
  }}
}})();
"""
    script_path = video_path + ".js"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Recording video ({duration_ms / 1000:.0f}s)...")
    result = subprocess.run(
        ["node", script_path],
        capture_output=True, text=True, timeout=duration_ms / 1000 + 30,
        cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
    )
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:500]}")
    os.unlink(script_path)
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / 1e6
        print(f"Video: {video_path} ({size_mb:.1f} MB)")
    else:
        print("Video recording failed — HTML file can still be opened in a browser")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")

    # Compute Shinjuku data
    shinjuku = compute_all_epochs(
        "Shinjuku", "experiments/data/plateau_shinjuku",
        "experiments/data/urbannav/Shinjuku/reference.csv",
        n_epochs=12, step=300,
    )

    # Compute Odaiba data
    odaiba = compute_all_epochs(
        "Odaiba", "experiments/data/plateau_odaiba",
        "experiments/data/urbannav/Odaiba/reference.csv",
        n_epochs=12, step=300,
    )

    # Generate HTML
    html_path = os.path.join(out_dir, "los_nlos_3d.html")
    generate_html([shinjuku, odaiba], html_path)

    # Record video
    video_path = os.path.join(out_dir, "los_nlos_3d.webm")
    try:
        record_video(html_path, video_path, duration_ms=35000)
    except Exception as e:
        print(f"Video recording skipped: {e}")
        print("Open the HTML file in a browser to view the 3D visualization")


if __name__ == "__main__":
    main()
