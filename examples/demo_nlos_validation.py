"""
CPU-only NLOS simulation and validation demo.

Run:
    PYTHONPATH=python python3 examples/demo_nlos_validation.py

This demo intentionally avoids CUDA extensions, GPU IQ generation, and
UrbanSignalSimulator. It uses pure-Python geometry reflection paths plus a
small local Moller-Trumbore LOS test.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_gpu.fresnel import reflection_coefficient
from gnss_gpu.raytrace import BuildingModel, horizontal_ground_plane
from gnss_gpu.validation import (
    bin_by_elevation,
    bin_by_los,
    compare_distributions,
    records_from_epoch,
    residual_array,
    summarize,
    write_csv,
)

C_MPS = 299792458.0
CHIP_M = C_MPS / 1.023e6
BETA = 0.5
SAT_RANGE_M = 2.0e7


def _box_triangles(center, width, depth, height):
    box = BuildingModel.create_box(np.asarray(center, dtype=float), width, depth, height)
    if hasattr(box, "triangles"):
        return np.asarray(box.triangles, dtype=float)
    return np.asarray(box, dtype=float)


def _make_scene():
    specs = [
        (-25.0, -72.0, 18.0, 30.0, 52.0),
        (27.0, -68.0, 20.0, 32.0, 46.0),
        (-26.0, -32.0, 22.0, 36.0, 58.0),
        (28.0, -25.0, 18.0, 30.0, 42.0),
        (-25.0, 12.0, 20.0, 34.0, 50.0),
        (27.0, 14.0, 18.0, 36.0, 60.0),
        (-27.0, 50.0, 22.0, 38.0, 45.0),
        (29.0, 56.0, 20.0, 34.0, 54.0),
        (-26.0, 88.0, 18.0, 30.0, 48.0),
        (28.0, 92.0, 22.0, 32.0, 57.0),
    ]

    triangles = []
    for cx, cy, width, depth, height in specs:
        triangles.append(_box_triangles([cx, cy, height * 0.5], width, depth, height))

    return BuildingModel(np.vstack(triangles)), len(specs)


def _trajectory(n_epochs, rx_height):
    t = np.linspace(-1.0, 1.0, n_epochs)
    y = 96.0 * t
    x = 3.0 * np.sin(1.3 * np.pi * t)
    z = np.full_like(x, rx_height)
    return np.column_stack([x, y, z])


def _directions_from_azel(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    ce = np.cos(el)
    return np.column_stack(
        [
            ce * np.sin(az),
            ce * np.cos(az),
            np.sin(el),
        ]
    )


def _satellite_positions(epoch, n_epochs):
    base_az = np.array([90, 270, 70, 250, 110, 290, 35, 215, 150, 330, 0, 180], dtype=float)
    base_el = np.array([6, 8, 12, 16, 24, 32, 42, 52, 62, 74, 28, 82], dtype=float)

    phase = 2.0 * np.pi * epoch / max(n_epochs - 1, 1)
    drift = np.arange(base_az.size, dtype=float)
    az = base_az + 6.0 * np.sin(phase + 0.37 * drift)
    el = np.clip(base_el + 2.0 * np.sin(0.7 * phase + drift), 5.5, 85.0)

    return SAT_RANGE_M * _directions_from_azel(az, el)


def _azel_from_positions(rx, sat_positions):
    rel = sat_positions - rx[None, :]
    east = rel[:, 0]
    north = rel[:, 1]
    up = rel[:, 2]
    horiz = np.hypot(east, north)

    el = np.rad2deg(np.arctan2(up, horiz))
    az = np.rad2deg(np.arctan2(east, north)) % 360.0
    return az, el


def _segment_intersects_triangle(p0, p1, tri, eps=1.0e-9):
    direction = p1 - p0
    v0, v1, v2 = tri
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if abs(a) < eps:
        return False

    f = 1.0 / a
    s = p0 - v0
    u = f * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return False

    q = np.cross(s, edge1)
    v = f * float(np.dot(direction, q))
    if v < -eps or u + v > 1.0 + eps:
        return False

    t = f * float(np.dot(edge2, q))
    return eps < t < 1.0 - eps


def _is_los(rx, sat, triangles):
    for tri in triangles:
        if _segment_intersects_triangle(rx, sat, tri):
            return False
    return True


def _path_value(path, name):
    value = getattr(path, name)
    if callable(value):
        value = value()
    return value


def _path_excess_delay(path):
    return float(_path_value(path, "excess_delay"))


def _path_triangle_id(path):
    return int(_path_value(path, "triangle_id"))


def _path_incidence_angle(path):
    return float(_path_value(path, "incidence_angle"))


def _path_amplitude(path, wall_material, ground_material):
    material = ground_material if _path_triangle_id(path) == -1 else wall_material
    amp = reflection_coefficient(_path_incidence_angle(path), material=material)
    return float(np.clip(amp, 0.0, 1.0))


def _simulate_epoch(
    model,
    rx,
    sat_positions,
    paths_by_sat,
    rng,
    wall_material,
    ground_material,
    elevation_mask_deg=5.0,
):
    azimuths, elevations = _azel_from_positions(rx, sat_positions)
    visible = elevations >= elevation_mask_deg
    is_los = np.zeros(sat_positions.shape[0], dtype=bool)
    residuals = np.full(sat_positions.shape[0], np.nan, dtype=float)
    cn0 = np.full(sat_positions.shape[0], np.nan, dtype=float)

    for i, sat in enumerate(sat_positions):
        if not visible[i]:
            continue

        is_los[i] = _is_los(rx, sat, model.triangles)
        paths = list(paths_by_sat[i])

        if is_los[i]:
            num = 0.0
            den = 1.0
            for path in paths:
                delay = _path_excess_delay(path)
                if 0.0 <= delay < CHIP_M:
                    amp = _path_amplitude(path, wall_material, ground_material)
                    num += amp * delay
                    den += amp
            residuals[i] = BETA * num / den + rng.normal(0.0, 0.8)
            cn0[i] = np.clip(45.0 + rng.normal(0.0, 2.0), 20.0, 55.0)
            continue

        if not paths:
            visible[i] = False
            continue

        scored = [
            (_path_amplitude(path, wall_material, ground_material), _path_excess_delay(path))
            for path in paths
        ]
        amp, delay = max(scored, key=lambda item: item[0])
        if amp <= 0.0:
            visible[i] = False
            continue

        residuals[i] = delay + rng.normal(0.0, 3.0)
        cn0[i] = np.clip(30.0 + rng.normal(0.0, 4.0), 12.0, 45.0)

    # ResidualSample stores angles in radians; az/el above are in degrees for
    # the elevation mask, so convert before handing them to the records layer.
    return (
        residuals,
        np.radians(elevations),
        np.radians(azimuths),
        is_los,
        visible,
        cn0,
    )


def _finite_residuals(samples):
    values = np.asarray(residual_array(samples), dtype=float)
    return values[np.isfinite(values)]


def _summary(samples):
    values = _finite_residuals(samples)
    if values.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "rms": math.nan,
            "mae": math.nan,
            "p50": math.nan,
            "p68": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "abs_p50": math.nan,
            "abs_p95": math.nan,
        }
    return summarize(values)


def _format_summary(label, stats):
    return (
        f"{label:>10s}  n={int(stats['count']):4d}  "
        f"mean={stats['mean']:8.2f} m  rms={stats['rms']:8.2f} m  "
        f"p50={stats['p50']:8.2f} m  p95={stats['p95']:8.2f} m"
    )


def _make_reference_distribution(rng, n, nlos_fraction):
    # Placeholder UrbanNav-like reference distribution.
    # Replace this synthetic mixture with real residuals for calibration.
    n_ref = max(1000, 3 * n)
    is_nlos = rng.random(n_ref) < nlos_fraction
    ref = rng.normal(0.0, 4.0, n_ref)

    n_nlos = int(np.count_nonzero(is_nlos))
    ref[is_nlos] = np.abs(rng.normal(20.0, 30.0, n_nlos))
    return ref


def _plot_cdf(sim, reference, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)

    for values, label in [(sim, "simulation"), (reference, "reference placeholder")]:
        values = np.sort(np.asarray(values, dtype=float))
        y = np.arange(1, values.size + 1, dtype=float) / values.size
        ax.plot(values, y, linewidth=2.0, label=label)

    ax.set_xlabel("Pseudorange residual [m]")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(out_dir=None) -> dict:
    seed = 20260611
    rng = np.random.default_rng(seed)

    n_epochs = 80
    n_satellites = 12
    rx_height = 2.0
    wall_material = "concrete"
    ground_material = "wet_ground"

    out_path = Path(tempfile.mkdtemp(prefix="nlos_validation_")) if out_dir is None else Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, n_buildings = _make_scene()
    ground_plane = horizontal_ground_plane(0.0)
    receivers = _trajectory(n_epochs, rx_height)
    prns = [f"G{i + 1:02d}" for i in range(n_satellites)]

    records = []
    for epoch, rx in enumerate(receivers):
        sat_positions = _satellite_positions(epoch, n_epochs)
        paths_by_sat = model.compute_reflection_paths(
            rx,
            sat_positions,
            max_paths=4,
            ground_plane=ground_plane,
        )

        residuals, elevations, azimuths, is_los, visible, cn0 = _simulate_epoch(
            model,
            rx,
            sat_positions,
            paths_by_sat,
            rng,
            wall_material,
            ground_material,
        )

        epoch_records = records_from_epoch(
            epoch,
            prns,
            residuals,
            elevations,
            azimuths,
            is_los,
            visible,
            cn0_dbhz=cn0,
        )
        records.extend([rec for rec in epoch_records if getattr(rec, "visible", True)])

    los_bins = bin_by_los(records)
    los_samples = los_bins.get("los", [])
    nlos_samples = los_bins.get("nlos", [])

    overall_stats = _summary(records)
    los_stats = _summary(los_samples)
    nlos_stats = _summary(nlos_samples)

    sim_residuals = _finite_residuals(records)
    nlos_fraction = len(nlos_samples) / max(len(records), 1)

    reference_residuals = _make_reference_distribution(rng, sim_residuals.size, nlos_fraction)
    compare = compare_distributions(sim_residuals, reference_residuals)

    csv_path = out_path / "nlos_validation_residuals.csv"
    plot_path = out_path / "nlos_validation_cdf.png"
    write_csv(records, csv_path)
    _plot_cdf(sim_residuals, reference_residuals, plot_path)

    elev_bins = bin_by_elevation(records, [5, 15, 30, 50, 90])

    print("CPU-only NLOS validation demo")
    print(f"epochs={n_epochs}, satellites={n_satellites}, buildings={n_buildings}")
    print(f"samples={len(records)}, nlos_fraction={nlos_fraction:.3f}")
    print(_format_summary("overall", overall_stats))
    print(_format_summary("LOS", los_stats))
    print(_format_summary("NLOS", nlos_stats))
    print("Elevation bins:")
    for label, samples in elev_bins.items():
        print(_format_summary(str(label), _summary(samples)))
    print(
        "Comparison to placeholder reference: "
        f"W1={compare['wasserstein']:.2f}, KS={compare['ks']:.3f}, "
        f"p50_delta={compare['p50_delta']:.2f}"
    )
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")

    return {
        "n_epochs": n_epochs,
        "n_satellites": n_satellites,
        "n_buildings": n_buildings,
        "nlos_fraction": float(nlos_fraction),
        "los_p50_m": float(los_stats["p50"]),
        "nlos_p50_m": float(nlos_stats["p50"]),
        "nlos_bias_mean_m": float(nlos_stats["mean"]),
        "overall_rms_m": float(overall_stats["rms"]),
        "compare": compare,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "n_samples": int(len(records)),
    }


if __name__ == "__main__":
    main()
