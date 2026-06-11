"""CPU-only demo: impact of knife-edge diffraction on NLOS residuals.

Builds the same urban scene as ``demo_nlos_validation`` and compares the
pseudorange-residual distribution **without** diffraction (reflection paths
only; deeply shadowed satellites are simply lost) against the distribution
**with** knife-edge diffraction (signals bend around building edges, so some
shadowed satellites become observable with a characteristic diffraction bias).

Pure Python + numpy + matplotlib (Agg). No GPU or CUDA extension required.

Run:  python examples/demo_nlos_diffraction.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_gpu.raytrace import horizontal_ground_plane
from gnss_gpu.diffraction import compute_diffraction_paths
from gnss_gpu.validation import (
    bin_by_los,
    compare_distributions,
    records_from_epoch,
    residual_array,
    summarize,
    write_csv,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_base_demo():
    """Load demo_nlos_validation as a module to reuse its scene/geometry."""
    demo_path = _REPO_ROOT / "examples" / "demo_nlos_validation.py"
    spec = importlib.util.spec_from_file_location("demo_nlos_validation", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_edges(triangles):
    """Extract diffraction edges from the scene mesh (experiments helper)."""
    exp_dir = str(_REPO_ROOT / "experiments")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    from utd_edge_features import extract_diffraction_edges

    return extract_diffraction_edges(
        triangles,
        include_boundary_edges=True,
        min_edge_length_m=1.0,
        min_dihedral_deg=20.0,
    )


def _simulate_epoch(
    base,
    model,
    rx,
    sat_positions,
    refl_paths,
    diff_paths,
    rng,
    wall_material,
    ground_material,
    *,
    use_diffraction,
    diffraction_acq_floor=0.05,
    elevation_mask_deg=5.0,
):
    """Per-epoch residuals; mirrors the base demo but adds diffraction rescue.

    Returns the same tuple shape as the base demo's ``_simulate_epoch`` plus a
    boolean array marking which satellites were rescued by diffraction.
    """
    azimuths, elevations = base._azel_from_positions(rx, sat_positions)
    n = sat_positions.shape[0]
    visible = elevations >= elevation_mask_deg
    is_los = np.zeros(n, dtype=bool)
    residuals = np.full(n, np.nan, dtype=float)
    cn0 = np.full(n, np.nan, dtype=float)
    rescued = np.zeros(n, dtype=bool)

    for i, sat in enumerate(sat_positions):
        if not visible[i]:
            continue

        is_los[i] = base._is_los(rx, sat, model.triangles)
        paths = list(refl_paths[i])

        if is_los[i]:
            num = 0.0
            den = 1.0
            for path in paths:
                delay = base._path_excess_delay(path)
                if 0.0 <= delay < base.CHIP_M:
                    amp = base._path_amplitude(path, wall_material, ground_material)
                    num += amp * delay
                    den += amp
            residuals[i] = base.BETA * num / den + rng.normal(0.0, 0.8)
            cn0[i] = np.clip(45.0 + rng.normal(0.0, 2.0), 20.0, 55.0)
            continue

        # NLOS: try a specular reflection first (same as the base demo).
        if paths:
            scored = [
                (base._path_amplitude(path, wall_material, ground_material),
                 base._path_excess_delay(path))
                for path in paths
            ]
            amp, delay = max(scored, key=lambda item: item[0])
            if amp > 0.0:
                residuals[i] = delay + rng.normal(0.0, 3.0)
                cn0[i] = np.clip(30.0 + rng.normal(0.0, 4.0), 12.0, 45.0)
                continue

        # No usable reflection. Without diffraction the satellite is lost.
        if not use_diffraction:
            visible[i] = False
            continue

        # Diffraction rescue: strongest knife-edge path above the acq. floor.
        dpaths = list(diff_paths[i]) if diff_paths is not None else []
        dpaths = [p for p in dpaths if p.amplitude >= diffraction_acq_floor]
        if not dpaths:
            visible[i] = False
            continue

        best = max(dpaths, key=lambda p: p.amplitude)
        # Diffracted excess delay maps to a positive residual; weaker (lower
        # amplitude => higher attenuation) paths carry more measurement noise.
        noise = 4.0 + 6.0 * (1.0 - best.amplitude)
        residuals[i] = float(best.excess_delay) + rng.normal(0.0, noise)
        cn0[i] = np.clip(
            42.0 - best.attenuation_db + rng.normal(0.0, 3.0), 8.0, 45.0)
        rescued[i] = True

    return (
        residuals,
        np.radians(elevations),
        np.radians(azimuths),
        is_los,
        visible,
        cn0,
        rescued,
    )


def _finite(samples):
    values = np.asarray(residual_array(samples), dtype=float)
    return values[np.isfinite(values)]


def _summary(samples):
    values = _finite(samples)
    if values.size == 0:
        return {k: math.nan for k in
                ("count", "mean", "rms", "mae", "p50", "p68", "p95", "p99",
                 "abs_p50", "abs_p95")}
    return summarize(values)


def _fmt(label, stats):
    return (
        f"{label:>14s}  n={int(stats['count']):4d}  "
        f"mean={stats['mean']:8.2f} m  rms={stats['rms']:8.2f} m  "
        f"p50={stats['p50']:8.2f} m  p95={stats['p95']:8.2f} m"
    )


def _plot_cdf(without, with_diff, path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)
    for values, label in [
        (without, "reflection only"),
        (with_diff, "reflection + diffraction"),
    ]:
        v = np.sort(np.asarray(values, dtype=float))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1, dtype=float) / v.size
        ax.plot(v, y, linewidth=2.0, label=f"{label} (n={v.size})")
    ax.set_xlabel("Pseudorange residual [m]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("NLOS residual distribution: effect of knife-edge diffraction")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(out_dir=None) -> dict:
    seed = 20260611
    base = _load_base_demo()

    n_epochs = 80
    n_satellites = 12
    rx_height = 2.0
    wall_material = "concrete"
    ground_material = "wet_ground"

    out_path = (Path(tempfile.mkdtemp(prefix="nlos_diffraction_"))
                if out_dir is None else Path(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    model, n_buildings = base._make_scene()
    ground_plane = horizontal_ground_plane(0.0)
    receivers = base._trajectory(n_epochs, rx_height)
    prns = [f"G{i + 1:02d}" for i in range(n_satellites)]
    edges = _extract_edges(model.triangles)

    records_off = []
    records_on = []
    n_rescued = 0

    # Use independent RNG streams so the two configurations share identical
    # geometry/noise draws for the common (non-rescued) satellites.
    rng_off = np.random.default_rng(seed)
    rng_on = np.random.default_rng(seed)

    for epoch, rx in enumerate(receivers):
        sat_positions = base._satellite_positions(epoch, n_epochs)
        refl_paths = model.compute_reflection_paths(
            rx, sat_positions, max_paths=4, ground_plane=ground_plane)
        diff_paths = compute_diffraction_paths(
            rx, sat_positions, edges, max_paths=4,
            max_ray_edge_distance_m=40.0, max_excess_path_m=150.0,
            max_edge_range_m=400.0)

        res_off = _simulate_epoch(
            base, model, rx, sat_positions, refl_paths, diff_paths,
            rng_off, wall_material, ground_material, use_diffraction=False)
        res_on = _simulate_epoch(
            base, model, rx, sat_positions, refl_paths, diff_paths,
            rng_on, wall_material, ground_material, use_diffraction=True)

        for res, sink in ((res_off, records_off), (res_on, records_on)):
            residuals, elevations, azimuths, is_los, visible, cn0 = res[:6]
            recs = records_from_epoch(
                epoch, prns, residuals, elevations, azimuths,
                is_los, visible, cn0_dbhz=cn0)
            sink.extend([r for r in recs if getattr(r, "visible", True)])

        n_rescued += int(np.count_nonzero(res_on[6]))

    off_res = _finite(records_off)
    on_res = _finite(records_on)

    off_nlos = bin_by_los(records_off).get("nlos", [])
    on_nlos = bin_by_los(records_on).get("nlos", [])

    overall_off = _summary(records_off)
    overall_on = _summary(records_on)
    nlos_off = _summary(off_nlos)
    nlos_on = _summary(on_nlos)

    compare = compare_distributions(on_res, off_res)

    csv_path = out_path / "nlos_diffraction_residuals.csv"
    plot_path = out_path / "nlos_diffraction_cdf.png"
    write_csv(records_on, csv_path)
    _plot_cdf(off_res, on_res, plot_path)

    print("CPU-only NLOS diffraction demo")
    print(f"epochs={n_epochs}, satellites={n_satellites}, buildings={n_buildings}, "
          f"edges={int(edges.size)}")
    print(f"diffraction-rescued satellite-epochs: {n_rescued}")
    print(_fmt("overall off", overall_off))
    print(_fmt("overall on", overall_on))
    print(_fmt("NLOS off", nlos_off))
    print(_fmt("NLOS on", nlos_on))
    print(
        "off vs on distribution: "
        f"W1={compare['wasserstein']:.2f}, KS={compare['ks']:.3f}, "
        f"p50_delta={compare['p50_delta']:.2f}")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")

    return {
        "n_epochs": n_epochs,
        "n_satellites": n_satellites,
        "n_buildings": n_buildings,
        "n_edges": int(edges.size),
        "n_rescued": int(n_rescued),
        "n_samples_off": int(len(records_off)),
        "n_samples_on": int(len(records_on)),
        "nlos_count_off": int(len(off_nlos)),
        "nlos_count_on": int(len(on_nlos)),
        "nlos_p50_off_m": float(nlos_off["p50"]),
        "nlos_p50_on_m": float(nlos_on["p50"]),
        "compare_on_vs_off": compare,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }


if __name__ == "__main__":
    main()
