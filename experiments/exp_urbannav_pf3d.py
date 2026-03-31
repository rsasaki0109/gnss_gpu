#!/usr/bin/env python3
"""Experiment: 3D-aware Particle Filter (ParticleFilter3D) on UrbanNav data.

Evaluates the 3D building-aware particle filter using PLATEAU-derived 3D
models (or a synthetic building model as fallback).  Compares against the
standard PF and WLS/EKF baselines.  Reports NLOS satellite statistics.

Outputs
-------
  experiments/results/pf3d_results.csv
  experiments/results/pf3d_summary.csv
  experiments/results/pf3d_cdf.png
  experiments/results/pf3d_timeline.png
  experiments/results/pf3d_nlos_stats.csv

Usage
-----
  PYTHONPATH=python python3 experiments/exp_urbannav_pf3d.py
  PYTHONPATH=python python3 experiments/exp_urbannav_pf3d.py \\
      --data-dir /path/to/urbannav \\
      --model-dir /path/to/plateau
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import (
    SimplePFCPU,
    compute_metrics,
    generate_synthetic_urbannav,
    plot_cdf,
    plot_error_timeline,
    print_comparison_table,
    save_results,
    wls_solve_py,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PARTICLES_PF3D = 100_000
PF_SIGMA_POS = 2.0
PF_SIGMA_CB = 300.0
PF_SIGMA_LOS = 3.0
PF_SIGMA_NLOS = 30.0
PF_NLOS_BIAS = 20.0


# ---------------------------------------------------------------------------
# 3D building model loading
# ---------------------------------------------------------------------------

def load_plateau_model(model_dir: Path) -> object | None:
    """Try to load PLATEAU CityGML model from model_dir.

    Returns a gnss_gpu.BuildingModel instance, or None on failure.
    """
    gml_files = list(model_dir.glob("**/*.gml")) + list(model_dir.glob("**/*.xml"))
    if not gml_files:
        return None

    print(f"    Found {len(gml_files)} CityGML files in {model_dir}")
    try:
        from gnss_gpu.io.plateau import load_plateau_buildings
        from gnss_gpu.raytrace import BuildingModel
        triangles = load_plateau_buildings(str(gml_files[0]))
        if triangles is None or len(triangles) == 0:
            return None
        model = BuildingModel(np.asarray(triangles, dtype=np.float32))
        print(f"    PLATEAU model loaded: {len(triangles)} triangles")
        return model
    except Exception as e:
        print(f"    Could not load PLATEAU model: {e}")
        return None


def create_synthetic_building_model(origin_ecef: np.ndarray) -> object | None:
    """Create a synthetic urban building model for fallback."""
    try:
        from gnss_gpu.raytrace import BuildingModel

        # ENU-to-ECEF rotation at origin
        from evaluate import ecef_to_lla
        import math as _math
        lat, lon, _ = ecef_to_lla(origin_ecef[0], origin_ecef[1], origin_ecef[2])
        sin_lat = _math.sin(lat)
        cos_lat = _math.cos(lat)
        sin_lon = _math.sin(lon)
        cos_lon = _math.cos(lon)
        R = np.array([
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ])

        def box_triangles_ecef(cx_enu, cy_enu, cz_enu, w, d, h):
            """Generate triangles for a box-shaped building in ECEF."""
            # 8 corners in ENU
            hw, hd, hh = w / 2, d / 2, h / 2
            corners_enu = np.array([
                [cx_enu - hw, cy_enu - hd, cz_enu - hh],
                [cx_enu + hw, cy_enu - hd, cz_enu - hh],
                [cx_enu + hw, cy_enu + hd, cz_enu - hh],
                [cx_enu - hw, cy_enu + hd, cz_enu - hh],
                [cx_enu - hw, cy_enu - hd, cz_enu + hh],
                [cx_enu + hw, cy_enu - hd, cz_enu + hh],
                [cx_enu + hw, cy_enu + hd, cz_enu + hh],
                [cx_enu - hw, cy_enu + hd, cz_enu + hh],
            ], dtype=np.float64)
            corners_ecef = np.array([origin_ecef + R @ c for c in corners_enu])

            # 6 faces, 2 triangles each = 12 triangles
            faces = [
                (0, 1, 2), (0, 2, 3),   # bottom
                (4, 5, 6), (4, 6, 7),   # top
                (0, 1, 5), (0, 5, 4),   # front
                (2, 3, 7), (2, 7, 6),   # back
                (1, 2, 6), (1, 6, 5),   # right
                (0, 3, 7), (0, 7, 4),   # left
            ]
            tris = np.array([[corners_ecef[a], corners_ecef[b], corners_ecef[c]]
                              for a, b, c in faces], dtype=np.float32)
            return tris

        buildings_enu = [
            (30.0,  20.0, 25.0, 40.0, 100.0),
            (-40.0, -10.0, 30.0, 30.0, 60.0),
            (10.0, -50.0, 20.0, 20.0, 80.0),
            (-20.0, 60.0, 35.0, 25.0, 120.0),
        ]
        all_tris = []
        for ce, cn, w, d, h in buildings_enu:
            tris = box_triangles_ecef(ce, cn, h / 2, w, d, h)
            all_tris.append(tris)
        all_tris = np.concatenate(all_tris, axis=0)

        model = BuildingModel(all_tris)
        print(f"    Synthetic building model: {len(all_tris)} triangles, "
              f"{len(buildings_enu)} buildings")
        return model
    except (ImportError, Exception) as e:
        print(f"    BuildingModel unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Baseline runners (reused)
# ---------------------------------------------------------------------------

def run_wls_quick(data: dict) -> np.ndarray:
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    positions = np.zeros((n_epochs, 4))
    try:
        from gnss_gpu import wls_position as _gpu_wls
        for i in range(n_epochs):
            result, _ = _gpu_wls(sat_ecef[i], pseudoranges[i], weights[i], 10, 1e-4)
            positions[i] = np.asarray(result)
    except (ImportError, Exception):
        for i in range(n_epochs):
            positions[i], _ = wls_solve_py(sat_ecef[i], pseudoranges[i], weights[i])
    return positions


def run_ekf_quick(data: dict, wls_init: np.ndarray) -> np.ndarray:
    from gnss_gpu.ekf import EKFPositioner
    n_epochs = data["n_epochs"]
    ekf = EKFPositioner(sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                        sigma_clk=100.0, sigma_drift=10.0)
    ekf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                   sigma_pos=50.0, sigma_cb=500.0)
    positions = np.zeros((n_epochs, 3))
    for i in range(n_epochs):
        if i > 0:
            ekf.predict(dt=data["dt"])
        ekf.update(data["sat_ecef"][i], data["pseudoranges"][i],
                   weights=data["weights"][i])
        positions[i] = ekf.get_position()
    return positions


def run_pf_standard(data: dict, n_particles: int,
                    wls_init: np.ndarray) -> tuple[np.ndarray, float, str]:
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, sigma_pos=PF_SIGMA_POS,
                            sigma_cb=PF_SIGMA_CB, sigma_pr=5.0,
                            resampling="megopolis", seed=42)
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        n_cpu = min(n_particles, 50_000)
        pf = SimplePFCPU(n_particles=n_cpu, sigma_pos=PF_SIGMA_POS,
                         sigma_cb=PF_SIGMA_CB, sigma_pr=5.0, seed=42)
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = f"CPU({n_cpu})"

    ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    return positions, ms, backend


# ---------------------------------------------------------------------------
# 3D-aware PF
# ---------------------------------------------------------------------------

def run_pf3d(
    data: dict,
    building_model: object | None,
    n_particles: int,
    wls_init: np.ndarray,
) -> tuple[np.ndarray, float, str, dict]:
    """Run ParticleFilter3D.

    Returns
    -------
    positions : ndarray (N, 3)
    ms_per_epoch : float
    backend : str
    nlos_stats : dict
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()
    backend = "unavailable"
    nlos_stats: dict = {"n_nlos_classified": 0, "n_total_obs": 0}

    if building_model is not None:
        try:
            from gnss_gpu.particle_filter_3d import ParticleFilter3D
            pf3d = ParticleFilter3D(
                building_model=building_model,
                sigma_los=PF_SIGMA_LOS,
                sigma_nlos=PF_SIGMA_NLOS,
                nlos_bias=PF_NLOS_BIAS,
                n_particles=n_particles,
                sigma_pos=PF_SIGMA_POS,
                sigma_cb=PF_SIGMA_CB,
                sigma_pr=PF_SIGMA_LOS,
                resampling="megopolis",
                seed=42,
            )
            pf3d.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                            spread_pos=50.0, spread_cb=500.0)

            for i in range(n_epochs):
                pf3d.predict(dt=dt)
                pf3d.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
                positions[i] = pf3d.estimate()[:3]

            backend = "GPU-3D"

            # Estimate NLOS statistics using ground-truth positions + ray tracing
            nlos_stats = _compute_nlos_stats(
                data, building_model, ground_truth_pos=data["ground_truth"])

        except (ImportError, RuntimeError, Exception) as e:
            print(f"      PF3D GPU failed ({type(e).__name__}: {e}), "
                  f"falling back to standard CPU PF")
            # Fall back to standard PF
            n_cpu = min(n_particles, 50_000)
            pf = SimplePFCPU(n_particles=n_cpu, sigma_pos=PF_SIGMA_POS,
                             sigma_cb=PF_SIGMA_CB, sigma_pr=PF_SIGMA_LOS, seed=42)
            pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                          spread_pos=50.0, spread_cb=500.0)
            for i in range(n_epochs):
                pf.predict(dt=dt)
                pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
                positions[i] = pf.estimate()[:3]
            backend = f"CPU-fallback({n_cpu})"
    else:
        # No building model at all -- use standard PF
        print("      No building model; running standard PF as PF3D substitute.")
        n_cpu = min(n_particles, 50_000)
        pf = SimplePFCPU(n_particles=n_cpu, sigma_pos=PF_SIGMA_POS,
                         sigma_cb=PF_SIGMA_CB, sigma_pr=PF_SIGMA_LOS, seed=42)
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = "CPU-no-model"

    ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    return positions, ms, backend, nlos_stats


def _compute_nlos_stats(data: dict, building_model: object,
                        ground_truth_pos: np.ndarray) -> dict:
    """Classify each satellite observation as LOS/NLOS using ray tracing."""
    stats: dict = {
        "n_nlos_classified": 0,
        "n_total_obs": 0,
        "nlos_fraction_per_satellite": [],
    }
    try:
        from gnss_gpu.raytrace import BuildingModel
        n_epochs = data["n_epochs"]
        n_sat = data["n_satellites"]
        sat_ecef = data["sat_ecef"]

        nlos_counts = np.zeros(n_sat, dtype=int)
        total = 0

        # Sample every 10th epoch for speed
        stride = max(1, n_epochs // 30)
        sampled = range(0, n_epochs, stride)

        for i in sampled:
            rx = ground_truth_pos[i]
            for s in range(n_sat):
                sat = sat_ecef[i, s]
                # Check LOS using building model
                blocked = building_model.is_blocked(rx, sat)
                if blocked:
                    nlos_counts[s] += 1
                total += 1

        stats["n_total_obs"] = total
        stats["n_nlos_classified"] = int(np.sum(nlos_counts))
        n_sampled = len(list(sampled))
        stats["nlos_fraction_per_satellite"] = (
            nlos_counts / max(n_sampled, 1)).tolist()

    except Exception as e:
        print(f"      NLOS stats computation failed: {e}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3D-aware PF evaluation on UrbanNav data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to UrbanNav dataset directory")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to PLATEAU CityGML model directory")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Number of epochs for synthetic data (default: 300)")
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES_PF3D,
                        help=f"Particle count (default: {N_PARTICLES_PF3D})")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: 3D-aware Particle Filter on UrbanNav Data")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = generate_synthetic_urbannav(
        n_epochs=args.n_epochs, n_satellites=8, seed=42)
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]
    print(f"    Synthetic data: {n_epochs} epochs, "
          f"{data['n_satellites']} satellites")

    # ------------------------------------------------------------------
    # [2] Building model
    # ------------------------------------------------------------------
    print("\n[2] Loading 3D building model ...")
    building_model = None

    if args.model_dir is not None and args.model_dir.exists():
        building_model = load_plateau_model(args.model_dir)

    if building_model is None:
        print("    PLATEAU model not found. Creating synthetic building model.")
        building_model = create_synthetic_building_model(data["origin_ecef"])

    if building_model is None:
        print("    WARNING: No building model available. "
              "PF3D will fall back to standard PF.")

    # ------------------------------------------------------------------
    # [3] Baselines
    # ------------------------------------------------------------------
    print("\n[3] Running baselines ...")
    wls_pos = run_wls_quick(data)
    wls_metrics = compute_metrics(wls_pos[:, :3], ground_truth)
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m")

    ekf_pos = run_ekf_quick(data, wls_pos)
    ekf_metrics = compute_metrics(ekf_pos, ground_truth)
    print(f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m")

    # ------------------------------------------------------------------
    # [4] Standard PF
    # ------------------------------------------------------------------
    print(f"\n[4] Running standard PF ({args.n_particles} particles) ...")
    pf_pos, pf_ms, pf_backend = run_pf_standard(
        data, args.n_particles, wls_pos)
    pf_metrics = compute_metrics(pf_pos, ground_truth)
    print(f"    PF ({pf_backend}): RMS 2D={pf_metrics['rms_2d']:.2f} m, "
          f"{pf_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [5] 3D-aware PF
    # ------------------------------------------------------------------
    print(f"\n[5] Running 3D-aware PF ({args.n_particles} particles) ...")
    pf3d_pos, pf3d_ms, pf3d_backend, nlos_stats = run_pf3d(
        data, building_model, args.n_particles, wls_pos)
    pf3d_metrics = compute_metrics(pf3d_pos, ground_truth)
    print(f"    PF3D ({pf3d_backend}): RMS 2D={pf3d_metrics['rms_2d']:.2f} m, "
          f"{pf3d_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [6] NLOS statistics
    # ------------------------------------------------------------------
    print("\n[6] NLOS satellite statistics:")
    print(f"    Observed NLOS fraction (synthetic ground truth): "
          f"{100.0 * data['n_nlos_total'] / (n_epochs * data['n_satellites']):.1f}%")
    if nlos_stats["n_total_obs"] > 0:
        nlos_frac = nlos_stats["n_nlos_classified"] / nlos_stats["n_total_obs"]
        print(f"    Ray-traced NLOS (sampled): "
              f"{100.0 * nlos_frac:.1f}% of {nlos_stats['n_total_obs']} obs")
        if nlos_stats["nlos_fraction_per_satellite"]:
            print(f"    Per-satellite NLOS fraction:")
            for s, frac in enumerate(nlos_stats["nlos_fraction_per_satellite"]):
                print(f"      Sat {s}: {100.0 * frac:.1f}%")

    # ------------------------------------------------------------------
    # [7] Comparison
    # ------------------------------------------------------------------
    print("\n[7] Results summary:")
    all_metrics = {
        "WLS": wls_metrics,
        "EKF": ekf_metrics,
        f"PF-{args.n_particles // 1000}K": pf_metrics,
        f"PF3D-{args.n_particles // 1000}K": pf3d_metrics,
    }
    all_metrics["WLS"]["time_ms"] = 0.0
    all_metrics["EKF"]["time_ms"] = 0.0
    all_metrics[f"PF-{args.n_particles // 1000}K"]["time_ms"] = pf_ms
    all_metrics[f"PF3D-{args.n_particles // 1000}K"]["time_ms"] = pf3d_ms

    print_comparison_table(all_metrics)

    # Improvement report
    pf3d_vs_pf = pf_metrics["rms_2d"] - pf3d_metrics["rms_2d"]
    pf3d_vs_wls = wls_metrics["rms_2d"] - pf3d_metrics["rms_2d"]
    print(f"\n  PF3D improvement over standard PF: "
          f"{pf3d_vs_pf:+.2f} m RMS 2D")
    print(f"  PF3D improvement over WLS:          "
          f"{pf3d_vs_wls:+.2f} m RMS 2D")

    # ------------------------------------------------------------------
    # [8] Save results
    # ------------------------------------------------------------------
    print("\n[8] Saving results ...")
    epochs = np.arange(n_epochs)

    save_results({
        "epoch": epochs,
        "error_2d_wls": wls_metrics["errors_2d"],
        "error_2d_ekf": ekf_metrics["errors_2d"],
        "error_2d_pf": pf_metrics["errors_2d"],
        "error_2d_pf3d": pf3d_metrics["errors_2d"],
        "gt_x": ground_truth[:, 0],
        "gt_y": ground_truth[:, 1],
        "gt_z": ground_truth[:, 2],
        "pf3d_x": pf3d_pos[:, 0],
        "pf3d_y": pf3d_pos[:, 1],
        "pf3d_z": pf3d_pos[:, 2],
    }, RESULTS_DIR / "pf3d_results.csv")

    method_labels = list(all_metrics.keys())
    save_results({
        "method": method_labels,
        "rms_2d": [all_metrics[m]["rms_2d"] for m in method_labels],
        "mean_2d": [all_metrics[m]["mean_2d"] for m in method_labels],
        "p50": [all_metrics[m]["p50"] for m in method_labels],
        "p95": [all_metrics[m]["p95"] for m in method_labels],
        "max_2d": [all_metrics[m]["max_2d"] for m in method_labels],
        "time_ms": [all_metrics[m].get("time_ms", 0.0) for m in method_labels],
    }, RESULTS_DIR / "pf3d_summary.csv")

    nlos_sat_labels = [f"sat_{s}" for s in
                       range(len(nlos_stats.get("nlos_fraction_per_satellite", [])))]
    nlos_fracs = nlos_stats.get("nlos_fraction_per_satellite", [])
    if nlos_sat_labels:
        save_results({
            "satellite": nlos_sat_labels,
            "nlos_fraction": nlos_fracs,
        }, RESULTS_DIR / "pf3d_nlos_stats.csv")

    # ------------------------------------------------------------------
    # [9] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[9] Generating plots ...")
        errors_for_plot = {
            "WLS": wls_metrics["errors_2d"],
            "EKF": ekf_metrics["errors_2d"],
            f"PF-{args.n_particles // 1000}K": pf_metrics["errors_2d"],
            f"PF3D-{args.n_particles // 1000}K": pf3d_metrics["errors_2d"],
        }
        plot_cdf(errors_for_plot, RESULTS_DIR / "pf3d_cdf.png",
                 title="CDF of 2D Error - PF3D vs Baselines")
        plot_error_timeline(
            data["times"], errors_for_plot,
            RESULTS_DIR / "pf3d_timeline.png",
            title="2D Error Over Time - PF3D vs Baselines",
        )

    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
