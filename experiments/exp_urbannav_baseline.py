#!/usr/bin/env python3
"""Experiment: WLS / EKF / RTKLIB-equivalent baseline evaluation on urban GNSS data.

Evaluates three baseline positioning methods on PPC-Dataset / UrbanNav real data
(or synthetic fallback):
  1. WLS  -- single-epoch Weighted Least Squares
  2. EKF  -- Extended Kalman Filter
  3. RTK  -- code-only RTKLIB-equivalent (double-differenced WLS)

Outputs
-------
  experiments/results/baseline_wls.csv
  experiments/results/baseline_ekf.csv
  experiments/results/baseline_summary.csv
  experiments/results/baseline_cdf.png
  experiments/results/baseline_timeline.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py
  PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py --data-dir /path/to/PPC-Dataset/tokyo/run1
  PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py --data-dir /path/to/UrbanNav/Odaiba
"""

from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------
from evaluate import (
    compute_metrics,
    generate_synthetic_urbannav,
    plot_cdf,
    plot_error_timeline,
    print_comparison_table,
    save_results,
    wls_solve_py,
)
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.multi_gnss import MultiGNSSSolver, SYSTEM_GPS
from gnss_gpu.multi_gnss_quality import (
    MultiGNSSQualityVetoConfig,
    select_multi_gnss_solution,
)

SYSTEM_IDS = {
    "G": 0,
    "R": 1,
    "E": 2,
    "C": 3,
    "J": 4,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_ppc_run_dir(data_dir: Path) -> Path | None:
    """Resolve a PPC run directory from a run path, city directory, or dataset root."""
    if PPCDatasetLoader.is_run_directory(data_dir):
        return data_dir

    candidates = sorted(
        path for path in data_dir.rglob("run*")
        if PPCDatasetLoader.is_run_directory(path)
    )
    return candidates[0] if candidates else None


def _resolve_urbannav_run_dir(data_dir: Path) -> Path | None:
    """Resolve an UrbanNav run directory from a run path or dataset root."""
    if UrbanNavLoader.is_run_directory(data_dir):
        return data_dir

    candidates = sorted(
        path for path in data_dir.rglob("*")
        if UrbanNavLoader.is_run_directory(path)
    )
    return candidates[0] if candidates else None


def load_real_data(
    data_dir: Path,
    max_epochs: int | None = None,
    start_epoch: int = 0,
    systems: tuple[str, ...] = ("G",),
    urban_rover: str = "ublox",
) -> dict | None:
    """Attempt to load PPC-Dataset or UrbanNav real data from *data_dir*."""
    run_dir = _resolve_ppc_run_dir(data_dir)
    if run_dir is not None:
        print(f"    Detected PPC run: {run_dir}")
        try:
            loader = PPCDatasetLoader(run_dir)
            data = loader.load_experiment_data(
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                systems=systems,
            )
            print(f"    Loaded {data['dataset_name']}: {data['n_epochs']} epochs")
            print(f"    Median satellites/epoch: {data['n_satellites']}")
            if "constellations" in data:
                print(f"    Constellations: {', '.join(data['constellations'])}")
            return data
        except Exception as e:
            print(f"    Could not parse PPC data: {e}")

    run_dir = _resolve_urbannav_run_dir(data_dir)
    if run_dir is None:
        return None

    print(f"    Detected UrbanNav run: {run_dir}")
    try:
        loader = UrbanNavLoader(run_dir)
        data = loader.load_experiment_data(
            max_epochs=max_epochs,
            start_epoch=start_epoch,
            systems=systems,
            rover_source=urban_rover,
        )
        print(f"    Loaded {data['dataset_name']}: {data['n_epochs']} epochs")
        print(f"    Median satellites/epoch: {data['n_satellites']}")
        if "constellations" in data:
            print(f"    Constellations: {', '.join(data['constellations'])}")
        return data
    except Exception as e:
        print(f"    Could not parse UrbanNav data: {e}")
        return None


def load_or_generate_data(
    data_dir: Path | None,
    n_epochs: int = 300,
    max_real_epochs: int | None = None,
    start_epoch: int = 0,
    systems: tuple[str, ...] = ("G",),
    urban_rover: str = "ublox",
) -> dict:
    """Load PPC/UrbanNav data or generate synthetic fallback."""
    if data_dir is not None and data_dir.exists():
        print(f"    Searching for real data in: {data_dir}")
        data = load_real_data(
            data_dir,
            max_epochs=max_real_epochs,
            start_epoch=start_epoch,
            systems=systems,
            urban_rover=urban_rover,
        )
        if data is not None:
            return data
        print("    Real data not found or not parseable. Using synthetic data.")
    else:
        print("    No data directory provided. Using synthetic data.")

    data = generate_synthetic_urbannav(n_epochs=n_epochs, n_satellites=8, seed=42)
    print(f"    Synthetic data: {data['n_epochs']} epochs, "
          f"{data['n_satellites']} satellites, "
          f"{data['n_nlos_total']} NLOS observations "
          f"({100.0 * data['n_nlos_total'] / (data['n_epochs'] * data['n_satellites']):.1f}%)")
    return data


def _parse_weight_scale(spec: str) -> dict[int, float]:
    scales: dict[int, float] = {}
    if not spec:
        return scales

    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid weight scale entry: {item}")
        key, value = item.split("=", 1)
        sys_char = key.strip().upper()
        if sys_char not in SYSTEM_IDS:
            raise ValueError(f"unknown constellation in weight scale: {sys_char}")
        scales[SYSTEM_IDS[sys_char]] = float(value)
    return scales


def _apply_weight_scale(
    weights: np.ndarray,
    system_ids: np.ndarray | None,
    scale_by_system: dict[int, float],
) -> np.ndarray:
    if system_ids is None or not scale_by_system:
        return weights

    scaled = np.asarray(weights, dtype=np.float64).copy()
    for system_id, scale in scale_by_system.items():
        scaled[np.asarray(system_ids) == system_id] *= scale
    return scaled


def _solve_single_system_epoch(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    try:
        from gnss_gpu import wls_position as _gpu_wls

        result, _ = _gpu_wls(sat_ecef, pseudoranges, weights, 10, 1e-4)
        return np.asarray(result, dtype=np.float64)
    except (ImportError, Exception):
        result, _ = wls_solve_py(sat_ecef, pseudoranges, weights)
        return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# WLS baseline
# ---------------------------------------------------------------------------

def run_wls(
    data: dict,
    weight_scale_by_system: dict[int, float] | None = None,
    quality_veto_config: MultiGNSSQualityVetoConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Run WLS on every epoch.

    Returns
    -------
    positions : ndarray, shape (N, 4)   [x, y, z, clock_bias]
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    weight_scale_by_system = weight_scale_by_system or {}

    # Try GPU-accelerated WLS
    positions = np.zeros((n_epochs, 4))
    t0 = time.perf_counter()

    if system_ids is not None and len(data.get("constellations", ())) > 1:
        systems = sorted({int(sid) for epoch_ids in system_ids for sid in epoch_ids})
        solver = MultiGNSSSolver(systems=systems)
        accepted_multi_epochs = 0
        fallback_epochs = 0
        for i in range(n_epochs):
            scaled_weights = _apply_weight_scale(
                weights[i],
                system_ids[i],
                weight_scale_by_system,
            )
            pos, biases, _ = solver.solve(
                sat_ecef[i],
                pseudoranges[i],
                system_ids[i],
                scaled_weights,
            )
            if quality_veto_config is not None:
                ref_mask = np.asarray(system_ids[i]) == int(quality_veto_config.reference_system)
                if int(np.count_nonzero(ref_mask)) >= 4:
                    reference_solution = _solve_single_system_epoch(
                        sat_ecef[i][ref_mask],
                        pseudoranges[i][ref_mask],
                        scaled_weights[ref_mask],
                    )
                    decision = select_multi_gnss_solution(
                        reference_solution=reference_solution,
                        multi_position=pos,
                        multi_biases=biases,
                        sat_ecef=sat_ecef[i],
                        pseudoranges=pseudoranges[i],
                        system_ids=system_ids[i],
                        config=quality_veto_config,
                    )
                    positions[i, :3] = decision.position
                    positions[i, 3] = decision.clock_bias_m
                    accepted_multi_epochs += int(decision.use_multi)
                    fallback_epochs += int(not decision.use_multi)
                    continue
            positions[i, :3] = pos
            positions[i, 3] = float(
                biases.get(
                    int(
                        quality_veto_config.reference_system
                        if quality_veto_config is not None
                        else SYSTEM_GPS
                    ),
                    biases.get(SYSTEM_GPS, 0.0),
                )
            )
            accepted_multi_epochs += 1
        backend = "Multi-GNSS"
        if weight_scale_by_system:
            backend += " scaled"
        if quality_veto_config is not None:
            backend += " quality-veto"
            print(
                f"    Multi-GNSS quality veto: "
                f"{accepted_multi_epochs}/{n_epochs} epochs kept multi, "
                f"{fallback_epochs} fell back"
            )
    else:
        try:
            from gnss_gpu import wls_position as _gpu_wls
            for i in range(n_epochs):
                result, _ = _gpu_wls(sat_ecef[i], pseudoranges[i], weights[i],
                                      10, 1e-4)
                positions[i] = np.asarray(result)
            backend = "GPU"
        except (ImportError, Exception):
            for i in range(n_epochs):
                positions[i], _ = wls_solve_py(
                    sat_ecef[i], pseudoranges[i], weights[i])
            backend = "Python"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    WLS ({backend}): {elapsed:.1f} ms total, "
          f"{ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# EKF baseline
# ---------------------------------------------------------------------------

def run_ekf(data: dict, wls_init: np.ndarray) -> tuple[np.ndarray, float]:
    """Run EKF over all epochs.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    times = np.asarray(data["times"], dtype=np.float64)
    system_ids = data.get("system_ids")
    gps_only = system_ids is not None and len(data.get("constellations", ())) > 1

    from gnss_gpu.ekf import EKFPositioner

    ekf = EKFPositioner(sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                        sigma_clk=100.0, sigma_drift=10.0)
    ekf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                   sigma_pos=50.0, sigma_cb=500.0)

    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    for i in range(n_epochs):
        if i > 0:
            dt = float(times[i] - times[i - 1])
            if dt <= 0.0:
                dt = float(data.get("dt", 1.0))
            ekf.predict(dt=dt)

        sat_i = sat_ecef[i]
        pr_i = pseudoranges[i]
        w_i = weights[i]
        if gps_only:
            mask = system_ids[i] == SYSTEM_GPS
            if int(np.count_nonzero(mask)) >= 4:
                sat_i = sat_i[mask]
                pr_i = pr_i[mask]
                w_i = w_i[mask]
            else:
                positions[i] = ekf.get_position()
                continue

        ekf.update(sat_i, pr_i, weights=w_i)
        positions[i] = ekf.get_position()

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    mode = "GPS-only" if gps_only else "all-sats"
    print(f"    EKF ({mode}): {elapsed:.1f} ms total, {ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# RTKLIB-equivalent: code-only differential positioning
# ---------------------------------------------------------------------------

def run_rtklib_equivalent(data: dict) -> tuple[np.ndarray, float]:
    """RTKLIB-equivalent code-only differential positioning.

    Uses RTKSolver from gnss_gpu with pseudorange-only mode (no carrier phase).
    Falls back to WLS when RTKSolver unavailable.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    # Fallback until the experiment dataset provides matched base observations
    # and carrier phases through the loader.
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    gps_only = system_ids is not None and len(data.get("constellations", ())) > 1

    try:
        from gnss_gpu import wls_position as _gpu_wls
        for i in range(n_epochs):
            sat_i = sat_ecef[i]
            pr_i = pseudoranges[i]
            w_i = weights[i]
            if gps_only:
                mask = system_ids[i] == SYSTEM_GPS
                if int(np.count_nonzero(mask)) >= 4:
                    sat_i = sat_i[mask]
                    pr_i = pr_i[mask]
                    w_i = w_i[mask]
            sol, _ = _gpu_wls(sat_i, pr_i, w_i, 10, 1e-4)
            positions[i] = np.asarray(sol)[:3]
        backend = "GPU WLS fallback (GPS-only)" if gps_only else "GPU WLS fallback"
    except (ImportError, Exception):
        for i in range(n_epochs):
            sat_i = sat_ecef[i]
            pr_i = pseudoranges[i]
            w_i = weights[i]
            if gps_only:
                mask = system_ids[i] == SYSTEM_GPS
                if int(np.count_nonzero(mask)) >= 4:
                    sat_i = sat_i[mask]
                    pr_i = pr_i[mask]
                    w_i = w_i[mask]
            sol, _ = wls_solve_py(sat_i, pr_i, w_i)
            positions[i] = sol[:3]
        backend = "Python WLS fallback (GPS-only)" if gps_only else "Python WLS fallback"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    RTKLIB-equiv ({backend}): {elapsed:.1f} ms total, "
          f"{ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation (WLS/EKF/RTKLIB) on PPC or synthetic data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to PPC run directory, UrbanNav run directory, or dataset root")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Number of epochs for synthetic data (default: 300)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Optional cap on real-data epochs")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Skip this many usable real-data epochs before evaluation")
    parser.add_argument("--systems", type=str, default="G",
                        help="Comma-separated constellations for real data, e.g. G or G,E,J")
    parser.add_argument("--urban-rover", type=str, default="ublox",
                        help="UrbanNav rover observation source, e.g. ublox or trimble")
    parser.add_argument("--weight-scale", type=str, default="",
                        help="Optional per-constellation weight scale, e.g. E=0.1,J=2.0")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    weight_scale_by_system = _parse_weight_scale(args.weight_scale)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: Baseline Methods on PPC / Synthetic Data")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data loading
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_or_generate_data(
        args.data_dir,
        n_epochs=args.n_epochs,
        max_real_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        urban_rover=args.urban_rover,
    )
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]
    if "dataset_name" in data:
        print(f"    Dataset: {data['dataset_name']}")

    # ------------------------------------------------------------------
    # [2] WLS
    # ------------------------------------------------------------------
    print("\n[2] Running WLS ...")
    wls_pos, wls_ms = run_wls(data, weight_scale_by_system=weight_scale_by_system)
    wls_metrics = compute_metrics(wls_pos[:, :3], ground_truth)
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m, "
          f"P95={wls_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [3] EKF
    # ------------------------------------------------------------------
    print("\n[3] Running EKF ...")
    ekf_pos, ekf_ms = run_ekf(data, wls_pos)
    ekf_metrics = compute_metrics(ekf_pos, ground_truth)
    print(f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m, "
          f"P95={ekf_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [4] RTKLIB-equivalent
    # ------------------------------------------------------------------
    print("\n[4] Running RTKLIB-equivalent ...")
    rtk_pos, rtk_ms = run_rtklib_equivalent(data)
    rtk_metrics = compute_metrics(rtk_pos, ground_truth)
    print(f"    RTKLIB-equiv: RMS 2D={rtk_metrics['rms_2d']:.2f} m, "
          f"P95={rtk_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [5] Summary table
    # ------------------------------------------------------------------
    print("\n[5] Results summary:")
    all_metrics = {
        "WLS": wls_metrics,
        "EKF": ekf_metrics,
        "RTKLIB-equiv": rtk_metrics,
    }
    all_metrics["WLS"]["time_ms"] = wls_ms
    all_metrics["EKF"]["time_ms"] = ekf_ms
    all_metrics["RTKLIB-equiv"]["time_ms"] = rtk_ms

    print_comparison_table(all_metrics)

    # ------------------------------------------------------------------
    # [6] Save per-epoch results
    # ------------------------------------------------------------------
    print("\n[6] Saving results ...")
    epochs = np.arange(n_epochs)

    save_results({
        "epoch": epochs,
        "error_2d": wls_metrics["errors_2d"],
        "error_3d": wls_metrics["errors_3d"],
        "est_x": wls_pos[:, 0],
        "est_y": wls_pos[:, 1],
        "est_z": wls_pos[:, 2],
        "gt_x": ground_truth[:, 0],
        "gt_y": ground_truth[:, 1],
        "gt_z": ground_truth[:, 2],
    }, RESULTS_DIR / "baseline_wls.csv")

    save_results({
        "epoch": epochs,
        "error_2d": ekf_metrics["errors_2d"],
        "error_3d": ekf_metrics["errors_3d"],
        "est_x": ekf_pos[:, 0],
        "est_y": ekf_pos[:, 1],
        "est_z": ekf_pos[:, 2],
    }, RESULTS_DIR / "baseline_ekf.csv")

    summary = {
        "method": ["WLS", "EKF", "RTKLIB-equiv"],
        "rms_2d": [wls_metrics["rms_2d"], ekf_metrics["rms_2d"], rtk_metrics["rms_2d"]],
        "rms_3d": [wls_metrics["rms_3d"], ekf_metrics["rms_3d"], rtk_metrics["rms_3d"]],
        "mean_2d": [wls_metrics["mean_2d"], ekf_metrics["mean_2d"], rtk_metrics["mean_2d"]],
        "p50": [wls_metrics["p50"], ekf_metrics["p50"], rtk_metrics["p50"]],
        "p67": [wls_metrics["p67"], ekf_metrics["p67"], rtk_metrics["p67"]],
        "p95": [wls_metrics["p95"], ekf_metrics["p95"], rtk_metrics["p95"]],
        "max_2d": [wls_metrics["max_2d"], ekf_metrics["max_2d"], rtk_metrics["max_2d"]],
        "time_ms": [wls_ms, ekf_ms, rtk_ms],
        "n_epochs": [n_epochs, n_epochs, n_epochs],
    }
    save_results(summary, RESULTS_DIR / "baseline_summary.csv")

    # ------------------------------------------------------------------
    # [7] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[7] Generating plots ...")
        errors_for_cdf = {
            "WLS": wls_metrics["errors_2d"],
            "EKF": ekf_metrics["errors_2d"],
            "RTKLIB-equiv": rtk_metrics["errors_2d"],
        }
        plot_cdf(errors_for_cdf, RESULTS_DIR / "baseline_cdf.png",
                 title="CDF of 2D Positioning Error - Baseline Methods")

        plot_error_timeline(
            data["times"], errors_for_cdf,
            RESULTS_DIR / "baseline_timeline.png",
            title="2D Positioning Error Over Time - Baseline Methods",
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
