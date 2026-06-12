"""Calibrate a parametric NLOS residual model against real UrbanNav residuals.

This is the capstone that connects every piece built for the NLOS effort:

    real UrbanNav RINEX -> purified pseudorange residuals (target distribution)
        + a physically-parameterised residual generator (the "simulator")
        + the distribution-distance calibration loop (grid + coordinate descent)
    => recovered urban NLOS statistics (NLOS fraction, bias, spread, LOS sigma)

The generator mixes a near-zero LOS cluster with a positive-only NLOS tail
(NLOS pseudoranges are always longer), then removes the per-realisation median
exactly as the real-data bridge removes the receiver clock. Fitting its
parameters to the real distribution recovers interpretable environment numbers.

Requires the compiled GPU ephemeris extension and a local UrbanNav run
directory (default: experiments/data/urbannav/Odaiba).

Run:  python examples/demo_urbannav_calibration.py [Odaiba|Shinjuku]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.validation import (
    compare_distributions,
    coordinate_descent,
    grid_search,
    residual_samples_from_experiment_data,
)
from gnss_gpu.validation.real_residuals import residual_array

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "experiments" / "data" / "urbannav"


def make_residual_generator(n_samples, seed=12345):
    """Return a deterministic params -> residual-sample function.

    Parameters (all metres / fraction):
        nlos_fraction : share of satellites that are NLOS (0..1)
        nlos_bias_m   : floor of the NLOS excess-range bias
        nlos_scale_m  : spread of the NLOS half-normal tail
        los_sigma_m   : LOS pseudorange noise
    """
    def gen(params):
        rng = np.random.default_rng(seed)
        f = float(np.clip(params["nlos_fraction"], 0.0, 1.0))
        bias = float(params["nlos_bias_m"])
        scale = max(float(params["nlos_scale_m"]), 1e-6)
        los_sigma = max(float(params["los_sigma_m"]), 1e-6)

        is_nlos = rng.random(n_samples) < f
        x = rng.normal(0.0, los_sigma, n_samples)
        n_nlos = int(np.count_nonzero(is_nlos))
        if n_nlos:
            x[is_nlos] = bias + np.abs(rng.normal(0.0, scale, n_nlos))
        # Receiver-clock removal mirrors the real-data bridge.
        return x - np.median(x)

    return gen


def _plot_fit(target, fitted, path, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)
    for values, label in [(target, "real (target)"), (fitted, "calibrated model")]:
        v = np.sort(np.asarray(values, dtype=float))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1, dtype=float) / v.size
        ax.plot(v, y, linewidth=2.0, label=f"{label} (n={v.size})")
    ax.set_xlabel("Pseudorange residual [m]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(site="Odaiba", data_root=None, max_epochs=120, rover_source="ublox",
         apply_tropo=True, out_dir=None) -> dict:
    data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    run_dir = data_root / site
    if not run_dir.is_dir():
        raise FileNotFoundError(f"UrbanNav run directory not found: {run_dir}")

    out_path = (Path(tempfile.mkdtemp(prefix="urbannav_calib_"))
                if out_dir is None else Path(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    loader = UrbanNavLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=max_epochs, rover_source=rover_source, systems=("G",))

    target = residual_array(
        residual_samples_from_experiment_data(data, apply_tropo=apply_tropo))
    target = target[np.isfinite(target)]

    gen = make_residual_generator(target.size)

    # Coarse grid search over physically plausible ranges.
    grid = grid_search(
        gen, target,
        {
            "nlos_fraction": [0.2, 0.35, 0.5, 0.65, 0.8],
            "nlos_bias_m": [0.0, 10.0, 20.0, 30.0],
            "nlos_scale_m": [10.0, 25.0, 40.0],
            "los_sigma_m": [3.0, 6.0, 10.0],
        },
        ks_weight=5.0,
    )

    # Refine the grid optimum with coordinate descent.
    refined = coordinate_descent(
        gen, target,
        init_params=grid["best_params"],
        bounds={
            "nlos_fraction": (0.05, 0.95),
            "nlos_bias_m": (0.0, 60.0),
            "nlos_scale_m": (2.0, 80.0),
            "los_sigma_m": (1.0, 20.0),
        },
        step={
            "nlos_fraction": 0.1,
            "nlos_bias_m": 8.0,
            "nlos_scale_m": 10.0,
            "los_sigma_m": 3.0,
        },
        n_iter=30,
        ks_weight=5.0,
    )

    best = refined["best_params"]
    fitted = gen(best)

    before = compare_distributions(gen(grid["best_params"]), target)
    after = compare_distributions(fitted, target)

    plot_path = out_path / f"urbannav_{site}_calibration_cdf.png"
    _plot_fit(target, fitted, plot_path,
              f"UrbanNav {site} residual calibration (grid + coordinate descent)")

    print(f"UrbanNav calibration: {data['dataset_name']}")
    print(f"epochs={data['n_epochs']}, target_samples={target.size}, "
          f"tropo={'on' if apply_tropo else 'off'}")
    print("recovered NLOS parameters:")
    print(f"  nlos_fraction = {best['nlos_fraction']:.3f}")
    print(f"  nlos_bias_m   = {best['nlos_bias_m']:.2f} m")
    print(f"  nlos_scale_m  = {best['nlos_scale_m']:.2f} m")
    print(f"  los_sigma_m   = {best['los_sigma_m']:.2f} m")
    print(f"grid  best score = {grid['best_score']:.3f} "
          f"(W1={before['wasserstein']:.2f}, KS={before['ks']:.3f})")
    print(f"final best score = {refined['best_score']:.3f} "
          f"(W1={after['wasserstein']:.2f}, KS={after['ks']:.3f})")
    print(f"Plot: {plot_path}")

    return {
        "dataset_name": data["dataset_name"],
        "site": site,
        "target_samples": int(target.size),
        "grid_score": float(grid["best_score"]),
        "final_score": float(refined["best_score"]),
        "wasserstein_before": float(before["wasserstein"]),
        "wasserstein_after": float(after["wasserstein"]),
        "ks_before": float(before["ks"]),
        "ks_after": float(after["ks"]),
        "best_params": {k: float(v) for k, v in best.items()},
        "plot_path": str(plot_path),
    }


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Odaiba"
    main(site=site)
