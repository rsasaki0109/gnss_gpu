"""Extract real pseudorange residuals from UrbanNav RINEX and summarise them.

This wires the real-data bridge end to end:

    UrbanNav rover RINEX obs + broadcast nav + ground truth
        -> UrbanNavLoader.load_experiment_data (GPU ephemeris -> sat ECEF)
        -> residual_samples_from_experiment_data (receiver-clock removed by
           per-epoch median)
        -> distribution summary, elevation binning, CSV + empirical-CDF plot

The residuals contain receiver-relative range errors: multipath / NLOS plus
any unmodelled atmospheric delay (no tropo/iono correction is applied). They
are exactly the kind of target distribution the urban NLOS simulator can be
calibrated against via gnss_gpu.validation.calibration.

Requires the compiled GPU ephemeris extension and a local UrbanNav run
directory (default: experiments/data/urbannav/Odaiba).

Run:  python examples/demo_urbannav_residuals.py [Odaiba|Shinjuku]
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
from gnss_gpu.io.nav_rinex import read_gps_klobuchar_from_nav_header
from gnss_gpu.validation import (
    bin_by_elevation,
    residual_samples_from_experiment_data,
    summarize,
    write_csv,
)
from gnss_gpu.validation.real_residuals import residual_array

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "experiments" / "data" / "urbannav"


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return summarize(values)


def _plot_cdf(series, path, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)
    for values, label in series:
        v = np.sort(np.asarray(values, dtype=float))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1, dtype=float) / v.size
        ax.plot(v, y, linewidth=2.0, label=f"{label} (n={v.size})")
    ax.set_xlabel("|Pseudorange residual| [m]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(site="Odaiba", data_root=None, max_epochs=120, rover_source="ublox",
         out_dir=None) -> dict:
    data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    run_dir = data_root / site
    if not run_dir.is_dir():
        raise FileNotFoundError(f"UrbanNav run directory not found: {run_dir}")

    out_path = (Path(tempfile.mkdtemp(prefix="urbannav_residuals_"))
                if out_dir is None else Path(out_dir))
    out_path.mkdir(parents=True, exist_ok=True)

    loader = UrbanNavLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=max_epochs, rover_source=rover_source, systems=("G",))

    # Klobuchar iono coefficients from the broadcast nav (fall back to the
    # atmosphere module defaults when the nav header lacks them, as the
    # UrbanNav merged nav files do).
    nav_path = run_dir / "base.nav"
    iono_alpha = iono_beta = None
    iono_source = "default"
    if nav_path.is_file():
        a, b = read_gps_klobuchar_from_nav_header(nav_path)
        if a is not None and b is not None:
            iono_alpha, iono_beta, iono_source = a, b, "nav-header"

    samples = residual_samples_from_experiment_data(data)
    samples_tropo = residual_samples_from_experiment_data(data, apply_tropo=True)
    samples_atmo = residual_samples_from_experiment_data(
        data, apply_tropo=True, apply_iono=True,
        iono_alpha=iono_alpha, iono_beta=iono_beta)

    residuals = residual_array(samples)
    abs_res = np.abs(residuals)
    abs_res_tropo = np.abs(residual_array(samples_tropo))
    abs_res_atmo = np.abs(residual_array(samples_atmo))

    overall = _summary(residuals)
    overall_tropo = _summary(residual_array(samples_tropo))
    overall_atmo = _summary(residual_array(samples_atmo))
    elev_bins = bin_by_elevation(samples, [5, 15, 30, 50, 90])

    csv_path = out_path / f"urbannav_{site}_residuals.csv"
    plot_path = out_path / f"urbannav_{site}_residual_cdf.png"
    write_csv(samples, csv_path)
    _plot_cdf(
        [(abs_res, "raw"), (abs_res_tropo, "+tropo"),
         (abs_res_atmo, f"+tropo+iono ({iono_source})")],
        plot_path,
        f"UrbanNav {site} ({rover_source}) residuals: atmospheric purification")

    print(f"UrbanNav real-data residuals: {data['dataset_name']}")
    print(f"epochs={data['n_epochs']}, median_sats={data['n_satellites']}, "
          f"samples={len(samples)}, iono_coeffs={iono_source}")
    if overall is not None:
        print(f"  raw         |residual| p50={np.percentile(abs_res, 50):7.2f} m  "
              f"p90={np.percentile(abs_res, 90):7.2f} m  rms={overall['rms']:7.2f} m")
        print(f"  +tropo      |residual| p50={np.percentile(abs_res_tropo, 50):7.2f} m  "
              f"p90={np.percentile(abs_res_tropo, 90):7.2f} m  rms={overall_tropo['rms']:7.2f} m")
        print(f"  +tropo+iono |residual| p50={np.percentile(abs_res_atmo, 50):7.2f} m  "
              f"p90={np.percentile(abs_res_atmo, 90):7.2f} m  rms={overall_atmo['rms']:7.2f} m")
    print("  by elevation (|residual| p90):")
    for label, bin_samples in elev_bins.items():
        vals = np.abs(residual_array(bin_samples))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f"    {str(label):>10s}  n={vals.size:4d}  "
                  f"p90={np.percentile(vals, 90):7.2f} m")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")

    return {
        "dataset_name": data["dataset_name"],
        "site": site,
        "n_epochs": int(data["n_epochs"]),
        "n_samples": int(len(samples)),
        "abs_p50_m": float(np.percentile(abs_res, 50)) if abs_res.size else float("nan"),
        "abs_p90_m": float(np.percentile(abs_res, 90)) if abs_res.size else float("nan"),
        "rms_m": float(overall["rms"]) if overall else float("nan"),
        "abs_p50_tropo_m": float(np.percentile(abs_res_tropo, 50)) if abs_res_tropo.size else float("nan"),
        "abs_p90_tropo_m": float(np.percentile(abs_res_tropo, 90)) if abs_res_tropo.size else float("nan"),
        "rms_tropo_m": float(overall_tropo["rms"]) if overall_tropo else float("nan"),
        "abs_p50_atmo_m": float(np.percentile(abs_res_atmo, 50)) if abs_res_atmo.size else float("nan"),
        "abs_p90_atmo_m": float(np.percentile(abs_res_atmo, 90)) if abs_res_atmo.size else float("nan"),
        "rms_atmo_m": float(overall_atmo["rms"]) if overall_atmo else float("nan"),
        "iono_coeffs": iono_source,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Odaiba"
    main(site=site)
