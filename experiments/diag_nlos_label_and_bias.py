#!/usr/bin/env python3
"""Validate ray-traced NLOS labels and the assumed NLOS bias model on real data.

Motivation: the synthetic per-particle 3DMA likelihood reached oracle accuracy,
but on real UrbanNav Odaiba turning the NLOS branch on monotonically *worsened*
the filter. Wiring is correct (p=0 reproduces the plain PF exactly), so either
the ray-traced labels are wrong or the assumed positive-bias Gaussian model is
mismatched.

This script isolates both effects using the reference trajectory as truth:

1. Compare each satellite's measured excess pseudorange
   ``excess = (pr - |sat - truth|) - clock`` against the ray-traced LOS/NLOS
   label. A well-calibrated label set has near-zero excess for ray-LOS and a
   clear positive excess for ray-NLOS.
2. Bin the ray-NLOS excess by elevation and compare it with the elevation model
   used by ``ParticleFilter3DBVH`` (``base + slope * max(0, ref - el)``).

Run from the repo root:

    PYTHONPATH=python:experiments:. python3 experiments/diag_nlos_label_and_bias.py \
      --data-dir data/urbannav/Tokyo/Odaiba \
      --model-dir data/plateau/odaiba --plateau-geoid 36.7 --plateau-bridges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "experiments" / "results" / "diag_nlos_label_and_bias_summary.json"
)

# Thresholds for calling an excess observation consistent with the label.
LOS_EXCESS_M = 8.0
NLOS_MIN_EXCESS_M = 5.0


def _load_mesh(model_dir: Path, zone: int, geoid: str, include_bridges: bool):
    from gnss_gpu.io.plateau import load_plateau

    if geoid.lower() == "none":
        correction: object = None
    else:
        try:
            correction = float(geoid)
        except ValueError:
            correction = geoid
    return load_plateau(
        model_dir,
        zone=zone,
        geoid_correction=correction,
        include_bridges=include_bridges,
    )


def _elevation_deg(truth_ecef: np.ndarray, sats_ecef: np.ndarray) -> np.ndarray:
    up = truth_ecef / np.linalg.norm(truth_ecef)
    los = sats_ecef - truth_ecef
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    sin_el = np.clip(los @ up, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_el))


def _summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "std_m": float(np.std(values)),
        "p10_m": float(np.percentile(values, 10)),
        "p90_m": float(np.percentile(values, 90)),
    }


def run_diag(
    data_dir: Path,
    model_dir: Path,
    *,
    zone: int = 9,
    geoid: str = "36.7",
    include_bridges: bool = True,
    systems: tuple[str, ...] = ("G",),
    urban_rover: str = "ublox",
    start_epoch: int = 0,
    max_epochs: int = 300,
    base_bias: float = 18.0,
    bias_slope: float = 1.25,
    elev_ref: float = 35.0,
    summary_json: Path | None = DEFAULT_SUMMARY_JSON,
) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from exp_urbannav_baseline import load_real_data
    from gnss_gpu.bvh import BVHAccelerator

    print(f"[1] Loading data from {data_dir} ...")
    data = load_real_data(
        data_dir,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        systems=systems,
        urban_rover=urban_rover,
    )
    if data is None:
        raise RuntimeError(f"could not load real data from {data_dir}")

    print(f"[2] Loading PLATEAU mesh from {model_dir} ...")
    model = _load_mesh(model_dir, zone, geoid, include_bridges)
    bvh = BVHAccelerator(model.triangles)
    print(f"    mesh: {model.triangles.shape[0]} triangles")

    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    n_epochs = data["n_epochs"]

    los_excess: list[float] = []
    nlos_excess: list[float] = []
    nlos_elev: list[float] = []
    los_elev: list[float] = []
    n_los = 0
    n_nlos = 0
    per_epoch_nlos_frac: list[float] = []

    for i in range(n_epochs):
        sats = np.asarray(sat_ecef[i], dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges[i], dtype=np.float64).ravel()
        if sats.shape[0] < 4 or pr.size != sats.shape[0]:
            continue
        truth = ground_truth[i, :3]
        ranges = np.linalg.norm(sats - truth, axis=1)
        clock = float(np.median(pr - ranges))
        excess = (pr - ranges) - clock
        is_los = np.asarray(bvh.check_los(truth, sats), dtype=bool)
        el = _elevation_deg(truth, sats)

        los_excess.extend(excess[is_los].tolist())
        nlos_excess.extend(excess[~is_los].tolist())
        los_elev.extend(el[is_los].tolist())
        nlos_elev.extend(el[~is_los].tolist())
        n_los += int(np.count_nonzero(is_los))
        n_nlos += int(np.count_nonzero(~is_los))
        per_epoch_nlos_frac.append(float(np.mean(~is_los)))

    los_excess_arr = np.asarray(los_excess, dtype=np.float64)
    nlos_excess_arr = np.asarray(nlos_excess, dtype=np.float64)
    nlos_elev_arr = np.asarray(nlos_elev, dtype=np.float64)

    # Label quality
    los_clean_frac = (
        float(np.mean(np.abs(los_excess_arr) <= LOS_EXCESS_M))
        if los_excess_arr.size
        else float("nan")
    )
    nlos_biased_frac = (
        float(np.mean(nlos_excess_arr >= NLOS_MIN_EXCESS_M))
        if nlos_excess_arr.size
        else float("nan")
    )
    nlos_zero_frac = (
        float(np.mean(np.abs(nlos_excess_arr) <= LOS_EXCESS_M))
        if nlos_excess_arr.size
        else float("nan")
    )

    # Elevation-binned NLOS bias vs model
    bins = [(lo, lo + 5) for lo in range(0, 90, 5)]
    elevation_bins = []
    for lo, hi in bins:
        mask = (nlos_elev_arr >= lo) & (nlos_elev_arr < hi)
        model_mid = base_bias + bias_slope * max(0.0, elev_ref - 0.5 * (lo + hi))
        observed = _summarize(nlos_excess_arr[mask])
        observed["elev_min_deg"] = lo
        observed["elev_max_deg"] = hi
        observed["assumed_model_m"] = model_mid
        elevation_bins.append(observed)

    # Empirical, heavy-tail-aware summary. A single linear ramp does not fit
    # the data (outliers dominate the fit), so report robust band statistics.
    low_mask = nlos_elev_arr < 25.0
    high_mask = nlos_elev_arr >= 30.0

    def _bands(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "median_m": float(np.median(values)),
            "p75_m": float(np.percentile(values, 75)),
            "p90_m": float(np.percentile(values, 90)),
            "mean_m": float(np.mean(values)),
            "std_m": float(np.std(values)),
            "near_zero_frac_le_8m": float(np.mean(np.abs(values) <= LOS_EXCESS_M)),
        }

    suggested = {
        "note": (
            "ray labels are valid; the assumed positive-floor Gaussian NLOS "
            "model is mis-specified. High-elevation NLOS excess is ~0 (the "
            "18 m floor is spurious) and the distribution is heavy-tailed "
            "(use a Laplace/Student-t-like NLOS likelihood)."
        ),
        "low_elev_lt_25_deg": _bands(nlos_excess_arr[low_mask]),
        "high_elev_ge_30_deg": _bands(nlos_excess_arr[high_mask]),
    }
    model_floor_active = base_bias > 0.0

    summary: dict[str, object] = {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "n_triangles": int(model.triangles.shape[0]),
        "n_epochs": int(n_epochs),
        "n_los": n_los,
        "n_nlos": n_nlos,
        "nlos_fraction": float(n_nlos / max(n_los + n_nlos, 1)),
        "nlos_frac_per_epoch_mean": float(np.mean(per_epoch_nlos_frac))
        if per_epoch_nlos_frac
        else float("nan"),
        "ray_los_excess": _summarize(los_excess_arr),
        "ray_nlos_excess": _summarize(nlos_excess_arr),
        "los_clean_frac_le_8m": los_clean_frac,
        "nlos_biased_frac_ge_5m": nlos_biased_frac,
        "nlos_near_zero_frac_le_8m": nlos_zero_frac,
        "assumed_model": {
            "base_m": base_bias,
            "slope_m_per_deg": bias_slope,
            "elev_ref_deg": elev_ref,
            "floor_active": model_floor_active,
        },
        "suggested_model": suggested,
        "elevation_bins": elevation_bins,
    }

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_json"] = str(summary_json)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _print_summary(s: dict[str, object]) -> None:
    print("\nRay-traced NLOS label / bias diagnostic")
    print("=" * 78)
    print(
        f"mesh={s['n_triangles']} tris  epochs={s['n_epochs']}  "
        f"NLOS={100.0 * s['nlos_fraction']:.1f}%  "
        f"({s['n_los']} LOS / {s['n_nlos']} NLOS)"
    )
    print("-" * 78)
    for key, label in (("ray_los_excess", "ray-LOS excess"),
                       ("ray_nlos_excess", "ray-NLOS excess")):
        v = s[key]
        if v.get("n", 0) == 0:
            print(f"{label:<16} (no samples)")
            continue
        print(
            f"{label:<16} n={v['n']:<7} mean={v['mean_m']:>8.2f} "
            f"median={v['median_m']:>8.2f} std={v['std_m']:>8.2f} "
            f"p10={v['p10_m']:>7.2f} p90={v['p90_m']:>7.2f} m"
        )
    print("-" * 78)
    print(f"ray-LOS clean (|excess|<=8m):  {100.0 * s['los_clean_frac_le_8m']:.1f}%")
    print(f"ray-NLOS biased (excess>=5m):  {100.0 * s['nlos_biased_frac_ge_5m']:.1f}%")
    print(f"ray-NLOS near-zero (<=8m):     {100.0 * s['nlos_near_zero_frac_le_8m']:.1f}%")
    print("-" * 78)
    print(f"{'elev bin':<14}{'model':>9}{'obs mean':>10}{'obs med':>9}{'obs std':>9}{'n':>7}")
    for b in s["elevation_bins"]:
        if b.get("n", 0) == 0:
            continue
        print(
            f"{b['elev_min_deg']:>2}-{b['elev_max_deg']:<3} deg    "
            f"{b['assumed_model_m']:>8.1f}{b['mean_m']:>10.1f}"
            f"{b['median_m']:>9.1f}{b['std_m']:>9.1f}{b['n']:>7}"
        )
    print("-" * 78)
    sug = s.get("suggested_model") or {}
    for key, label in (("low_elev_lt_25_deg", "NLOS <25deg"),
                       ("high_elev_ge_30_deg", "NLOS >=30deg")):
        band = sug.get(key, {})
        if band.get("n", 0) == 0:
            continue
        print(
            f"suggested {label:<14} n={band['n']:<5} median={band['median_m']:>7.2f} "
            f"p90={band['p90_m']:>7.2f} mean={band['mean_m']:>7.2f} "
            f"std={band['std_m']:>6.2f} near0={100.0 * band['near_zero_frac_le_8m']:>5.1f}%"
        )
    print("-" * 78)
    if s.get("summary_json"):
        print(f"summary: {s['summary_json']}")


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--plateau-zone", type=int, default=9)
    parser.add_argument("--plateau-geoid", type=str, default="36.7")
    parser.add_argument("--plateau-bridges", action="store_true")
    parser.add_argument("--systems", type=str, default="G")
    parser.add_argument("--urban-rover", type=str, default="ublox")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    summary = run_diag(
        args.data_dir,
        args.model_dir,
        zone=args.plateau_zone,
        geoid=args.plateau_geoid,
        include_bridges=args.plateau_bridges,
        systems=systems,
        urban_rover=args.urban_rover,
        start_epoch=args.start_epoch,
        max_epochs=args.max_epochs,
        summary_json=args.summary_json,
    )
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
