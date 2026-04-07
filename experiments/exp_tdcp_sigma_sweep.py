#!/usr/bin/env python3
"""TDCP-guided particle filter parameter sweep over sigma_pos_tdcp × position_update_sigma.

Goal: find the combination that pushes P50 below 1 m on both Odaiba and Shinjuku.

One dataset load per city; all grid combos reuse the cached dataset dict.

Example::

    cd experiments
    PYTHONPATH=.:../python:../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python \
      python3 exp_tdcp_sigma_sweep.py --data-root /tmp/UrbanNav-Tokyo

    # Preview grid without running
    python3 exp_tdcp_sigma_sweep.py --data-root /tmp/UrbanNav-Tokyo --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _PROJECT_ROOT / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exp_pf_smoother_eval import (
    RESULTS_DIR,
    load_pf_smoother_dataset,
    run_pf_with_optional_smoother,
)
from exp_urbannav_pf3d import PF_SIGMA_POS

# ---------- sweep grid ----------
DEFAULT_SIGMA_POS_TDCP = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
DEFAULT_PU_SIGMA = (1.5, 1.95, 2.5, 3.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TDCP sigma sweep: sigma_pos_tdcp x position_update_sigma"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument(
        "--sigma-pos-tdcp",
        type=str,
        default="",
        help="Override sigma_pos_tdcp list, comma floats (else use built-in grid)",
    )
    ap.add_argument(
        "--pu",
        type=str,
        default="",
        help="Override position_update_sigma list, comma floats (else use built-in grid)",
    )
    ap.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="TDCP guide: sin(el)^2 WLS weights",
    )
    ap.add_argument("--tdcp-el-sin-floor", type=float, default=0.1)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parameter grid and exit without running",
    )
    args = ap.parse_args()

    # parse grids
    if args.sigma_pos_tdcp.strip():
        sp_tdcp_list = tuple(
            float(x.strip()) for x in args.sigma_pos_tdcp.split(",") if x.strip()
        )
    else:
        sp_tdcp_list = DEFAULT_SIGMA_POS_TDCP

    if args.pu.strip():
        pu_list = tuple(float(x.strip()) for x in args.pu.split(",") if x.strip())
    else:
        pu_list = DEFAULT_PU_SIGMA

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    grid = list(itertools.product(sp_tdcp_list, pu_list))

    print(f"Runs:            {runs}")
    print(f"sigma_pos_tdcp:  {sp_tdcp_list}")
    print(f"PU sigma:        {pu_list}")
    print(f"Grid size:       {len(grid)} combos x {len(runs)} datasets = {len(grid) * len(runs)} runs")
    print(f"n_particles:     {args.n_particles}")
    print(f"max_epochs:      {args.max_epochs or 'full'}")
    print(f"predict_guide:   tdcp")
    print(f"elev_weight:     {args.tdcp_elevation_weight}")
    print()

    if args.dry_run:
        print("Parameter grid:")
        for i, (sp_tdcp, pu) in enumerate(grid, 1):
            print(f"  [{i:3d}] sigma_pos_tdcp={sp_tdcp}  position_update_sigma={pu}")
        print("\n--dry-run: exiting without running.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*76}")
        print(f"  {run_name}  (TDCP sigma sweep, {len(grid)} combos)")
        print(f"{'='*76}")
        print("  loading dataset (once)...", flush=True)
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)

        for sp_tdcp, pu in grid:
            print(
                f"  sp_tdcp={sp_tdcp}  PU={pu} ...",
                end=" ",
                flush=True,
            )
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                n_particles=args.n_particles,
                sigma_pos=float(PF_SIGMA_POS),
                sigma_pr=3.0,
                position_update_sigma=float(pu),
                predict_guide="tdcp",
                use_smoother=False,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                sigma_pos_tdcp=float(sp_tdcp),
                dataset=dataset,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            )
            fm = out["forward_metrics"]
            assert fm is not None, f"No metrics for {run_name} sp_tdcp={sp_tdcp} PU={pu}"
            print(
                f"P50={fm['p50']:.3f}m  P95={fm['p95']:.3f}m  RMS={fm['rms_2d']:.3f}m  "
                f"({fm['n_epochs']} ep)"
            )
            rows.append(
                {
                    "dataset": run_name,
                    "sigma_pos_tdcp": sp_tdcp,
                    "position_update_sigma": pu,
                    "P50": fm["p50"],
                    "P95": fm["p95"],
                    "RMS": fm["rms_2d"],
                    "n_epochs": fm["n_epochs"],
                    "sigma_pos": PF_SIGMA_POS,
                    "n_particles": args.n_particles,
                    "elapsed_ms": out["elapsed_ms"],
                    "tdcp_elevation_weight": args.tdcp_elevation_weight,
                    "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
                }
            )

    out_path = RESULTS_DIR / "tdcp_sigma_sweep.csv"
    if rows:
        fields = [
            "dataset",
            "sigma_pos_tdcp",
            "position_update_sigma",
            "P50",
            "P95",
            "RMS",
            "n_epochs",
            "sigma_pos",
            "n_particles",
            "elapsed_ms",
            "tdcp_elevation_weight",
            "tdcp_el_sin_floor",
        ]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")

        # per-dataset best
        for run_name in runs:
            sub = [r for r in rows if r["dataset"] == run_name]
            if sub:
                best = min(sub, key=lambda r: float(r["P50"]))
                print(
                    f"Best P50 {run_name}: "
                    f"sp_tdcp={best['sigma_pos_tdcp']}  PU={best['position_update_sigma']} "
                    f"=> P50={best['P50']:.3f}m  P95={best['P95']:.3f}m  RMS={best['RMS']:.3f}m"
                )

        # overall harmonic-mean ranking (lower is better)
        print("\nTop-5 combos by harmonic-mean P50 across datasets:")
        from collections import defaultdict

        combo_p50: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            key = (r["sigma_pos_tdcp"], r["position_update_sigma"])
            combo_p50[key].append(float(r["P50"]))
        ranked = sorted(
            combo_p50.items(),
            key=lambda kv: len(kv[1]) / sum(1.0 / p for p in kv[1]),  # harmonic mean
        )
        for i, (key, p50s) in enumerate(ranked[:5], 1):
            hmean = len(p50s) / sum(1.0 / p for p in p50s)
            parts = "  ".join(f"{p:.3f}m" for p in p50s)
            print(f"  #{i}  sp_tdcp={key[0]}  PU={key[1]}  hmean_P50={hmean:.3f}m  [{parts}]")


if __name__ == "__main__":
    main()
