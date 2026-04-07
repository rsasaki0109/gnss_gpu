#!/usr/bin/env python3
"""TDCP adaptive mode sweep: tdcp_rms_threshold × sigma_pos_tdcp × position_update_sigma.

Sweeps the adaptive fallback threshold that controls when TDCP velocity is
rejected (postfit RMS >= threshold) in favour of Doppler/random-walk predict.

Example::

    cd experiments
    PYTHONPATH=.:../python:../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python \
      python3 exp_tdcp_adaptive_sweep.py --data-root /tmp/UrbanNav-Tokyo

    # Preview grid without running
    python3 exp_tdcp_adaptive_sweep.py --data-root /tmp/UrbanNav-Tokyo --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
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
DEFAULT_RMS_THRESHOLD = (1.0, 2.0, 3.0, 5.0, 8.0)
DEFAULT_SIGMA_POS_TDCP = (1.0, 1.5, 2.0, 3.0)
DEFAULT_PU_SIGMA = (1.5, 1.95)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TDCP adaptive sweep: rms_threshold x sigma_pos_tdcp x position_update_sigma"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument(
        "--rms-threshold",
        type=str,
        default="",
        help="Override rms_threshold list, comma floats",
    )
    ap.add_argument(
        "--sigma-pos-tdcp",
        type=str,
        default="",
        help="Override sigma_pos_tdcp list, comma floats",
    )
    ap.add_argument(
        "--pu",
        type=str,
        default="",
        help="Override position_update_sigma list, comma floats",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parameter grid and exit without running",
    )
    args = ap.parse_args()

    # parse grids
    if args.rms_threshold.strip():
        rms_list = tuple(
            float(x.strip()) for x in args.rms_threshold.split(",") if x.strip()
        )
    else:
        rms_list = DEFAULT_RMS_THRESHOLD

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
    grid = list(itertools.product(rms_list, sp_tdcp_list, pu_list))

    print(f"Runs:              {runs}")
    print(f"rms_threshold:     {rms_list}")
    print(f"sigma_pos_tdcp:    {sp_tdcp_list}")
    print(f"PU sigma:          {pu_list}")
    print(f"Grid size:         {len(grid)} combos x {len(runs)} datasets = {len(grid) * len(runs)} runs")
    print(f"n_particles:       {args.n_particles}")
    print(f"max_epochs:        {args.max_epochs or 'full'}")
    print(f"predict_guide:     tdcp_adaptive")
    print()

    if args.dry_run:
        print("Parameter grid:")
        for i, (rms_th, sp_tdcp, pu) in enumerate(grid, 1):
            print(
                f"  [{i:3d}] rms_threshold={rms_th}  "
                f"sigma_pos_tdcp={sp_tdcp}  position_update_sigma={pu}"
            )
        print(f"\n--dry-run: exiting without running.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*76}")
        print(f"  {run_name}  (TDCP adaptive sweep, {len(grid)} combos)")
        print(f"{'='*76}")
        print("  loading dataset (once)...", flush=True)
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)

        for rms_th, sp_tdcp, pu in grid:
            print(
                f"  rms_th={rms_th}  sp_tdcp={sp_tdcp}  PU={pu} ...",
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
                predict_guide="tdcp_adaptive",
                use_smoother=False,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                sigma_pos_tdcp=float(sp_tdcp),
                dataset=dataset,
                tdcp_rms_threshold=float(rms_th),
            )
            fm = out["forward_metrics"]
            assert fm is not None, (
                f"No metrics for {run_name} rms_th={rms_th} sp_tdcp={sp_tdcp} PU={pu}"
            )
            print(
                f"P50={fm['p50']:.3f}m  P95={fm['p95']:.3f}m  RMS={fm['rms_2d']:.3f}m  "
                f"({fm['n_epochs']} ep)  "
                f"tdcp_used={out['n_tdcp_used']}  fallback={out['n_tdcp_fallback']}"
            )
            rows.append(
                {
                    "dataset": run_name,
                    "tdcp_rms_threshold": rms_th,
                    "sigma_pos_tdcp": sp_tdcp,
                    "position_update_sigma": pu,
                    "P50": fm["p50"],
                    "P95": fm["p95"],
                    "RMS": fm["rms_2d"],
                    "n_epochs": fm["n_epochs"],
                    "n_tdcp_used": out["n_tdcp_used"],
                    "n_tdcp_fallback": out["n_tdcp_fallback"],
                    "sigma_pos": PF_SIGMA_POS,
                    "n_particles": args.n_particles,
                    "elapsed_ms": out["elapsed_ms"],
                }
            )

    out_path = RESULTS_DIR / "tdcp_adaptive_sweep.csv"
    if rows:
        fields = [
            "dataset",
            "tdcp_rms_threshold",
            "sigma_pos_tdcp",
            "position_update_sigma",
            "P50",
            "P95",
            "RMS",
            "n_epochs",
            "n_tdcp_used",
            "n_tdcp_fallback",
            "sigma_pos",
            "n_particles",
            "elapsed_ms",
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
                    f"rms_th={best['tdcp_rms_threshold']}  "
                    f"sp_tdcp={best['sigma_pos_tdcp']}  PU={best['position_update_sigma']} "
                    f"=> P50={best['P50']:.3f}m  P95={best['P95']:.3f}m  RMS={best['RMS']:.3f}m"
                )

        # overall harmonic-mean ranking (lower is better)
        print("\nTop-5 combos by harmonic-mean P50 across datasets:")
        combo_p50: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            key = (r["tdcp_rms_threshold"], r["sigma_pos_tdcp"], r["position_update_sigma"])
            combo_p50[key].append(float(r["P50"]))
        ranked = sorted(
            combo_p50.items(),
            key=lambda kv: len(kv[1]) / sum(1.0 / p for p in kv[1]),  # harmonic mean
        )
        for i, (key, p50s) in enumerate(ranked[:5], 1):
            hmean = len(p50s) / sum(1.0 / p for p in p50s)
            parts = "  ".join(f"{p:.3f}m" for p in p50s)
            print(
                f"  #{i}  rms_th={key[0]}  sp_tdcp={key[1]}  PU={key[2]}  "
                f"hmean_P50={hmean:.3f}m  [{parts}]"
            )


if __name__ == "__main__":
    main()
