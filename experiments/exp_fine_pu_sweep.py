#!/usr/bin/env python3
"""Fine grid around position_update sigma (PU) where P50 was best (~2.0 m on Odaiba).

SPP velocity guide only; ``sigma_pos`` fixed at ``PF_SIGMA_POS``. One dataset load per city.

Example::

    PYTHONPATH=python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python \\
      python3 experiments/exp_fine_pu_sweep.py --data-root /tmp/UrbanNav-Tokyo
"""

from __future__ import annotations

import argparse
import csv
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

from exp_pf_smoother_eval import RESULTS_DIR, load_pf_smoother_dataset, run_pf_with_optional_smoother
from exp_urbannav_pf3d import PF_SIGMA_POS

# PU grid (m): centered on prior best ~2.0
DEFAULT_PU_VALUES = (
    1.65,
    1.7,
    1.75,
    1.8,
    1.85,
    1.9,
    1.95,
    2.0,
    2.05,
    2.1,
    2.15,
    2.2,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine PU sweep (SPP guide)")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument(
        "--pu",
        type=str,
        default="",
        help="Override PU list, comma floats (else use built-in grid)",
    )
    args = ap.parse_args()

    if args.pu.strip():
        pu_list = tuple(float(x.strip()) for x in args.pu.split(",") if x.strip())
    else:
        pu_list = DEFAULT_PU_VALUES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*72}\n  {run_name}  (fine PU, SPP guide)\n{'='*72}")
        print("  loading dataset (once)...", flush=True)
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)

        for pu in pu_list:
            tag = f"spp_pu_{pu:.3g}".replace(".", "p")
            print(f"  PU={pu} ...", end=" ", flush=True)
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                n_particles=args.n_particles,
                sigma_pos=float(PF_SIGMA_POS),
                sigma_pr=3.0,
                position_update_sigma=float(pu),
                predict_guide="spp",
                use_smoother=False,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                dataset=dataset,
            )
            fm = out["forward_metrics"]
            assert fm is not None
            print(f"P50={fm['p50']:.3f}m  RMS={fm['rms_2d']:.3f}m")
            rows.append(
                {
                    "run": run_name,
                    "tag": tag,
                    "position_update_sigma": pu,
                    "predict_guide": "spp",
                    "sigma_pos": PF_SIGMA_POS,
                    "forward_p50": fm["p50"],
                    "forward_p95": fm["p95"],
                    "forward_rms_2d": fm["rms_2d"],
                    "n_epochs": fm["n_epochs"],
                    "elapsed_ms": out["elapsed_ms"],
                    "n_particles": args.n_particles,
                }
            )

    out_path = RESULTS_DIR / "fine_pu_sweep.csv"
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")
        for run_name in runs:
            sub = [r for r in rows if r["run"] == run_name]
            if sub:
                best = min(sub, key=lambda r: float(r["forward_p50"]))
                print(
                    f"Best P50 {run_name}: PU={best['position_update_sigma']} "
                    f"=> P50={best['forward_p50']:.4f}m  RMS={best['forward_rms_2d']:.4f}m"
                )


if __name__ == "__main__":
    main()
