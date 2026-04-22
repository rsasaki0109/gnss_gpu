#!/usr/bin/env python3
"""Try stacked knobs toward sub-meter P50 on UrbanNav (gnssplusplus PF).

Runs several configurations (TDCP + tighter predict noise, optional PU sweep)
without the forward-backward smoother by default (faster). Use --smoother to
include one smoother variant for the best-looking config.

Example::

    PYTHONPATH=python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python \\
      python3 experiments/exp_submeter_sweep.py \\
      --data-root /tmp/UrbanNav-Tokyo --runs Odaiba --n-particles 100000
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

from exp_pf_smoother_eval import (
    RESULTS_DIR,
    load_pf_smoother_dataset,
    run_pf_with_optional_smoother,
)
from exp_urbannav_pf3d import PF_SIGMA_POS

CONFIGS: tuple[dict, ...] = (
    {
        "tag": "spp_baseline",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "spp_pu24",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.4,
    },
    {
        "tag": "spp_pu22",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.2,
    },
    {
        "tag": "spp_sp15_pu30",
        "predict_guide": "spp",
        "sigma_pos": 1.5,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "tdcp_sp10",
        "predict_guide": "tdcp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": 1.0,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "tdcp_sp10_pu25",
        "predict_guide": "tdcp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": 1.0,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.5,
    },
    {
        "tag": "tdcp_sp10_pu20",
        "predict_guide": "tdcp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": 1.0,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.0,
    },
    {
        "tag": "tdcp_sp08_t15",
        "predict_guide": "tdcp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": 1.0,
        "sigma_pos_tdcp_tight": 0.8,
        "tdcp_tight_rms_max_m": 15.0,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "tdcp_sp06",
        "predict_guide": "tdcp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": 0.6,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sub-meter oriented PF config sweep")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument(
        "--smoother",
        action="store_true",
        help="Also run tdcp_sp10 + smoother (slow)",
    )
    ap.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="TDCP guide: sin(el)^2 row weights when elevation is available",
    )
    ap.add_argument("--tdcp-el-sin-floor", type=float, default=0.1)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*72}\n  {run_name}\n{'='*72}")
        print("  loading dataset (once)...", flush=True)
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)
        for cfg in CONFIGS:
            pu = cfg["position_update_sigma"]
            print(
                f"  {cfg['tag']}  guide={cfg['predict_guide']} "
                f"sp={cfg['sigma_pos']} sp_tdcp={cfg['sigma_pos_tdcp']} PU={pu} ...",
                end=" ",
                flush=True,
            )
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                n_particles=args.n_particles,
                sigma_pos=float(cfg["sigma_pos"]),
                sigma_pr=3.0,
                position_update_sigma=float(pu),
                predict_guide=str(cfg["predict_guide"]),
                use_smoother=False,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                sigma_pos_tdcp=cfg["sigma_pos_tdcp"],
                sigma_pos_tdcp_tight=cfg["sigma_pos_tdcp_tight"],
                tdcp_tight_rms_max_m=float(cfg["tdcp_tight_rms_max_m"]),
                dataset=dataset,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            )
            fm = out["forward_metrics"]
            assert fm is not None
            print(f"P50={fm['p50']:.2f}m  RMS={fm['rms_2d']:.2f}m")
            rows.append(
                {
                    "run": run_name,
                    "tag": cfg["tag"],
                    **{k: cfg[k] for k in cfg},
                    "forward_p50": fm["p50"],
                    "forward_p95": fm["p95"],
                    "forward_rms_2d": fm["rms_2d"],
                    "n_epochs": fm["n_epochs"],
                    "elapsed_ms": out["elapsed_ms"],
                }
            )

        if args.smoother:
            cfg = next(c for c in CONFIGS if c["tag"] == "tdcp_sp10")
            print(
                f"  {cfg['tag']}_smoothed  ...",
                end=" ",
                flush=True,
            )
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                n_particles=args.n_particles,
                sigma_pos=float(cfg["sigma_pos"]),
                sigma_pr=3.0,
                position_update_sigma=float(cfg["position_update_sigma"]),
                predict_guide=str(cfg["predict_guide"]),
                use_smoother=True,
                rover_source=args.urban_rover,
                max_epochs=args.max_epochs,
                sigma_pos_tdcp=cfg["sigma_pos_tdcp"],
                sigma_pos_tdcp_tight=cfg["sigma_pos_tdcp_tight"],
                tdcp_tight_rms_max_m=float(cfg["tdcp_tight_rms_max_m"]),
                dataset=dataset,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            )
            fm = out["forward_metrics"]
            sm = out["smoothed_metrics"]
            assert fm is not None
            sm_str = ""
            if sm:
                sm_str = f"  SMTH P50={sm['p50']:.2f}m RMS={sm['rms_2d']:.2f}m"
            print(f"FWD P50={fm['p50']:.2f}m{sm_str}")
            rows.append(
                {
                    "run": run_name,
                    "tag": cfg["tag"] + "_smoothed",
                    **{k: cfg[k] for k in cfg},
                    "forward_p50": fm["p50"],
                    "forward_p95": fm["p95"],
                    "forward_rms_2d": fm["rms_2d"],
                    "smoothed_p50": sm["p50"] if sm else None,
                    "smoothed_rms_2d": sm["rms_2d"] if sm else None,
                    "n_epochs": fm["n_epochs"],
                    "elapsed_ms": out["elapsed_ms"],
                }
            )

    out_path = RESULTS_DIR / "submeter_sweep.csv"
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")

        best = min(rows, key=lambda r: float(r["forward_p50"]))
        print(
            f"\nBest forward P50: {best['tag']} on {best['run']} "
            f"=> P50={best['forward_p50']:.3f}m  RMS={best['forward_rms_2d']:.3f}m"
        )


if __name__ == "__main__":
    main()
