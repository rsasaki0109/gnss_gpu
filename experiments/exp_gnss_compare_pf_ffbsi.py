#!/usr/bin/env python3
"""GNSS: fair PF forward vs FFBSi on the same UrbanNav stack (aligned metrics).

Loads the dataset once, runs ``run_pf_with_optional_smoother`` (forward-only) and
``run_pf_with_ffbsi`` with **identical** ``n_particles``, ``max_epochs``, PU, guide,
``resampling``, and TDCP knobs.

Forward uses the **weighted mean**; genealogy smoothing uses **sampled ancestral paths**
(averaged over ``--n-ffbsi-paths``), so P50 is not a strict plug-in Bayes comparison.

Example::

  PYTHONPATH=python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python \\
    python3 experiments/exp_gnss_compare_pf_ffbsi.py \\
      --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \\
      --n-particles 100000 --max-epochs 800 --position-update-sigma 1.95
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
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exp_ffbsi_eval import run_pf_with_ffbsi
from exp_pf_smoother_eval import load_pf_smoother_dataset, run_pf_with_optional_smoother

RESULTS_DIR = _SCRIPT_DIR / "results"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GNSS PF: forward vs FFBSi with matched N and epoch cap"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=800)
    ap.add_argument("--sigma-pos", type=float, default=2.0)
    ap.add_argument("--sigma-pr", type=float, default=3.0)
    ap.add_argument("--position-update-sigma", type=float, default=1.95)
    ap.add_argument("--predict-guide", choices=("spp", "tdcp"), default="spp")
    ap.add_argument("--n-ffbsi-paths", type=int, default=16)
    ap.add_argument(
        "--smoother",
        choices=("genealogy", "marginal"),
        default="genealogy",
        help="Particle smoother after forward (see exp_ffbsi_eval)",
    )
    ap.add_argument(
        "--resampling",
        choices=("systematic", "megopolis"),
        default="systematic",
        help="Must match smoother expectations: genealogy needs systematic",
    )
    ap.add_argument("--ffbsi-seed", type=int, default=12345)
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument("--sigma-pos-tdcp", type=float, default=None)
    ap.add_argument("--sigma-pos-tdcp-tight", type=float, default=None)
    ap.add_argument("--tdcp-tight-rms-max", type=float, default=1.0e9)
    ap.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="Match TDCP WLS elevation weighting on forward and FFBSi runs",
    )
    ap.add_argument("--tdcp-el-sin-floor", type=float, default=0.1)
    args = ap.parse_args()

    pu = float(args.position_update_sigma)
    if pu < 0:
        pu_use: float | None = None
    else:
        pu_use = pu

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict[str, object]] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n=== {run_name}: loading dataset once ===")
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)

        print("--- forward PF (smoother off) ---")
        fwd = run_pf_with_optional_smoother(
            run_dir,
            run_name,
            n_particles=args.n_particles,
            sigma_pos=args.sigma_pos,
            sigma_pr=args.sigma_pr,
            position_update_sigma=pu_use,
            predict_guide=args.predict_guide,
            use_smoother=False,
            rover_source=args.urban_rover,
            max_epochs=args.max_epochs,
            sigma_pos_tdcp=args.sigma_pos_tdcp,
            sigma_pos_tdcp_tight=args.sigma_pos_tdcp_tight,
            tdcp_tight_rms_max_m=args.tdcp_tight_rms_max,
            dataset=dataset,
            resampling=args.resampling,
            tdcp_elevation_weight=args.tdcp_elevation_weight,
            tdcp_el_sin_floor=args.tdcp_el_sin_floor,
        )
        fm = fwd["forward_metrics"]

        print(f"--- smoother ({args.smoother}, resampling={args.resampling}) ---")
        fb = run_pf_with_ffbsi(
            run_dir,
            run_name,
            n_particles=args.n_particles,
            sigma_pos=args.sigma_pos,
            sigma_pr=args.sigma_pr,
            position_update_sigma=pu_use,
            predict_guide=args.predict_guide,
            n_ffbsi_paths=args.n_ffbsi_paths,
            smoother_mode=args.smoother,
            rover_source=args.urban_rover,
            max_epochs=args.max_epochs,
            sigma_pos_tdcp=args.sigma_pos_tdcp,
            sigma_pos_tdcp_tight=args.sigma_pos_tdcp_tight,
            tdcp_tight_rms_max_m=args.tdcp_tight_rms_max,
            dataset=dataset,
            ffbsi_seed=args.ffbsi_seed,
            tdcp_elevation_weight=args.tdcp_elevation_weight,
            tdcp_el_sin_floor=args.tdcp_el_sin_floor,
        )
        fbm = fb["ffbsi_metrics"]

        if fm:
            print(
                f"  forward: P50={fm['p50']:.3f}m  RMS2D={fm['rms_2d']:.3f}m  "
                f"n_ep={fm['n_epochs']}"
            )
        if fbm:
            print(
                f"  smooth : P50={fbm['p50']:.3f}m  RMS2D={fbm['rms_2d']:.3f}m  "
                f"n_ep={fbm['n_epochs']}"
            )
        elif fm:
            print("  smooth : (no metrics — missing alignment?)")

        rows.append(
            {
                "run": run_name,
                "n_particles": args.n_particles,
                "max_epochs": args.max_epochs,
                "predict_guide": args.predict_guide,
                "position_update_sigma": pu_use if pu_use is not None else "off",
                "sigma_pos": args.sigma_pos,
                "sigma_pr": args.sigma_pr,
                "n_ffbsi_paths": args.n_ffbsi_paths,
                "smoother": args.smoother,
                "resampling": args.resampling,
                "tdcp_elevation_weight": args.tdcp_elevation_weight,
                "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
                "forward_p50": fm["p50"] if fm else None,
                "forward_rms_2d": fm["rms_2d"] if fm else None,
                "forward_n_epochs": fm["n_epochs"] if fm else None,
                "forward_elapsed_ms": fwd.get("elapsed_ms"),
                "ffbsi_p50": fbm["p50"] if fbm else None,
                "ffbsi_rms_2d": fbm["rms_2d"] if fbm else None,
                "ffbsi_n_epochs": fbm["n_epochs"] if fbm else None,
                "ffbsi_elapsed_ms": fb.get("elapsed_ms"),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "gnss_pf_ffbsi_fair_compare.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
