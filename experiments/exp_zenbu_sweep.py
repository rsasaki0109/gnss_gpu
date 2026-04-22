#!/usr/bin/env python3
"""Comprehensive UrbanNav PF sweep: Tokyo runs × many forward configs × selected smoothers.

- Loads each run's RINEX + GT once per city, then reuses ``dataset=`` for all configs.
- Forward passes: all entries in ``ZENBU_CONFIGS`` (SPP grid + TDCP variants).
- Optional smoother re-runs for a short whitelist of tags (see ``--smoother-tags``).

Example::

    PYTHONPATH=python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python \\
      python3 experiments/exp_zenbu_sweep.py \\
      --data-root /tmp/UrbanNav-Tokyo --n-particles 100000
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
from exp_submeter_sweep import CONFIGS as SUBMETER_CONFIGS
from exp_urbannav_pf3d import PF_SIGMA_POS

# Extra configs beyond exp_submeter_sweep (same schema as those dicts)
_EXTRA: tuple[dict, ...] = (
    {
        "tag": "spp_pu26",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.6,
    },
    {
        "tag": "spp_pu28",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.8,
    },
    {
        "tag": "spp_pu20",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 2.0,
    },
    {
        "tag": "spp_pu18",
        "predict_guide": "spp",
        "sigma_pos": PF_SIGMA_POS,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 1.8,
    },
    {
        "tag": "spp_sp12_pu30",
        "predict_guide": "spp",
        "sigma_pos": 1.2,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "spp_sp18_pu30",
        "predict_guide": "spp",
        "sigma_pos": 1.8,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "spp_sp22_pu30",
        "predict_guide": "spp",
        "sigma_pos": 2.2,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
    {
        "tag": "spp_sp25_pu30",
        "predict_guide": "spp",
        "sigma_pos": 2.5,
        "sigma_pos_tdcp": None,
        "sigma_pos_tdcp_tight": None,
        "tdcp_tight_rms_max_m": 1.0e9,
        "position_update_sigma": 3.0,
    },
)

# Dedupe by tag (submeter first, then extras skip if tag exists)
def _merge_configs() -> tuple[dict, ...]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in SUBMETER_CONFIGS + _EXTRA:
        t = str(c["tag"])
        if t in seen:
            continue
        seen.add(t)
        out.append(c)
    return tuple(out)


ZENBU_CONFIGS = _merge_configs()

DEFAULT_SMOOTHER_TAGS = (
    "spp_baseline",
    "spp_pu22",
    "spp_pu24",
    "tdcp_sp10",
    "tdcp_sp10_pu20",
)


def _run_one(
    *,
    run_dir: Path,
    run_name: str,
    dataset: dict,
    cfg: dict,
    n_particles: int,
    rover: str,
    max_epochs: int,
    use_smoother: bool,
    rows: list[dict],
    tdcp_elevation_weight: bool = False,
    tdcp_el_sin_floor: float = 0.1,
) -> None:
    pu = cfg["position_update_sigma"]
    tag = cfg["tag"]
    sm_lab = "_smth" if use_smoother else ""
    print(
        f"  {tag}{sm_lab}  guide={cfg['predict_guide']} sp={cfg['sigma_pos']} "
        f"sp_tdcp={cfg['sigma_pos_tdcp']} PU={pu} sm={use_smoother} ...",
        end=" ",
        flush=True,
    )
    out = run_pf_with_optional_smoother(
        run_dir,
        run_name,
        n_particles=n_particles,
        sigma_pos=float(cfg["sigma_pos"]),
        sigma_pr=3.0,
        position_update_sigma=float(pu),
        predict_guide=str(cfg["predict_guide"]),
        use_smoother=use_smoother,
        rover_source=rover,
        max_epochs=max_epochs,
        sigma_pos_tdcp=cfg["sigma_pos_tdcp"],
        sigma_pos_tdcp_tight=cfg["sigma_pos_tdcp_tight"],
        tdcp_tight_rms_max_m=float(cfg["tdcp_tight_rms_max_m"]),
        dataset=dataset,
        tdcp_elevation_weight=tdcp_elevation_weight,
        tdcp_el_sin_floor=tdcp_el_sin_floor,
    )
    fm = out["forward_metrics"]
    sm = out["smoothed_metrics"]
    assert fm is not None
    line = f"FWD P50={fm['p50']:.2f}m RMS={fm['rms_2d']:.2f}m"
    if sm:
        line += f" | SMTH P50={sm['p50']:.2f}m RMS={sm['rms_2d']:.2f}m"
    print(line)
    base = {k: cfg[k] for k in cfg}
    base.pop("tag", None)
    row = {
        **base,
        "run": run_name,
        "tag": tag + ("+smoother" if use_smoother else ""),
        "smooth": use_smoother,
        "forward_p50": fm["p50"],
        "forward_p95": fm["p95"],
        "forward_rms_2d": fm["rms_2d"],
        "smoothed_p50": sm["p50"] if sm else None,
        "smoothed_p95": sm["p95"] if sm else None,
        "smoothed_rms_2d": sm["rms_2d"] if sm else None,
        "n_epochs": fm["n_epochs"],
        "elapsed_ms": out["elapsed_ms"],
        "tdcp_elevation_weight": tdcp_elevation_weight,
        "tdcp_el_sin_floor": tdcp_el_sin_floor,
    }
    rows.append(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zenbu (full) PF sweep on UrbanNav Tokyo")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--urban-rover", type=str, default="trimble")
    ap.add_argument(
        "--smoother-tags",
        type=str,
        default=",".join(DEFAULT_SMOOTHER_TAGS),
        help="Comma tags to re-run with forward-backward smoother (empty to skip)",
    )
    ap.add_argument(
        "--forward-only",
        action="store_true",
        help="Only forward passes (ignore smoother-tags)",
    )
    ap.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="TDCP guide: sin(el)^2 WLS weights when elevation is set on measurements",
    )
    ap.add_argument("--tdcp-el-sin-floor", type=float, default=0.1)
    args = ap.parse_args()

    smoother_tags = set()
    if not args.forward_only and args.smoother_tags.strip():
        smoother_tags = {t.strip() for t in args.smoother_tags.split(",") if t.strip()}

    tag_to_cfg = {str(c["tag"]): c for c in ZENBU_CONFIGS}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*76}\n  {run_name}\n{'='*76}")
        print("  loading dataset (once)...", flush=True)
        dataset = load_pf_smoother_dataset(run_dir, args.urban_rover)

        for cfg in ZENBU_CONFIGS:
            _run_one(
                run_dir=run_dir,
                run_name=run_name,
                dataset=dataset,
                cfg=cfg,
                n_particles=args.n_particles,
                rover=args.urban_rover,
                max_epochs=args.max_epochs,
                use_smoother=False,
                rows=rows,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            )

        for st in sorted(smoother_tags):
            if st not in tag_to_cfg:
                print(f"  [skip smoother] unknown tag {st!r}", flush=True)
                continue
            _run_one(
                run_dir=run_dir,
                run_name=run_name,
                dataset=dataset,
                cfg=tag_to_cfg[st],
                n_particles=args.n_particles,
                rover=args.urban_rover,
                max_epochs=args.max_epochs,
                use_smoother=True,
                rows=rows,
                tdcp_elevation_weight=args.tdcp_elevation_weight,
                tdcp_el_sin_floor=args.tdcp_el_sin_floor,
            )

    out_path = RESULTS_DIR / "zenbu_sweep.csv"
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")

        fwd_only = [r for r in rows if not r.get("smooth")]
        if fwd_only:
            best = min(fwd_only, key=lambda r: float(r["forward_p50"]))
            print(
                f"\nBest forward P50: {best['tag']} @ {best['run']} "
                f"=> P50={best['forward_p50']:.3f}m RMS={best['forward_rms_2d']:.3f}m"
            )
        sm_rows = [r for r in rows if r.get("smooth")]
        if sm_rows:
            best_s = min(sm_rows, key=lambda r: float(r.get("smoothed_p50") or r["forward_p50"]))
            sp50 = best_s.get("smoothed_p50")
            srms = best_s.get("smoothed_rms_2d")
            print(
                f"Best smoothed P50: {best_s['tag']} @ {best_s['run']} "
                f"=> P50={float(sp50):.3f}m RMS={float(srms):.3f}m"
            )


if __name__ == "__main__":
    main()
