#!/usr/bin/env python3
"""Sweep PF NLOS down-weight factor k on a single PPC run (mask-soft variants only).

Reuses an existing smoke baseline when ``ppc_pf_nlos_smoke_{city}_{run}_baseline_runs.csv``
is present; otherwise runs baseline once. Compares honest/segment PPC vs baseline for each k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_pf_nlos_smoke import (
    DEFAULT_DATA_ROOT,
    PROFILE_DEFAULT_START_EPOCH,
    PROFILE_METHODS,
    _child_env,
    _mask_path,
    _metric_delta,
    _ppc_extra_args,
    _read_run_row,
    _resolve_hybrid_pos_dir,
    _resolve_methods,
    _run_dir_ok,
    _run_ppc,
    _validate_profile_args,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _parse_k_values(text: str) -> list[float]:
    out: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise SystemExit("no k values parsed")
    return out


def _baseline_prefix(city: str, run_name: str, profile: str) -> str:
    return f"ppc_pf_nlos_smoke_{city}_{run_name}_{profile}_baseline"


def _load_or_run_baseline(
    *,
    run: str,
    city: str,
    run_name: str,
    data_root: Path,
    methods: str,
    max_epochs: int,
    n_particles: int,
    start_epoch: int,
    hybrid_pos_dir: Path | None,
    ppc_extra_args: list[str],
    force_baseline: bool,
    profile: str,
) -> tuple[dict[str, float | str | None], Path]:
    prefix = _baseline_prefix(city, run_name, profile)
    runs_csv = RESULTS_DIR / f"{prefix}_runs.csv"
    if runs_csv.is_file() and not force_baseline:
        print(f"[ksweep] reuse baseline {runs_csv}", flush=True)
        return _read_run_row(runs_csv, methods), runs_csv
    runs_csv = _run_ppc(
        run=run,
        data_root=data_root,
        results_prefix=prefix,
        methods=methods,
        pf_nlos_preset=None,
        pf_nlos_mask_path=None,
        max_epochs=max_epochs,
        n_particles=n_particles,
        start_epoch=start_epoch,
        hybrid_pos_dir=hybrid_pos_dir,
        ppc_extra_args=ppc_extra_args,
    )
    return _read_run_row(runs_csv, methods), runs_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_METHODS),
        default="signal",
    )
    parser.add_argument("--methods", default=None)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=None)
    parser.add_argument("--ppc-extra-args", default="")
    parser.add_argument(
        "--k-values",
        default="3,5,10,20",
        help="Comma-separated PF NLOS k_weak=k_strong values",
    )
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip k when ppc_pf_nlos_ksweep_{city}_{run}_k{K}_runs.csv exists",
    )
    args = parser.parse_args(argv)

    run = str(args.run).strip().strip("/")
    city, run_name = run.split("/", 1)
    data_root = args.data_root.resolve()
    methods = _resolve_methods(args)
    ppc_extra_args = _ppc_extra_args(args)
    start_epoch = int(args.start_epoch)
    if start_epoch == 0 and args.profile in PROFILE_DEFAULT_START_EPOCH:
        start_epoch = int(PROFILE_DEFAULT_START_EPOCH[args.profile])
    hybrid_pos_dir = _resolve_hybrid_pos_dir(
        data_root,
        run,
        args.hybrid_pos_dir,
        str(args.profile),
    )
    _validate_profile_args(args, methods, run, hybrid_pos_dir)

    if not _run_dir_ok(data_root, run):
        print(f"[error] missing PPC run data under {data_root / run}", file=sys.stderr)
        return 2

    mask_csv = _mask_path(city, run_name)
    if not mask_csv.is_file():
        print(f"[error] missing mask CSV: {mask_csv}", file=sys.stderr)
        return 2

    k_values = _parse_k_values(str(args.k_values))
    baseline_metrics, baseline_csv = _load_or_run_baseline(
        run=run,
        city=city,
        run_name=run_name,
        data_root=data_root,
        methods=methods,
        max_epochs=int(args.max_epochs),
        n_particles=int(args.n_particles),
        start_epoch=start_epoch,
        hybrid_pos_dir=hybrid_pos_dir,
        ppc_extra_args=ppc_extra_args,
        force_baseline=bool(args.force_baseline),
        profile=str(args.profile),
    )

    variants: list[dict[str, object]] = []
    for k in k_values:
        k_tag = str(int(k)) if float(k).is_integer() else str(k).replace(".", "p")
        prefix = f"ppc_pf_nlos_ksweep_{city}_{run_name}_k{k_tag}"
        runs_csv = RESULTS_DIR / f"{prefix}_runs.csv"
        if args.skip_existing and runs_csv.is_file():
            print(f"[ksweep] reuse k={k} {runs_csv}", flush=True)
        else:
            runs_csv = _run_ppc(
                run=run,
                data_root=data_root,
                results_prefix=prefix,
                methods=methods,
                pf_nlos_preset=None,
                pf_nlos_mask_path=str(mask_csv),
                pf_nlos_k_weak=float(k),
                pf_nlos_k_strong=float(k),
                max_epochs=int(args.max_epochs),
                n_particles=int(args.n_particles),
                start_epoch=start_epoch,
                hybrid_pos_dir=hybrid_pos_dir,
                ppc_extra_args=ppc_extra_args,
            )
        metrics = _read_run_row(runs_csv, methods)
        variants.append(
            {
                "k": float(k),
                "runs_csv": str(runs_csv),
                "honest_ppc_pct": metrics.get("honest_ppc_pct"),
                "segment_ppc_pct": metrics.get("segment_ppc_pct"),
                "delta_honest_pp": _metric_delta(baseline_metrics, metrics, "honest_ppc_pct"),
                "delta_segment_pp": _metric_delta(baseline_metrics, metrics, "segment_ppc_pct"),
                "delta_segment_pass_m": _metric_delta(
                    baseline_metrics, metrics, "segment_pass_m"
                ),
                "metrics": metrics,
            }
        )

    summary: dict[str, object] = {
        "run": run,
        "profile": str(args.profile),
        "methods": methods,
        "start_epoch": start_epoch,
        "max_epochs": int(args.max_epochs),
        "mask_csv": str(mask_csv),
        "baseline_csv": str(baseline_csv),
        "baseline_metrics": baseline_metrics,
        "variants": variants,
    }
    out_json = RESULTS_DIR / f"ppc_pf_nlos_ksweep_{city}_{run_name}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
