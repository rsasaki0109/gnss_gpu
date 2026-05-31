#!/usr/bin/env python3
"""Sweep fixed GLO ICB RTK sources and TDCP-height materialized candidates.

This is a diagnostic candidate generator for the PPC CT-RBPF/FGO rescue pool:
for each requested segment it runs libgnss++ RTK with fixed GLONASS ICB values,
then keeps the resulting ECEF X/Y and projects ECEF Z onto a TDCP-derived
ellipsoidal-height prior.  The script writes both source RTK outputs and
discoverable TDCP-height candidate directories, plus a CSV with raw/window
metrics for triage.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exp_ppc_ctrbpf_fgo import _load_full_reference, _load_hybrid_pos_file  # noqa: E402
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from materialize_ppc_tdcp_height_prior_candidate import (  # noqa: E402
    _copy_csv_window,
    _estimate_tdcp_height_series,
    _rewrite_pos_window,
)


_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    if not out:
        raise ValueError(f"empty numeric list: {spec!r}")
    return out


def _icb_tag(value: float) -> str:
    if float(value).is_integer():
        iv = int(value)
        return f"m{abs(iv)}" if iv < 0 else f"p{iv}"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _window_rows(path: Path, only_windows: set[tuple[int, int]]) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("diagnosis", "")).strip() not in {"", "candidate_generation_needed"}:
            continue
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        if only_windows and (start_idx, end_idx) not in only_windows:
            continue
        out.append(row)
    return out


def _parse_only_windows(spec: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"bad window spec: {chunk!r}")
        a, b = chunk.split("-", 1)
        out.add((int(a), int(b)))
    return out


def _window_tows(ref: list[tuple[float, np.ndarray]], start_idx: int, end_idx: int) -> tuple[float, float]:
    if start_idx < 0 or end_idx >= len(ref) or end_idx < start_idx:
        raise IndexError(f"bad index window {start_idx}-{end_idx} for reference length {len(ref)}")
    return round(float(ref[start_idx][0]), 1), round(float(ref[end_idx][0]), 1)


def _load_est_for_window(pos_path: Path, ref_window: list[tuple[float, np.ndarray]]) -> np.ndarray:
    positions, _statuses = _load_hybrid_pos_file(pos_path)
    est = np.zeros((len(ref_window), 3), dtype=np.float64)
    for i, (tow, _truth) in enumerate(ref_window):
        pos = positions.get(round(float(tow), 1))
        if pos is not None and np.all(np.isfinite(pos)):
            est[i] = np.asarray(pos, dtype=np.float64)
    return est


def _metric_row(est: np.ndarray, ref_window: list[tuple[float, np.ndarray]]) -> dict[str, float]:
    truth = np.asarray([p for _tow, p in ref_window], dtype=np.float64)
    score = score_ppc2024(est, truth)
    err = np.linalg.norm(est - truth, axis=1)
    finite = np.isfinite(err) & ~np.all(est == 0.0, axis=1)
    if np.any(finite):
        p50 = float(np.percentile(err[finite], 50))
        p95 = float(np.percentile(err[finite], 95))
        lt1 = 100.0 * float(np.mean(err[finite] < 1.0))
        valid = int(np.sum(finite))
    else:
        p50 = float("nan")
        p95 = float("nan")
        lt1 = 0.0
        valid = 0
    return {
        "ppc_pct": float(score.score_pct),
        "pass_m": float(score.pass_distance_m),
        "total_m": float(score.total_distance_m),
        "epoch_pass_pct": float(score.epoch_pass_pct),
        "p50_3d_m": p50,
        "p95_3d_m": p95,
        "lt1m_valid_pct": lt1,
        "valid_epochs": float(valid),
    }


def _run_gnss_solve(
    *,
    solver: Path,
    run_dir: Path,
    skip_epochs: int,
    max_epochs: int,
    out_pos: Path,
    out_csv: Path,
    l1: float,
    l2: float,
    ratio: float,
    carrier_phase_sigma: float,
    min_hold_count: int,
    hold_ratio_threshold: float,
    elevation_mask_deg: float,
) -> None:
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(solver),
        "--rover",
        str(run_dir / "rover.obs"),
        "--base",
        str(run_dir / "base.obs"),
        "--nav",
        str(run_dir / "base.nav"),
        "--skip-epochs",
        str(skip_epochs),
        "--max-epochs",
        str(max_epochs),
        "--out",
        str(out_pos),
        "--diagnostics-csv",
        str(out_csv),
        "--no-kml",
        "--preset",
        "low-cost",
        "--ratio",
        str(ratio),
        "--carrier-phase-sigma",
        str(carrier_phase_sigma),
        "--min-hold-count",
        str(min_hold_count),
        "--hold-ratio-threshold",
        str(hold_ratio_threshold),
        "--elevation-mask-deg",
        str(elevation_mask_deg),
        "--glonass-ar",
        "on",
        "--glonass-icb-l1",
        str(l1),
        "--glonass-icb-l2",
        str(l2),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=_SCRIPT_DIR / "results" / "ppc_relative_bias_oracle_phase11eo_n2_candgen.csv",
    )
    parser.add_argument("--only-windows", default="")
    parser.add_argument("--icb-l1-values", default="-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6")
    parser.add_argument("--icb-l2-values", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--solver", type=Path, default=_PROJECT_ROOT / "third_party/gnssplusplus/build/apps/gnss_solve")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_SCRIPT_DIR / "results" / "libgnss_rtk_segment_probe_phase11es_n2_icbsweep",
    )
    parser.add_argument(
        "--diag-root",
        type=Path,
        default=_SCRIPT_DIR / "results" / "libgnss_diag_phase10",
    )
    parser.add_argument("--out-csv", type=Path, default=_SCRIPT_DIR / "results" / "ppc_fixed_icb_tdcp_height_sweep_n2.csv")
    parser.add_argument("--out-prefix", default="tdcp_height_prior_n2_icbsweep")
    parser.add_argument("--max-tdcp-epochs", type=int, default=7200)
    parser.add_argument("--tdcp-postfit-max-m", type=float, default=2.0)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--carrier-phase-sigma", type=float, default=0.0005)
    parser.add_argument("--min-hold-count", type=int, default=1)
    parser.add_argument("--hold-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--elevation-mask-deg", type=float, default=3.0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    run_dir = args.data_root / args.city / args.run
    ref = _load_full_reference(run_dir / "reference.csv")
    only_windows = _parse_only_windows(args.only_windows)
    windows = _window_rows(args.windows_csv, only_windows)
    if args.limit > 0:
        windows = windows[: int(args.limit)]
    if not windows:
        raise SystemExit("no windows selected")

    height_series, accepted = _estimate_tdcp_height_series(
        run_dir,
        int(args.max_tdcp_epochs),
        float(args.tdcp_postfit_max_m),
    )
    l1_values = _parse_float_list(args.icb_l1_values)
    l2_values = _parse_float_list(args.icb_l2_values)
    rows: list[dict[str, object]] = []

    for row in windows:
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        start_tow, end_tow = _window_tows(ref, start_idx, end_idx)
        ref_window = [item for item in ref if start_tow <= round(float(item[0]), 1) <= end_tow]
        heights = [h for tow, h in height_series.items() if start_tow <= float(tow) <= end_tow]
        if not heights:
            print(f"skip {start_idx}-{end_idx}: no TDCP heights", flush=True)
            continue
        height_m = float(np.median(heights))

        for l1 in l1_values:
            for l2 in l2_values:
                tag = f"l1{_icb_tag(l1)}_l2{_icb_tag(l2)}"
                stem = f"{args.city}_{args.run}_{start_idx}_{end_idx}_{tag}"
                source_pos = args.source_root / f"{stem}.pos"
                source_csv = args.source_root / f"{stem}.csv"
                if not args.reuse or not (source_pos.is_file() and source_csv.is_file()):
                    print(f"solve {start_idx}-{end_idx} {tag}", flush=True)
                    _run_gnss_solve(
                        solver=args.solver,
                        run_dir=run_dir,
                        skip_epochs=start_idx,
                        max_epochs=end_idx - start_idx + 1,
                        out_pos=source_pos,
                        out_csv=source_csv,
                        l1=l1,
                        l2=l2,
                        ratio=float(args.ratio),
                        carrier_phase_sigma=float(args.carrier_phase_sigma),
                        min_hold_count=int(args.min_hold_count),
                        hold_ratio_threshold=float(args.hold_ratio_threshold),
                        elevation_mask_deg=float(args.elevation_mask_deg),
                    )

                cand_dir_name = f"{args.out_prefix}_{start_idx}_{end_idx}_{tag}"
                cand_dir = args.diag_root / cand_dir_name
                cand_dir.mkdir(parents=True, exist_ok=True)
                cand_pos = cand_dir / f"{args.city}_{args.run}_full.pos"
                cand_csv = cand_dir / f"{args.city}_{args.run}_full.csv"
                raw_est = _load_est_for_window(source_pos, ref_window)
                raw_metrics = _metric_row(raw_est, ref_window)
                n_pos = _rewrite_pos_window(source_pos, cand_pos, start_tow, end_tow, height_m)
                n_csv = _copy_csv_window(source_csv, cand_csv, start_tow, end_tow)
                tdcp_est = _load_est_for_window(cand_pos, ref_window)
                tdcp_metrics = _metric_row(tdcp_est, ref_window)
                out = {
                    "city": args.city,
                    "run": args.run,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_tow": f"{start_tow:.1f}",
                    "end_tow": f"{end_tow:.1f}",
                    "icb_l1": l1,
                    "icb_l2": l2,
                    "tag": tag,
                    "candidate_label": f"xd_{cand_dir_name}",
                    "candidate_dir": cand_dir_name,
                    "height_prior_m": f"{height_m:.4f}",
                    "tdcp_pairs_accepted": accepted,
                    "pos_rows": n_pos,
                    "csv_rows": n_csv,
                    "source_pos": source_pos,
                    "source_csv": source_csv,
                    "candidate_pos": cand_pos,
                    "candidate_csv": cand_csv,
                }
                for prefix, metrics in (("raw", raw_metrics), ("tdcp", tdcp_metrics)):
                    for key, value in metrics.items():
                        out[f"{prefix}_{key}"] = value
                rows.append(out)
                print(
                    f"{start_idx}-{end_idx} {tag} "
                    f"raw_p50={raw_metrics['p50_3d_m']:.3f} "
                    f"tdcp_p50={tdcp_metrics['p50_3d_m']:.3f} "
                    f"tdcp_pass={tdcp_metrics['pass_m']:.3f}",
                    flush=True,
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"saved {args.out_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
