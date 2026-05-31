#!/usr/bin/env python3
"""Scan candidate position files for sparse absolute-anchor potential."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exp_ppc_ctrbpf_fgo import _load_full_reference, _load_hybrid_pos_file  # noqa: E402


DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
RESULTS = _SCRIPT_DIR / "results"


def _parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _read_text_csv(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return _parse_csv_text(path.read_text().strip())


def _load_candidates(args: argparse.Namespace) -> list[tuple[str, Path]]:
    labels = _parse_csv_text(args.labels)
    dirs = [Path(x) for x in _parse_csv_text(args.candidate_dirs)]
    for path in args.labels_file:
        labels.extend(_read_text_csv(path))
    for path in args.candidate_dirs_file:
        dirs.extend(Path(x) for x in _read_text_csv(path))
    if len(labels) != len(dirs):
        raise ValueError(f"labels/dirs length mismatch: {len(labels)} vs {len(dirs)}")
    return list(zip(labels, dirs, strict=True))


def _candidate_pos_path(candidate_dir: Path, city: str, run: str) -> Path:
    return candidate_dir / f"{city}_{run}_full.pos"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="nagoya/run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--start-tow", type=float, default=557046.2)
    parser.add_argument("--end-tow", type=float, default=557051.6)
    parser.add_argument("--thresholds-m", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--candidate-dirs", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--candidate-dirs-file", type=Path, action="append", default=[])
    parser.add_argument("--labels-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=RESULTS / "nr2_anchor_candidate_pool_6637_summary.csv",
    )
    parser.add_argument(
        "--out-epochs",
        type=Path,
        default=RESULTS / "nr2_anchor_candidate_pool_6637_epochs.csv",
    )
    args = parser.parse_args()

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    ref = {
        round(float(t), 1): np.asarray(p, dtype=np.float64)
        for t, p in _load_full_reference(run_dir / "reference.csv")
    }
    tows = [
        tow
        for tow in sorted(ref)
        if float(args.start_tow) <= float(tow) <= float(args.end_tow)
    ]
    if not tows:
        raise SystemExit("no reference rows in requested window")
    thresholds = [float(x) for x in _parse_csv_text(args.thresholds_m)]
    candidates = _load_candidates(args)

    summary_rows: list[dict[str, object]] = []
    epoch_best: dict[float, dict[str, object]] = {
        tow: {
            "run": str(args.run),
            "tow": tow,
            "n_candidates_present": 0,
            "best_label": "",
            "best_error_m": float("nan"),
        }
        for tow in tows
    }
    for thr in thresholds:
        for row in epoch_best.values():
            row[f"best_pass_{thr:g}m"] = 0

    for label, cand_dir in candidates:
        pos_path = _candidate_pos_path(cand_dir, city, run)
        row: dict[str, object] = {
            "run": str(args.run),
            "label": label,
            "candidate_dir": str(cand_dir),
            "pos_path": str(pos_path),
            "exists": int(pos_path.is_file()),
            "window_rows": len(tows),
            "present_rows": 0,
            "median_error_m": float("nan"),
            "p90_error_m": float("nan"),
            "min_error_m": float("nan"),
            "max_error_m": float("nan"),
        }
        for thr in thresholds:
            row[f"pass_rows_{thr:g}m"] = 0
            row[f"pass_frac_{thr:g}m"] = 0.0
        if not pos_path.is_file():
            summary_rows.append(row)
            continue

        pos_by_tow, _status = _load_hybrid_pos_file(pos_path)
        errs: list[float] = []
        for tow in tows:
            pos = pos_by_tow.get(tow)
            truth = ref.get(tow)
            if pos is None or truth is None:
                continue
            err = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
            errs.append(err)
            best = epoch_best[tow]
            best["n_candidates_present"] = int(best["n_candidates_present"]) + 1
            cur = best["best_error_m"]
            if not np.isfinite(float(cur)) or err < float(cur):
                best["best_error_m"] = err
                best["best_label"] = label
            for thr in thresholds:
                if err <= thr:
                    best[f"any_pass_{thr:g}m"] = 1
        if errs:
            vals = np.asarray(errs, dtype=np.float64)
            row["present_rows"] = int(vals.size)
            row["median_error_m"] = float(np.median(vals))
            row["p90_error_m"] = float(np.percentile(vals, 90))
            row["min_error_m"] = float(np.min(vals))
            row["max_error_m"] = float(np.max(vals))
            for thr in thresholds:
                n_pass = int(np.count_nonzero(vals <= thr))
                row[f"pass_rows_{thr:g}m"] = n_pass
                row[f"pass_frac_{thr:g}m"] = float(n_pass / max(vals.size, 1))
        summary_rows.append(row)

    for tow, row in epoch_best.items():
        best_err = float(row["best_error_m"])
        for thr in thresholds:
            row[f"best_pass_{thr:g}m"] = int(np.isfinite(best_err) and best_err <= thr)
            row.setdefault(f"any_pass_{thr:g}m", 0)

    oracle_row: dict[str, object] = {
        "run": str(args.run),
        "label": "__oracle_best__",
        "candidate_dir": "",
        "pos_path": "",
        "exists": 1,
        "window_rows": len(tows),
        "present_rows": int(sum(int(r["n_candidates_present"]) > 0 for r in epoch_best.values())),
    }
    best_errs = [
        float(r["best_error_m"])
        for r in epoch_best.values()
        if np.isfinite(float(r["best_error_m"]))
    ]
    if best_errs:
        vals = np.asarray(best_errs, dtype=np.float64)
        oracle_row["median_error_m"] = float(np.median(vals))
        oracle_row["p90_error_m"] = float(np.percentile(vals, 90))
        oracle_row["min_error_m"] = float(np.min(vals))
        oracle_row["max_error_m"] = float(np.max(vals))
        for thr in thresholds:
            n_pass = int(np.count_nonzero(vals <= thr))
            oracle_row[f"pass_rows_{thr:g}m"] = n_pass
            oracle_row[f"pass_frac_{thr:g}m"] = float(n_pass / max(vals.size, 1))
    summary_rows.append(oracle_row)

    summary_rows.sort(
        key=lambda r: (
            -int(r.get("pass_rows_0.5m", r.get("pass_rows_0.5m", 0)) or 0),
            float(r.get("median_error_m", float("inf"))),
            str(r.get("label", "")),
        )
    )
    _write_rows(args.out_summary, summary_rows)
    _write_rows(args.out_epochs, list(epoch_best.values()))

    top = summary_rows[:10]
    print(f"candidates={len(candidates)} rows={len(tows)}")
    for row in top:
        print(
            f"{row['label']}: present={row.get('present_rows')} "
            f"pass0.5={row.get('pass_rows_0.5m', row.get('pass_rows_0.5m'))} "
            f"pass1={row.get('pass_rows_1m')} med={row.get('median_error_m')}"
        )
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_epochs}")


if __name__ == "__main__":
    main()
