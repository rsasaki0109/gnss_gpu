#!/usr/bin/env python3
"""Locate PPC epoch windows where libgnss hybrid coverage is low.

Uses the per-epoch NLOS mask CSV for the epoch_idx -> tow mapping (full run)
and the libgnss .pos file for hybrid availability (|tow - pos_tow| <= max_dt).
Reports contiguous hybrid gaps and the best fixed-length smoke windows ranked
by number of hybrid-missing epochs, so mask A/B can target segments where the
PF actually determines the output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DEFAULT_MASK_DIR = RESULTS_DIR / "plateau_nlos_phase33"
DEFAULT_POS_DIR = RESULTS_DIR / "libgnss_rtk_pos_v5"


def _load_epoch_tows(mask_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique (epoch_idx, tow) pairs from the mask CSV."""
    tow_by_epoch: dict[int, float] = {}
    with mask_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch_idx"])
            if epoch not in tow_by_epoch:
                tow_by_epoch[epoch] = float(row["tow"])
    epochs = np.asarray(sorted(tow_by_epoch), dtype=np.int64)
    tows = np.asarray([tow_by_epoch[int(e)] for e in epochs], dtype=np.float64)
    return epochs, tows


def _load_pos_tows(pos_path: Path, min_status: int) -> np.ndarray:
    tows: list[float] = []
    with pos_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                tow = float(parts[1])
                status = int(float(parts[8]))
            except ValueError:
                continue
            if status >= min_status or min_status <= 0:
                tows.append(tow)
    return np.asarray(sorted(tows), dtype=np.float64)


def _hybrid_available(tows: np.ndarray, pos_tows: np.ndarray, max_dt: float) -> np.ndarray:
    idx = np.searchsorted(pos_tows, tows)
    best = np.full(tows.shape, np.inf)
    left_ok = idx > 0
    best[left_ok] = np.abs(tows[left_ok] - pos_tows[idx[left_ok] - 1])
    right_ok = idx < len(pos_tows)
    best[right_ok] = np.minimum(
        best[right_ok], np.abs(tows[right_ok] - pos_tows[idx[right_ok]])
    )
    return best <= max_dt


def _gap_segments(epochs: np.ndarray, available: np.ndarray) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    start: int | None = None
    for i, ok in enumerate(available):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            segments.append(
                {
                    "start_epoch": int(epochs[start]),
                    "end_epoch": int(epochs[i - 1]),
                    "length": int(i - start),
                }
            )
            start = None
    if start is not None:
        segments.append(
            {
                "start_epoch": int(epochs[start]),
                "end_epoch": int(epochs[-1]),
                "length": int(len(available) - start),
            }
        )
    return segments


def _best_windows(
    epochs: np.ndarray,
    available: np.ndarray,
    window: int,
    top: int,
) -> list[dict[str, object]]:
    missing = (~available).astype(np.int64)
    if len(missing) < window:
        window = len(missing)
    csum = np.concatenate([[0], np.cumsum(missing)])
    counts = csum[window:] - csum[:-window]
    order = np.argsort(counts)[::-1]
    picked: list[dict[str, object]] = []
    used: list[int] = []
    for i in order:
        if len(picked) >= top:
            break
        # keep windows apart so we don't report near-duplicates
        if any(abs(int(i) - u) < window // 2 for u in used):
            continue
        used.append(int(i))
        picked.append(
            {
                "start_epoch": int(epochs[i]),
                "window": int(window),
                "hybrid_missing": int(counts[i]),
                "hybrid_missing_pct": round(100.0 * counts[i] / window, 2),
            }
        )
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--mask-csv", type=Path, default=None)
    parser.add_argument("--pos-file", type=Path, default=None)
    parser.add_argument("--max-dt", type=float, default=0.5)
    parser.add_argument(
        "--min-status",
        type=int,
        default=0,
        help="Only count pos rows with Status >= this (0 = all rows)",
    )
    parser.add_argument("--window", type=int, default=1200)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    run = str(args.run).strip().strip("/")
    city, run_name = run.split("/", 1)
    mask_csv = args.mask_csv or DEFAULT_MASK_DIR / f"{city}_{run_name}_per_epoch_nlos.csv"
    pos_file = args.pos_file or DEFAULT_POS_DIR / f"{city}_{run_name}_full.pos"

    epochs, tows = _load_epoch_tows(mask_csv)
    pos_tows = _load_pos_tows(pos_file, int(args.min_status))
    available = _hybrid_available(tows, pos_tows, float(args.max_dt))

    segments = _gap_segments(epochs, available)
    segments.sort(key=lambda s: -s["length"])
    summary = {
        "run": run,
        "mask_csv": str(mask_csv),
        "pos_file": str(pos_file),
        "max_dt_s": float(args.max_dt),
        "min_status": int(args.min_status),
        "epochs_total": int(len(epochs)),
        "hybrid_available": int(available.sum()),
        "hybrid_missing": int((~available).sum()),
        "hybrid_coverage_pct": round(100.0 * available.sum() / len(available), 2),
        "longest_gaps": segments[:10],
        "best_windows": _best_windows(epochs, available, int(args.window), int(args.top)),
    }
    out_json = RESULTS_DIR / f"pf_nlos_hybrid_gaps_{city}_{run_name}.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
