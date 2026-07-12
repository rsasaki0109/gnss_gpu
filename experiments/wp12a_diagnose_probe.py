#!/usr/bin/env python3
"""Summarize WP12a per-epoch telemetry around divergence onset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def load_telemetry(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize_block(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"label": label, "n": 0}
    pos_err = np.asarray([_f(r, "pos_err_m") for r in rows], dtype=np.float64)
    raw_rms = np.asarray([_f(r, "dd_pr_rms_raw_m") for r in rows], dtype=np.float64)
    huber_rms = np.asarray([_f(r, "dd_pr_rms_huber_m") for r in rows], dtype=np.float64)
    n_dd = np.asarray([_f(r, "n_dd_factors", 0.0) for r in rows], dtype=np.float64)
    gnss = sum(1 for r in rows if str(r.get("gnss_solved", "")).lower() in ("true", "1"))
    recovery = sum(1 for r in rows if str(r.get("recovery_fired", "")).lower() in ("true", "1"))
    finite_err = pos_err[np.isfinite(pos_err)]
    return {
        "label": label,
        "n": len(rows),
        "pos_err_mean_m": float(np.mean(finite_err)) if finite_err.size else float("nan"),
        "pos_err_max_m": float(np.max(finite_err)) if finite_err.size else float("nan"),
        "dd_pr_rms_raw_mean_m": float(np.nanmean(raw_rms[np.isfinite(raw_rms)])),
        "dd_pr_rms_huber_mean_m": float(np.nanmean(huber_rms[np.isfinite(huber_rms)])),
        "n_dd_factors_mean": float(np.mean(n_dd)),
        "frac_gnss_solved": gnss / len(rows),
        "recovery_events": recovery,
        "frac_lm_converged": sum(
            1 for r in rows if str(r.get("lm_converged", "")).lower() in ("true", "1")
        )
        / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-csv", type=Path, required=True)
    parser.add_argument("--onset-epoch", type=int, default=2842)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_telemetry(args.telemetry_csv)
    onset = int(args.onset_epoch)
    w = int(args.window)
    blocks = {
        "pre_onset": [r for r in rows if int(float(r["epoch"])) < onset - w // 2],
        "onset_window": [
            r for r in rows if onset - w // 2 <= int(float(r["epoch"])) < onset + w // 2
        ],
        "post_onset": [r for r in rows if int(float(r["epoch"])) >= onset + w // 2],
    }
    summary = {k: summarize_block(v, k) for k, v in blocks.items()}

    first_100m = next(
        (r for r in rows if _f(r, "pos_err_m") > 100.0),
        None,
    )
    if first_100m is not None:
        summary["first_err_100m"] = {
            "epoch": int(float(first_100m["epoch"])),
            "tow": _f(first_100m, "tow"),
            "pos_err_m": _f(first_100m, "pos_err_m"),
            "n_dd_factors": _f(first_100m, "n_dd_factors", 0.0),
            "dd_pr_rms_raw_m": _f(first_100m, "dd_pr_rms_raw_m"),
            "dd_pr_rms_huber_m": _f(first_100m, "dd_pr_rms_huber_m"),
        }

    print(json.dumps(summary, indent=2))
    out = args.out_json or args.telemetry_csv.with_suffix(".onset_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
