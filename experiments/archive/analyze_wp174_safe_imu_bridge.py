#!/usr/bin/env python3
"""Audit a precomputed, causal WP174 GNSS/IMU bridge against reference truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _quantile(values: pd.Series, fraction: float) -> float | None:
    finite = values[pd.notna(values)]
    return None if finite.empty else float(finite.quantile(fraction))


def analyze(bridge_path: Path, reference_path: Path, domain: str) -> dict:
    bridge = pd.read_csv(bridge_path)
    reference = pd.read_csv(reference_path, skipinitialspace=True)
    reference.columns = [column.strip() for column in reference.columns]
    reference = reference.rename(
        columns={
            "GPS TOW (s)": "tow",
            "ECEF X (m)": "truth_x",
            "ECEF Y (m)": "truth_y",
            "ECEF Z (m)": "truth_z",
        }
    )
    joined = bridge.merge(
        reference[["tow", "truth_x", "truth_y", "truth_z"]],
        on="tow",
        how="inner",
        validate="one_to_one",
    )
    joined["error_m"] = (
        (joined["bridge_ecef_x"] - joined["truth_x"]) ** 2
        + (joined["bridge_ecef_y"] - joined["truth_y"]) ** 2
        + (joined["bridge_ecef_z"] - joined["truth_z"]) ** 2
    ) ** 0.5

    windows = {}
    for maximum_age_s in (0.2, 0.4, 1.0, 2.0, 5.0, 10.0):
        selected = joined[
            (joined["anchor"] == 0)
            & (joined["anchor_age_s"] > 0.0)
            & (joined["anchor_age_s"] <= maximum_age_s)
            & joined["error_m"].map(math.isfinite)
        ]
        below = int((selected["error_m"] < 0.5).sum())
        windows[str(maximum_age_s)] = {
            "epochs": int(len(selected)),
            "sub50cm_epochs": below,
            "sub50cm_rate_pct": (
                100.0 * below / len(selected) if len(selected) else None
            ),
            "false_fix_if_declared_epochs": int(
                (selected["error_m"] >= 0.5).sum()
            ),
            "error_p50_m": _quantile(selected["error_m"], 0.5),
            "error_p95_m": _quantile(selected["error_m"], 0.95),
            "error_max_m": (
                float(selected["error_m"].max()) if len(selected) else None
            ),
        }
    return {
        "domain": domain,
        "matched_epochs": int(len(joined)),
        "anchor_epochs": int((joined["anchor"] == 1).sum()),
        "bridge_windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime-ms-per-epoch", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.bridge, args.reference, args.domain)
    result["runtime_ms_per_epoch"] = args.runtime_ms_per_epoch
    result["runtime_target_pass"] = args.runtime_ms_per_epoch <= 100.0
    result["promotion_ready"] = False
    result["conclusion"] = (
        "runtime passes, but every tested non-anchor age window contains "
        "errors >=50 cm; IMU bridge must not declare FIX"
    )
    payload = {
        "schema": "gnss_gpu_wp174_safe_imu_bridge_audit_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
