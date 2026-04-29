#!/usr/bin/env python3
"""Audit local GSDC2023 dataset assets needed for MATLAB parity."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT, collect_matlab_parity_audit


def _settings_path(data_root: Path, split: str) -> Path:
    return data_root / f"settings_{split}.csv"


def _audit_split(data_root: Path, split: str, *, include_imu_sync: bool = True) -> pd.DataFrame:
    settings_path = _settings_path(data_root, split)
    if not settings_path.is_file():
        return pd.DataFrame()
    settings = pd.read_csv(settings_path)
    rows = []
    for row in settings.itertuples(index=False):
        trip = f"{split}/{row.Course}/{row.Phone}"
        rows.append(collect_matlab_parity_audit(data_root, trip, include_imu_sync=include_imu_sync))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=["train", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip expensive raw GNSS/IMU parsing and audit file availability only",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print why base_correction is blocked and required file layout (after JSON)",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    frames = []
    include_imu_sync = not args.quick
    for split in args.datasets:
        frame = _audit_split(data_root, split, include_imu_sync=include_imu_sync)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit("no settings files found for requested datasets")

    audit = pd.concat(frames, ignore_index=True)
    status_counts = {
        str(key): int(value)
        for key, value in audit["base_correction_status"].value_counts(dropna=False).to_dict().items()
    }
    base1_nonempty = 0
    if "base_name" in audit.columns:
        base1_nonempty = int(audit["base_name"].notna().sum())
    summary = {
        "data_root": str(data_root),
        "quick": bool(args.quick),
        "n_trips": int(len(audit)),
        "base_correction_ready": int(audit["base_correction_ready"].sum()),
        "status_counts": status_counts,
        "settings_base1_nonempty_count": base1_nonempty,
        "device_imu_present": int(audit["device_imu_present"].sum()),
        "imu_sync_ready": int(audit["imu_sync_ready"].sum()),
        "ref_height_present": int(audit["ref_height_present"].sum()),
        "ground_truth_present": int(audit["ground_truth_present"].sum()),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir.resolve() / f"gsdc2023_matlab_parity_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "audit.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"audit_dir={out_dir}")
    if args.explain:
        _print_explain(summary, data_root)


def _print_explain(summary: dict, data_root: Path) -> None:
    n = summary.get("n_trips", 0)
    nz = summary.get("settings_base1_nonempty_count", 0)
    print()
    print("=== parity remediation (read-only hints) ===")
    if nz == 0 and n > 0:
        print(
            "- Base1: no non-empty Base1 in settings CSV rows. "
            "Kaggle settings often ship with Base1 empty; the taroz auxiliary zip "
            "(filled settings, brdc.*, base RINEX) is separate and the public URL "
            "is often 404 — see https://github.com/taroz/gsdc2023/issues/9 "
            "When you have a reference dataset_2023 tree, run "
            "experiments/merge_gsdc_settings_base1.py --reference-root ..."
        )
    print(
        "- Expected layout (see collect_matlab_parity_audit in gsdc2023_raw_bridge.py):",
    )
    print(f"  - {data_root.parent / 'base' / 'base_position.csv'}")
    print(f"  - {data_root.parent / 'base' / 'base_offset.csv'}")
    print(
        f"  - {data_root}/<train|test>/<course>/{'<Base1>_rnx3.obs'}  (when RINEX column is V3); "
        "fetch: experiments/fetch_gsdc2023_base_obs.py (needs: pip install hatanaka)"
    )
    print(
        f"  - broadcast ephemeris: {data_root}/<train|test>/<course>/brdc.* "
        f"(fetch: experiments/fetch_gsdc2023_brdc.py)"
    )
    print(
        "  - base_position.csv / base_offset.csv: under ref/gsdc2023/base/ (default) — "
        "experiments/generate_gsdc2023_base_metadata.py; "
        "Kaggle …/sdc2023 trees fall back to that folder in collect_matlab_parity_audit."
    )
    print(f"  - optional ref height: {data_root}/<train|test>/<course>/ref_hight.mat")
    print("- MATLAB reference: ref/gsdc2023/MATLAB_PARITY_AUDIT.md (workspace-local)")


if __name__ == "__main__":
    main()
