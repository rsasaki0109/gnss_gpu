#!/usr/bin/env python3
"""Build ``settings_train.csv`` / ``settings_test.csv`` from a Kaggle-style ``sdc2023`` tree.

Mirrors ``ref/gsdc2023/ensure_raw_settings.m``: one row per ``train|test/<course>/<phone>``
with ``device_gnss.csv``. ``IdxEnd`` is ``ground_truth.csv`` height on train, or
``sample_submission.csv`` row count per ``tripId`` on test.

Example::

  PYTHONPATH=python python3 experiments/generate_gsdc2023_settings_from_tree.py \\
    --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT

REQUIRED_COLS = ["Course", "Phone", "Type", "L5", "BDS", "RINEX", "Base1", "IdxStart", "IdxEnd", "RPYReset"]


def _sample_counts_by_trip(sample_path: Path) -> dict[str, int]:
    sub = pd.read_csv(sample_path)
    if "tripId" not in sub.columns:
        raise ValueError(f"tripId column missing in {sample_path}")
    return sub.groupby("tripId").size().to_dict()


def _rows_for_split(data_root: Path, split: str, sample_counts: dict[str, int] | None) -> list[dict]:
    out: list[dict] = []
    split_dir = data_root / split
    if not split_dir.is_dir():
        return out
    for course_dir in sorted(split_dir.iterdir()):
        if not course_dir.is_dir() or course_dir.name.startswith("."):
            continue
        course = course_dir.name
        for phone_dir in sorted(course_dir.iterdir()):
            if not phone_dir.is_dir() or phone_dir.name.startswith("."):
                continue
            phone = phone_dir.name
            if not (phone_dir / "device_gnss.csv").is_file():
                continue
            idx_end = 0
            gt_path = phone_dir / "ground_truth.csv"
            if gt_path.is_file():
                gt = pd.read_csv(gt_path)
                idx_end = int(len(gt))
            elif split == "test" and sample_counts is not None:
                tid = f"{course}/{phone}"
                idx_end = int(sample_counts.get(tid, 0))
            out.append(
                {
                    "Course": course,
                    "Phone": phone,
                    "Type": "Street",
                    "L5": 0,
                    "BDS": 0,
                    "RINEX": "V3",
                    "Base1": "",
                    "IdxStart": 1,
                    "IdxEnd": idx_end,
                    "RPYReset": 0,
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--splits", nargs="*", default=["train", "test"])
    args = p.parse_args()
    data_root = args.data_root.resolve()

    sample_path = data_root / "sample_submission.csv"
    sample_counts = _sample_counts_by_trip(sample_path) if sample_path.is_file() else None
    if "test" in args.splits and sample_counts is None:
        print(f"warning: no {sample_path}; test IdxEnd will be 0", file=sys.stderr)

    for split in args.splits:
        rows = _rows_for_split(data_root, split, sample_counts if split == "test" else None)
        if not rows:
            print(f"warning: no trips under {data_root}/{split}", file=sys.stderr)
            continue
        out_path = data_root / f"settings_{split}.csv"
        if out_path.is_file():
            shutil.copy2(out_path, out_path.with_suffix(out_path.suffix + ".bak"))
        df = pd.DataFrame(rows)
        for c in REQUIRED_COLS:
            if c not in df.columns:
                raise RuntimeError(f"internal: missing column {c}")
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
