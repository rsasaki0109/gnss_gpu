#!/usr/bin/env python3
"""Fill ``Base1`` in GSDC2023 ``settings_*.csv`` using a coarse geographic heuristic.

Official taroz ``dataset_2023.zip`` (filled settings) is often unavailable (404); this
script assigns a **4-character base id** so filenames match ``preprocessing.m``::

  filernx = path + setting.Base1 + "_rnx3.obs"

Heuristic (documented, **not** bit-identical to taroz rows):

- Course path segment ``us-ca-lax`` (Los Angeles drives) → ``VDCY``
- All other courses in this challenge snapshot → ``SLAC``

Rationale: public writeups for GSDC 2023 reference **SLAC** and **VDCY** as the two
CORS-style bases used with this dataset. Verify against your own ``*_rnx3.obs`` files
before trusting for MATLAB parity.

Use ``merge_gsdc_settings_base1.py`` when you have a real reference ``settings_*.csv``.

Example::

  PYTHONPATH=python python3 experiments/apply_gsdc2023_base1_heuristic.py \\
    --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 --dry-run
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


def base1_for_course(course: str, *, lax_markers: tuple[str, ...] = ("-us-ca-lax-",)) -> str:
    c = course.lower()
    for m in lax_markers:
        if m in c:
            return "VDCY"
    return "SLAC"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--splits", nargs="*", default=["train", "test"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    data_root = args.data_root.resolve()

    for split in args.splits:
        path = data_root / f"settings_{split}.csv"
        if not path.is_file():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        if "Course" not in df.columns or "Base1" not in df.columns:
            raise ValueError(f"{path} needs Course and Base1 columns")
        assigned = df["Course"].astype(str).map(base1_for_course)
        counts = assigned.value_counts().to_dict()
        if not args.dry_run:
            df["Base1"] = assigned
        print(f"{split}: Base1 assignment counts {counts} dry_run={args.dry_run}")
        if not args.dry_run:
            backup = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup)
            df.to_csv(path, index=False)
            print(f"  wrote {path} backup {backup}")


if __name__ == "__main__":
    main()
