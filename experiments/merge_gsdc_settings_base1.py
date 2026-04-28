#!/usr/bin/env python3
"""Merge Base1 (and optionally RINEX) from taroz-style settings into local Kaggle-style CSV.

Kaggle ``settings_*.csv`` often has ``Base1`` empty. The preprocessed bundle from taroz
(``dataset_2023.zip``) is supposed to contain filled values, but the public wget URL has
been returning **404** for some time; see
https://github.com/taroz/gsdc2023/issues/9

If you have an **unzipped** ``dataset_2023`` from any mirror (old local copy, team share,
etc.), pass ``--reference-root`` to that tree. The Kaggle competition download alone does
**not** ship ``brdc.*``, base RINEX, or filled ``Base1`` in ``settings_*.csv`` — issue #9
describes that gap.

Example::

  PYTHONPATH=python python3 experiments/merge_gsdc_settings_base1.py \\
    --target-root ../ref/gsdc2023/dataset_2023 \\
    --reference-root /path/to/unzipped/dataset_2023
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


def merge_file(
    target_path: Path,
    reference_path: Path,
    *,
    dry_run: bool,
    keys: tuple[str, str] = ("Course", "Phone"),
) -> dict[str, int]:
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference settings missing: {reference_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"target settings missing: {target_path}")

    tgt = pd.read_csv(target_path)
    ref = pd.read_csv(reference_path)
    for c in keys:
        if c not in tgt.columns or c not in ref.columns:
            raise ValueError(f"need columns {keys} in both files")

    ref_cols = [c for c in ("Base1", "RINEX") if c in ref.columns]
    if not ref_cols:
        raise ValueError("reference has no Base1/RINEX columns")

    ref_sub = ref[list(keys) + ref_cols].copy()
    rename = {c: f"{c}_taroz" for c in ref_cols}
    ref_sub = ref_sub.rename(columns=rename)
    merged = tgt.merge(ref_sub, on=list(keys), how="left")

    updates = 0
    for c in ref_cols:
        tc = f"{c}_taroz"
        if tc not in merged.columns:
            continue
        if c not in merged.columns:
            merged[c] = merged[tc]
            updates += int(merged[c].notna().sum())
            merged.drop(columns=[tc], inplace=True)
            continue
        merged[c] = merged[c].astype(object)
        mask = merged[c].isna() & merged[tc].notna()
        merged.loc[mask, c] = merged.loc[mask, tc].astype(object)
        updates += int(mask.sum())
        mask2 = merged[c].notna() & merged[tc].notna() & (merged[c].astype(str) != merged[tc].astype(str))
        if mask2.any():
            merged.loc[mask2, c] = merged.loc[mask2, tc].astype(object)
            updates += int(mask2.sum())
        merged.drop(columns=[tc], inplace=True)

    out = merged

    if not dry_run:
        backup = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup)
        out.to_csv(target_path, index=False)

    return {"rows": int(len(out)), "cells_updated": updates}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-root", type=Path, required=True, help="local dataset_2023 directory")
    p.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="taroz dataset_2023 directory with filled settings_*.csv",
    )
    p.add_argument("--splits", nargs="*", default=["train", "test"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.reference_root is None:
        print(
            "No --reference-root. Need a taroz-style dataset_2023 with filled settings_*.csv.\n"
            "  Public wget often 404: https://github.com/taroz/gsdc2023/issues/9\n"
            "  If you obtain dataset_2023.zip from a mirror, unzip and pass:\n"
            "    --reference-root /path/to/dataset_2023\n"
            "  Raw Kaggle data (kaggle competitions download -c smartphone-decimeter-2023)\n"
            "  does not include filled Base1 — use merge only after you have a reference tree.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    ref_root = args.reference_root.resolve()
    tgt_root = args.target_root.resolve()
    total = 0
    for split in args.splits:
        tp = tgt_root / f"settings_{split}.csv"
        rp = ref_root / f"settings_{split}.csv"
        stats = merge_file(tp, rp, dry_run=args.dry_run)
        print(f"{split}: rows={stats['rows']} cells_updated={stats['cells_updated']} dry_run={args.dry_run}")
        total += stats["cells_updated"]
    print(f"total cells_updated={total}")


if __name__ == "__main__":
    main()
