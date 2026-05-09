"""Output directory and summary helpers for GSDC2023 audit scripts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def unique_output_dir(parent: Path, stem: str, *, create: bool = True) -> Path:
    parent = Path(parent)
    out_dir = parent / stem
    if not out_dir.exists():
        if create:
            out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir
    for idx in range(1, 1000):
        candidate = parent / f"{stem}_{idx:02d}"
        if not candidate.exists():
            if create:
                candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"could not allocate unique output directory under {parent}")


def timestamped_output_dir(parent: Path, prefix: str, *, create: bool = True) -> Path:
    stem = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return unique_output_dir(parent, stem, create=create)


def write_summary_json(out_dir: Path, payload: dict[str, Any], *, filename: str = "summary.json") -> Path:
    path = Path(out_dir) / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_summary_and_output_dir(
    payload: dict[str, Any],
    out_dir: Path,
    *,
    label: str = "comparison_dir",
) -> None:
    print(json.dumps(payload, indent=2))
    print(f"{label}={out_dir}")


__all__ = [
    "print_summary_and_output_dir",
    "timestamped_output_dir",
    "unique_output_dir",
    "write_summary_json",
]
