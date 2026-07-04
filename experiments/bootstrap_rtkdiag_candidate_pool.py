#!/usr/bin/env python3
"""Bootstrap a small libgnss RTK candidate pool + per-run manifests for Wave 2 ranker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNSS_SOLVE = PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "apps" / "gnss_solve"
DEFAULT_DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")
WAVE2_ROOT = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_wave2"
MANIFEST_DIR = PROJECT_ROOT / "experiments" / "results" / "rtkdiag_manifest"

TOKYO_BASE = [
    "--preset",
    "low-cost",
    "--arfilter",
    "--arfilter-margin",
    "0.35",
]
NAGOYA_BASE = ["--preset", "low-cost"]

# (label, extra args appended after city base)
WAVE2_VARIANTS: tuple[tuple[str, list[str]], ...] = (
    ("w2_def", []),
    ("w2_hold5", ["--min-hold-count", "5"]),
    ("w2_hold7", ["--min-hold-count", "7"]),
    ("w2_hold10", ["--min-hold-count", "10"]),
    ("w2_ratio20", ["--hold-ratio-threshold", "2.0"]),
    ("w2_ratio30", ["--hold-ratio-threshold", "3.0"]),
)

FULL_RUNS = (
    "tokyo/run1",
    "tokyo/run2",
    "tokyo/run3",
    "nagoya/run1",
    "nagoya/run2",
    "nagoya/run3",
)


def _to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def _city_profile(city: str, extra: list[str]) -> list[str]:
    if city == "tokyo":
        base = TOKYO_BASE + ["--min-hold-count", "8", "--hold-ratio-threshold", "2.6"]
    else:
        base = NAGOYA_BASE + ["--min-hold-count", "7", "--hold-ratio-threshold", "2.4"]
    # extra overrides: gnss_solve uses last flag wins for duplicates in simple argv
    merged = list(base)
    for i in range(0, len(extra), 2):
        flag = extra[i]
        val = extra[i + 1] if i + 1 < len(extra) else ""
        if flag in merged:
            idx = merged.index(flag)
            if idx + 1 < len(merged):
                merged[idx + 1] = val
            continue
        merged.extend([flag, val] if val else [flag])
    return merged


def _parse_runs(text: str) -> tuple[str, ...]:
    if str(text).strip().lower() in {"", "all"}:
        return FULL_RUNS
    return tuple(r.strip().strip("/") for r in str(text).split(",") if r.strip())


def _solve_one(
    *,
    city: str,
    run_name: str,
    label: str,
    extra: list[str],
    data_root: Path,
    force: bool,
) -> Path:
    out_dir = WAVE2_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pos = out_dir / f"{city}_{run_name}_full.pos"
    out_csv = out_dir / f"{city}_{run_name}_full.csv"
    if out_pos.is_file() and out_csv.is_file() and not force:
        print(f"[wave2] reuse {label} {city}/{run_name}", flush=True)
        return out_dir

    run_dir = data_root / city / run_name
    profile = _city_profile(city, extra)
    cmd = [
        "wsl",
        _to_wsl(GNSS_SOLVE),
        "--rover",
        _to_wsl(run_dir / "rover.obs"),
        "--base",
        _to_wsl(run_dir / "base.obs"),
        "--nav",
        _to_wsl(run_dir / "base.nav"),
        "--skip-epochs",
        "0",
        "--out",
        _to_wsl(out_pos),
        "--diagnostics-csv",
        _to_wsl(out_csv),
        "--no-kml",
        *profile,
    ]
    print(f"[wave2] {label} {city}/{run_name}", flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def _write_manifest(city: str, run_name: str, dirs: list[Path], labels: list[str]) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{city}_{run_name}"
    rel_dirs = [
        str(d.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/") for d in dirs
    ]
    payload = {
        "run": f"{city}/{run_name}",
        "dirs": rel_dirs,
        "labels": labels,
    }
    out_json = MANIFEST_DIR / f"{key}.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / f"{key}_dirs.txt").write_text(",".join(rel_dirs) + "\n", encoding="utf-8")
    (MANIFEST_DIR / f"{key}_labels.txt").write_text(",".join(labels) + "\n", encoding="utf-8")
    return out_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not GNSS_SOLVE.is_file():
        print(f"[error] missing gnss_solve: {GNSS_SOLVE}", file=sys.stderr)
        return 2

    for run in _parse_runs(args.runs):
        city, run_name = run.split("/", 1)
        dirs: list[Path] = []
        labels: list[str] = []
        for label, extra in WAVE2_VARIANTS:
            out_dir = _solve_one(
                city=city,
                run_name=run_name,
                label=label,
                extra=list(extra),
                data_root=args.data_root.resolve(),
                force=bool(args.force),
            )
            dirs.append(out_dir)
            labels.append(label)
        manifest = _write_manifest(city, run_name, dirs, labels)
        print(f"[wave2] manifest {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
