#!/usr/bin/env python3
"""Generate an L5-enabled RTK candidate (.pos + diagnostics .csv) for the PPC pool.

Runs ``gnss_solve --enable-l5 [--enable-wide-lane-ar]`` for each (city, run)
under ``--data-root`` and emits the matching pair of files
(``{city}_{run}_full.pos`` and ``{city}_{run}_full.csv``) under
``--pos-out-dir`` / ``--diag-out-dir`` so that the PPC selector can pick it up.

Phase 18 Step 7: 6-run smoke verification of L5 N5 filter state path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_BIN = _PROJECT_ROOT / "third_party/gnssplusplus/build/apps/gnss_solve"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

_PROFILES = {
    "tokyo": [
        "--preset", "low-cost",
        "--arfilter",
        "--arfilter-margin", "0.35",
        "--min-hold-count", "8",
        "--hold-ratio-threshold", "2.6",
    ],
    "nagoya": [
        "--preset", "low-cost",
        "--min-hold-count", "7",
        "--hold-ratio-threshold", "2.4",
    ],
}


def _run_one(
    *,
    bin_path: Path,
    data_dir: Path,
    out_pos: Path,
    out_csv: Path,
    profile: list[str],
    enable_wide_lane_ar: bool,
    extra_args: list[str],
    timeout_s: float,
) -> int:
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bin_path),
        "--rover", str(data_dir / "rover.obs"),
        "--base", str(data_dir / "base.obs"),
        "--nav", str(data_dir / "base.nav"),
        "--out", str(out_pos),
        "--diagnostics-csv", str(out_csv),
        "--no-kml",
        "--enable-l5",
        *profile,
    ]
    if enable_wide_lane_ar:
        cmd.append("--enable-wide-lane-ar")
    cmd += extra_args
    completed = subprocess.run(cmd, timeout=timeout_s)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--bin", type=Path, default=_DEFAULT_BIN)
    parser.add_argument("--label", type=str, default="l5")
    parser.add_argument(
        "--pos-out-dir",
        type=Path,
        default=_PROJECT_ROOT / "experiments/results/libgnss_rtk_pos_l5",
    )
    parser.add_argument(
        "--diag-out-dir",
        type=Path,
        default=_PROJECT_ROOT / "experiments/results/libgnss_diag_l5",
    )
    parser.add_argument(
        "--enable-wide-lane-ar",
        action="store_true",
        help="Pass --enable-wide-lane-ar to gnss_solve so L1-L5 widelane AR participates.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Forwarded to gnss_solve (repeat per token).",
    )
    parser.add_argument("--per-run-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--only-runs",
        default="",
        help="Comma list like 'tokyo/run1,nagoya/run3' to limit which runs to process",
    )
    args = parser.parse_args()

    bin_path = args.bin.resolve()
    if not bin_path.exists():
        print(f"ERROR: gnss_solve binary not found: {bin_path}", file=sys.stderr)
        return 2
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    only_runs: set[tuple[str, str]] = set()
    if args.only_runs.strip():
        for token in args.only_runs.split(","):
            token = token.strip()
            if "/" not in token:
                continue
            city, run = token.split("/", 1)
            only_runs.add((city, run))

    print("=" * 72)
    print(f"  Materialize PPC L5 candidate '{args.label}'")
    print("=" * 72)
    print(f"  Binary       : {bin_path}")
    print(f"  Data root    : {data_root}")
    print(f"  Pos out dir  : {args.pos_out_dir}")
    print(f"  Diag out dir : {args.diag_out_dir}")
    print(f"  Wide-lane AR : {args.enable_wide_lane_ar}")
    if args.extra_arg:
        print(f"  Extra args   : {args.extra_arg!r}")
    print(flush=True)

    failures = 0
    for city, run in _RUNS:
        if only_runs and (city, run) not in only_runs:
            continue
        data_dir = data_root / city / run
        if not data_dir.is_dir():
            print(f"  [SKIP] {city}/{run}: not found", flush=True)
            continue
        out_pos = args.pos_out_dir / f"{city}_{run}_full.pos"
        out_csv = args.diag_out_dir / f"{city}_{run}_full.csv"
        print(f"  [RUN ] {city}/{run} -> {out_pos.name}", flush=True)
        rc = _run_one(
            bin_path=bin_path,
            data_dir=data_dir,
            out_pos=out_pos,
            out_csv=out_csv,
            profile=_PROFILES[city],
            enable_wide_lane_ar=args.enable_wide_lane_ar,
            extra_args=args.extra_arg,
            timeout_s=args.per_run_timeout_s,
        )
        if rc != 0:
            print(f"  [FAIL] {city}/{run}: gnss_solve exit {rc}", flush=True)
            failures += 1
        else:
            print(f"  [OK  ] {city}/{run}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
