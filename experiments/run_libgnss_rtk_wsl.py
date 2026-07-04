#!/usr/bin/env python3
"""Generate libgnss++ RTK .pos for one PPC run via WSL gnss_solve."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNSS_SOLVE = (
    PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "apps" / "gnss_solve"
)
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5"
DEFAULT_DIAG_OUT = PROJECT_ROOT / "experiments" / "results" / "libgnss_rtk_pos_v5_diag"

TOKYO_PROFILE = [
    "--preset",
    "low-cost",
    "--arfilter",
    "--arfilter-margin",
    "0.35",
    "--min-hold-count",
    "8",
    "--hold-ratio-threshold",
    "2.6",
]
NAGOYA_PROFILE = [
    "--preset",
    "low-cost",
    "--min-hold-count",
    "7",
    "--hold-ratio-threshold",
    "2.4",
]


def _to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run1")
    parser.add_argument("--data-root", type=Path, default=Path("E:/datasets/PPC-Dataset-data"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_OUT)
    parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Also write gnss_solve --diagnostics-csv (required for rtkdiag_pf)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not GNSS_SOLVE.is_file():
        print(f"[error] missing WSL gnss_solve build: {GNSS_SOLVE}", file=sys.stderr)
        return 2

    city, run_name = str(args.run).strip("/").split("/", 1)
    run_dir = args.data_root / city / run_name
    for name in ("rover.obs", "base.obs", "base.nav"):
        if not (run_dir / name).is_file():
            print(f"[error] missing {run_dir / name}", file=sys.stderr)
            return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pos = out_dir / f"{city}_{run_name}_full.pos"
    diag_dir = args.diag_dir
    out_csv = diag_dir / f"{city}_{run_name}_full.csv"
    need_pos = args.force or not out_pos.is_file()
    need_diag = args.with_diagnostics and (args.force or not out_csv.is_file())
    if not need_pos and not need_diag:
        print(f"[rtk] reuse {out_pos}", flush=True)
        if args.with_diagnostics:
            print(f"[rtk] reuse {out_csv}", flush=True)
        return 0

    profile = TOKYO_PROFILE if city == "tokyo" else NAGOYA_PROFILE
    if need_diag:
        diag_dir.mkdir(parents=True, exist_ok=True)
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
        "--no-kml",
        *profile,
    ]
    if need_diag:
        cmd.extend(["--diagnostics-csv", _to_wsl(out_csv)])
    print("[rtk] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[rtk] wrote {out_pos}", flush=True)
    if need_diag:
        print(f"[rtk] wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
