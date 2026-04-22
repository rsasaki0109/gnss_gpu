#!/usr/bin/env python3
"""GNSS: placeholder hook for external factor-graph (FGO) benchmarks.

This repo focuses on GPU PF; full FGO is out of tree. This script:

- Prints how to plug in **gtsam_gnss** / RTKLIB post-processing / PPK so you can
  compare the **same UrbanNav RINEX** against ``experiments/exp_pf_smoother_eval.py``.
- If ``FGO_BENCH_CMD`` is set, runs it via the shell (advanced; you supply the command).

Environment
-----------
``FGO_BENCH_CMD``: optional shell command. Example (pseudo)::

  export FGO_BENCH_CMD='your_runner --obs "$OBS" --nav "$NAV" --out "$OUT"'

The command is executed with cwd = this repo's ``experiments`` directory and these
environment variables set:

- ``URBANNAV_RUN_DIR`` — path to one run (e.g. .../Odaiba)
- ``ROVER_OBS`` — rover RINEX obs used by PF scripts (e.g. rover_trimble.obs)
- ``NAV_FILE`` — e.g. base.nav
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

REFERENCE_URLS = (
    "https://arxiv.org/abs/2502.08158",  # Open-source FGO package for GNSS (overview paper)
    "https://github.com/taroz/gsdc2023",  # GSDC / FGO GNSS+IMU examples (Taro Suzuki)
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="FGO benchmark hook (GNSS, out-of-tree)")
    ap.add_argument(
        "--run-dir",
        type=Path,
        help="UrbanNav run directory (e.g. /tmp/UrbanNav-Tokyo/Odaiba)",
    )
    ap.add_argument(
        "--rover-key",
        type=str,
        default="trimble",
        help="Rover obs stem: rover_{key}.obs",
    )
    args = ap.parse_args()

    print("=== GNSS FGO / batch smooth: out-of-tree benchmark hook ===\n")
    print("PF stack in this repo: experiments/exp_pf_smoother_eval.py, exp_gnss_compare_pf_ffbsi.py\n")
    print("Typical comparison path:")
    print("  1) Export or reuse the same rover.obs + base.nav + GT timeline as PF eval.")
    print("  2) Run your FGO / PPK solver to ECEF or LLH time series.")
    print("  3) Align to UrbanNav ground_truth times and reuse evaluate.compute_metrics.\n")
    print("Reference / starting points (verify freshness before citing):")
    for url in REFERENCE_URLS:
        print(f"  - {url}")
    print()

    cmd = os.environ.get("FGO_BENCH_CMD", "").strip()
    if not args.run_dir:
        if cmd:
            print("Note: --run-dir not passed; env export still available for FGO_BENCH_CMD.")
        else:
            print("Set FGO_BENCH_CMD to run an external solver, or pass --run-dir to show file paths.")
        return

    run_dir = args.run_dir.resolve()
    rover_obs = run_dir / f"rover_{args.rover_key}.obs"
    nav_file = run_dir / "base.nav"
    print(f"run_dir     : {run_dir}")
    print(f"rover obs   : {rover_obs} ('exists'={rover_obs.is_file()})")
    print(f"nav         : {nav_file} ('exists'={nav_file.is_file()})")
    print()

    env = os.environ.copy()
    env["URBANNAV_RUN_DIR"] = str(run_dir)
    env["ROVER_OBS"] = str(rover_obs)
    env["NAV_FILE"] = str(nav_file)

    if not cmd:
        print("FGO_BENCH_CMD unset — not executing anything.")
        print("Example: FGO_BENCH_CMD='echo obs=$ROVER_OBS nav=$NAV_FILE' python3 exp_fgo_benchmark_hook.py --run-dir ...")
        return

    print(f"Running: {cmd}\n")
    sys.stdout.flush()
    r = subprocess.run(cmd, shell=True, cwd=str(_SCRIPT_DIR), env=env)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
