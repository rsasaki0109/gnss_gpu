#!/usr/bin/env python3
"""WP12e full-length 3-run parallel launcher."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wp12e"

BASELINE = {
    "run1": ROOT / "results/wp10/sweep/run1/a0_baseline_no_wp10.pos",
    "run2": ROOT / "results/wp10/sweep/run2/b0_baseline_no_wp10.pos",
    "run3": ROOT / "results/wp10/sweep/run3/b0_baseline_no_wp10.pos",
}

COMMON_FLAGS = [
    "--recovery",
    "--anchor-fix",
    "--anchor-float",
    "--dynamic-dd-rebuild",
    "--dd-carrier",
    "--persistent-ambiguities",
    "--schur-marginal",
    "--lambda-ar",
    "--ar-cert-max-pos-sigma",
    "0.15",
    "--ar-cert-max-dd-pr-rms",
    "1.0",
    "--ar-cert-min-epochs-since-recovery",
    "25",
    "--ar-cert-max-epochs-since-anchor",
    "50",
]


def _run_one(run: str) -> dict:
    pos = OUT / f"full_{run}.pos"
    telem = OUT / f"full_{run}_telemetry.csv"
    log = OUT / f"full_{run}.log"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/wp12_run_tc_fgo.py"),
        "--run",
        f"tokyo/{run}",
        "--baseline-pos",
        str(BASELINE[run]),
        "--export-pos",
        str(pos),
        "--telemetry-csv",
        str(telem),
    ] + COMMON_FLAGS
    with log.open("w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env={"PYTHONPATH": "python"}, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"{run} failed rc={proc.returncode}")
    score_cmd = [
        sys.executable,
        str(ROOT / "experiments/score_vs_inuex35.py"),
        "--traj",
        str(pos),
        "--city",
        "tokyo",
        "--run",
        run,
        "--format",
        "pos",
        "--out-json",
        str(pos.with_suffix(".score.json")),
    ]
    subprocess.run(score_cmd, cwd=ROOT, check=True, env={"PYTHONPATH": "python"})
    return json.loads(pos.with_suffix(".score.json").read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_run_one, r): r for r in ("run1", "run2", "run3")}
        for fut in as_completed(futs):
            run = futs[fut]
            results[run] = fut.result()
            print(f"{run} done: <50cm_full%={results[run].get('lt50cm_full_pct')} FixRMS={results[run].get('fix_rms_m')}")
    (OUT / "full_3run_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
