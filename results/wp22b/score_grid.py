"""WP22b: score every ablation-grid .pos output and merge with health-stat CSVs.

Not a permanent pipeline module -- a one-off aggregation script for the WP22b
report table. Reads the .pos files under results/wp22b/pos/<config>_<arm>/ and
the per-run health-stat CSVs under experiments/results/wp22b_<config>_<arm>_runs.csv,
scores each .pos with score_vs_inuex35, and writes a single combined CSV.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POS_ROOT = REPO / "results" / "wp22b" / "pos"
CSV_ROOT = REPO / "experiments" / "results"
OUT_CSV = REPO / "results" / "wp22b" / "csv" / "wp22b_grid_scored.csv"

CONFIGS = ["baseline", "temper", "gmm", "nlos", "allon"]
ARMS = ["off", "preint"]
RUNS = ["run1", "run2", "run3"]

HEALTH_COLS = [
    "mean_ess_ratio", "resample_rate", "ms_per_epoch" if False else None,
    "epoch_tempering", "epoch_tempering_mean_alpha", "epoch_tempering_mean_post_ratio",
    "cn0_gmm", "cn0_gmm_mean_w_los",
    "particle_nlos",
]
HEALTH_COLS = [c for c in HEALTH_COLS if c]


def load_health_row(config: str, arm: str, run: str) -> dict:
    csv_path = CSV_ROOT / f"wp22b_{config}_{arm}_runs.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        rid = str(row.get("run_id", row.get("run", "")))
        if run in rid:
            return row
    # Fallback: positional match against RUNS order if run_id column absent
    idx = RUNS.index(run)
    if idx < len(rows):
        return rows[idx]
    return {}


def score_pos(pos_path: Path, run: str) -> dict:
    out_json = pos_path.with_suffix(".score.json")
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "score_vs_inuex35.py"),
        "--traj", str(pos_path),
        "--city", "tokyo",
        "--run", run,
        "--format", "pos",
        "--fix-statuses", "1",
        "--out-json", str(out_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO), capture_output=True, text=True)
    import json
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for config in CONFIGS:
        for arm in ARMS:
            for run in RUNS:
                pos_path = POS_ROOT / f"{config}_{arm}" / f"tokyo_{run}_RBPF-velKF+DD+gate.pos"
                if not pos_path.exists():
                    print(f"MISSING: {pos_path}", file=sys.stderr)
                    continue
                score = score_pos(pos_path, run)
                health = load_health_row(config, arm, run)
                row = {
                    "config": config,
                    "imu": arm,
                    "run": run,
                    "AllRMS": score.get("all_rms_m", score.get("AllRMS")),
                    "lt50cm_pct": score.get("lt50cm_full_pct", score.get("<50cm_full%")),
                    "ppc_official_pct": score.get("ppc_official_pct", score.get("ppc_official%")),
                }
                for c in HEALTH_COLS:
                    row[c] = health.get(c, "")
                out_rows.append(row)
                print(row)

    fieldnames = ["config", "imu", "run", "AllRMS", "lt50cm_pct", "ppc_official_pct"] + HEALTH_COLS
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
