#!/usr/bin/env python3
"""Score WP12b ablation .pos files on the run1 4000-epoch probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from score_vs_inuex35 import load_reference_grid, load_trajectory, score_trajectory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/wp12b"))
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run1")
    parser.add_argument("--glob", default="probe_*_4000.pos")
    args = parser.parse_args()

    reference = load_reference_grid(args.city, args.run)
    rows: list[dict] = []
    for pos_path in sorted(args.results_dir.glob(str(args.glob))):
        label = pos_path.stem.replace("probe_", "").replace("_4000", "")
        epochs = load_trajectory(pos_path, fmt="pos")
        score = score_trajectory(
            epochs, reference, city=args.city, run=args.run, traj_path=pos_path, fmt="pos"
        )
        rows.append(
            {
                "config": label,
                "path": str(pos_path),
                "n_scored": score.n_scored,
                "coverage_pct": score.coverage_pct,
                "all_rms_m": score.all_rms_m,
                "fix_rms_m": score.fix_rms_m,
                "fix_pct": score.fix_pct,
                "lt50cm_full_pct": score.lt50cm_full_pct,
            }
        )
        out_json = pos_path.with_suffix(".score.json")
        out_json.write_text(json.dumps(score.to_json_dict(), indent=2), encoding="utf-8")

    print(
        f"{'config':<24} {'coverage%':>10} {'AllRMS':>10} {'FixRMS':>10} "
        f"{'fix%':>8} {'<50cm_full%':>12}"
    )
    for row in rows:
        fix_rms = row["fix_rms_m"]
        fix_rms_s = f"{fix_rms:10.3f}" if fix_rms is not None else f"{'n/a':>10}"
        print(
            f"{row['config']:<24} {row['coverage_pct']:10.1f} "
            f"{row['all_rms_m']:10.2f} {fix_rms_s} "
            f"{row['fix_pct']:8.2f} {row['lt50cm_full_pct']:12.2f}"
        )
    summary_path = args.results_dir / "ablation_4000_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
