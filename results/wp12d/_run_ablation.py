#!/usr/bin/env python3
"""WP12d ablation runner: score each probe and emit summary JSON."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wp12d"
COMMON = [
    sys.executable,
    str(ROOT / "experiments/wp12_run_tc_fgo.py"),
    "--run",
    "tokyo/run1",
    "--max-epochs",
    "4000",
    "--recovery",
    "--anchor-fix",
    "--dynamic-dd-rebuild",
    "--dd-carrier",
    "--persistent-ambiguities",
    "--schur-marginal",
]

CERT_TIGHT = [
    "--ar-cert-max-pos-sigma",
    "0.15",
    "--ar-cert-max-dd-pr-rms",
    "1.0",
]

STAGES = [
    ("float_no_ar", []),
    (
        "cert_tight",
        [
            "--lambda-ar",
            "--no-ar-subset",
            "--no-ar-ddpr-crossval",
            "--no-ar-post-ar-gate",
            "--no-ar-hold",
        ]
        + CERT_TIGHT,
    ),
    (
        "cert_tight_subset",
        [
            "--lambda-ar",
            "--no-ar-ddpr-crossval",
            "--no-ar-post-ar-gate",
            "--no-ar-hold",
        ]
        + CERT_TIGHT,
    ),
    (
        "cert_tight_hold",
        ["--lambda-ar"] + CERT_TIGHT,
    ),
]


def score_pos(pos_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "experiments/score_vs_inuex35.py"),
        "--traj",
        str(pos_path),
        "--city",
        "tokyo",
        "--run",
        "run1",
        "--format",
        "pos",
        "--out-json",
        str(pos_path.with_suffix(".score.json")),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**dict(**{"PYTHONPATH": "python"})})
    return json.loads(pos_path.with_suffix(".score.json").read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    rows = []
    for name, extra in STAGES:
        pos = OUT / f"probe_{name}_4000.pos"
        telem = OUT / f"probe_{name}_4000_telemetry.csv"
        if force and pos.exists():
            pos.unlink()
            if telem.exists():
                telem.unlink()
        if not pos.exists():
            cmd = COMMON + [
                "--export-pos",
                str(pos),
                "--telemetry-csv",
                str(telem),
            ] + extra
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True, env={"PYTHONPATH": "python"})
        score = score_pos(pos)
        fix_rms = score.get("fix_rms_m")
        rows.append(
            {
                "stage": name,
                "all_rms_m": score.get("all_rms_m"),
                "fix_rms_m": fix_rms,
                "fix_pct": score.get("fix_pct"),
                "lt50cm_full_pct": score.get("lt50cm_full_pct"),
                "n_fix": score.get("n_fix"),
                "pos": str(pos.relative_to(ROOT)),
            }
        )
        print(
            f"{name}: AllRMS={score.get('all_rms_m'):.3f} FixRMS={fix_rms} fix%={score.get('fix_pct')} "
            f"<50cm_full%={score.get('lt50cm_full_pct')}",
            flush=True,
        )
    summary = {"probe_epochs": 4000, "rows": rows}
    (OUT / "ablation_4000_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
