"""Analyze WP12d fix truth-error distribution from telemetry CSV."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def analyze(path: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    fixed = [r for r in rows if r.get("epoch_fixed", "").lower() == "true"]
    cert_pass = [r for r in rows if r.get("ar_cert_passed", "").lower() == "true"]
    offered = [r for r in rows if r.get("ar_offered", "").lower() == "true"]
    errs = [float(r["fix_truth_err_m"]) for r in fixed if r.get("fix_truth_err_m", "nan") not in ("", "nan")]
    print(f"\n=== {path.name} ===")
    print(f"epochs={len(rows)} cert_pass={len(cert_pass)} offered={len(offered)} fixed={len(fixed)}")
    if errs:
        arr = sorted(errs)
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"fix truth: n={len(errs)} RMS={rms:.3f} m median={arr[len(arr)//2]:.3f} p90={arr[int(0.9*len(arr))]:.3f} max={max(errs):.3f}")
        lt50 = sum(1 for e in errs if e < 0.5)
        lt80 = sum(1 for e in errs if e < 0.8)
        lt1 = sum(1 for e in errs if e < 1.0)
        print(f"  <0.5m={100*lt50/len(errs):.1f}% <0.8m={100*lt80/len(errs):.1f}% <1.0m={100*lt1/len(errs):.1f}%")
    open_fixed = [
        float(r["fix_truth_err_m"])
        for r in fixed
        if 500 <= int(r["epoch"]) <= 799 and r.get("fix_truth_err_m", "nan") not in ("", "nan")
    ]
    if open_fixed:
        rms = math.sqrt(sum(e * e for e in open_fixed) / len(open_fixed))
        print(f"open-sky fixed ep500-799: n={len(open_fixed)} RMS={rms:.3f} m")


def main() -> int:
    for name in sys.argv[1:] or [
        "probe_cert_subset_4000_telemetry.csv",
        "probe_cert_lambda_4000_telemetry.csv",
        "probe_full_stack_4000_telemetry.csv",
        "probe_naive_lambda_4000_telemetry.csv",
    ]:
        p = ROOT / "results/wp12d" / name
        if p.exists():
            analyze(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
