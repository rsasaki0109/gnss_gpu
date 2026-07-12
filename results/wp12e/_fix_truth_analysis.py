"""WP12e cert failure breakdown from telemetry CSV."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def analyze(path: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    fixed = [r for r in rows if r.get("epoch_fixed", "").lower() == "true"]
    cert_pass = [r for r in rows if r.get("ar_cert_passed", "").lower() == "true"]
    offered = [r for r in rows if r.get("ar_offered", "").lower() == "true"]
    print(f"\n=== {path.name} ===")
    print(f"epochs={len(rows)} cert_pass={len(cert_pass)} offered={len(offered)} fixed={len(fixed)}")
    fail_reasons: Counter[str] = Counter()
    for r in rows:
        if r.get("ar_cert_passed", "").lower() == "true":
            continue
        if r.get("ar_offered", "").lower() != "true" and int(r.get("epoch", 0)) < 10:
            continue
        # Infer from marginal sigma / dd pr when cert failed without explicit reasons in CSV
        try:
            sig = float(r.get("ar_cert_marginal_sigma_m", "nan"))
            pr = float(r.get("ar_cert_dd_pr_rms_m", "nan"))
            rec = int(r.get("epochs_since_recovery", 0))
        except ValueError:
            continue
        if sig > 0.15:
            fail_reasons["marginal_sigma"] += 1
        elif pr > 1.0:
            fail_reasons["dd_pr_rms"] += 1
        elif rec < 25:
            fail_reasons["recovery_recency"] += 1
        else:
            fail_reasons["other_or_anchor"] += 1
    print("inferred cert-fail buckets (epochs with AR path active):", dict(fail_reasons))
    drift = [r for r in rows if 1000 <= int(r["epoch"]) <= 3999]
    drift_pass = sum(1 for r in drift if r.get("ar_cert_passed", "").lower() == "true")
    drift_fix = sum(1 for r in drift if r.get("epoch_fixed", "").lower() == "true")
    print(f"drift ep1000-3999: n={len(drift)} cert_pass={drift_pass} fixed={drift_fix}")


def main() -> int:
    for name in sys.argv[1:] or ["probe_cert_dense_hold_4000_telemetry.csv"]:
        p = ROOT / "results/wp12e" / name
        if p.exists():
            analyze(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
