"""Calibrate AR quality certificate thresholds from WP12c telemetry."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "results/wp12c/probe_schur_fixed_4000_telemetry.csv"
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def seg(lo: int, hi: int) -> tuple[int, float, float, float, int]:
    s = [r for r in rows if lo <= int(r["epoch"]) <= hi]
    pr = [float(r["dd_pr_rms_raw_m"]) for r in s if float(r["dd_pr_rms_raw_m"]) < 1e6]
    pe = [float(r["pos_err_m"]) for r in s]
    rec = sum(1 for r in s if r.get("recovery_fired", "").lower() == "true")
    pos_rms = math.sqrt(sum(e * e for e in pe) / len(pe)) if pe else float("nan")
    pr_mean = sum(pr) / len(pr) if pr else float("nan")
    pr_max = max(pr) if pr else float("nan")
    return len(s), pos_rms, pr_mean, pr_max, rec


for label, lo, hi in [
    ("open500-799", 500, 799),
    ("drift1000-1499", 1000, 1499),
    ("anchor0-199", 0, 199),
]:
    n, pos_rms, pr_mean, pr_max, rec = seg(lo, hi)
    print(f"{label}: n={n} pos_rms={pos_rms:.3f} dd_pr_mean={pr_mean:.3f} dd_pr_max={pr_max:.3f} rec={rec}")

n = len(rows)
dd2 = sum(1 for r in rows if float(r["dd_pr_rms_raw_m"]) < 2.0)
pos05 = sum(1 for r in rows if float(r["pos_err_m"]) < 0.5)
both = sum(
    1
    for r in rows
    if float(r["dd_pr_rms_raw_m"]) < 2.0 and float(r["pos_err_m"]) < 1.0
)
print(f"dd_pr<2: {dd2}/{n}")
print(f"pos<0.5m: {pos05}/{n}")
print(f"dd_pr<2 & pos<1m: {both}/{n}")
