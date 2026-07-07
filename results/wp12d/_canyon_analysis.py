"""Canyon segment RMS for WP12d run1 (tow 188990-189070)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canyon_rms(pos_path: Path, ref_path: Path, t_min: float, t_max: float) -> dict:
    ref: dict[float, tuple[float, float, float]] = {}
    with ref_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tow = round(float(row["GPS TOW (s)"]), 1)
            ref[tow] = (
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            )

    errs: list[float] = []
    n_fix = 0
    n = 0
    for line in pos_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%") or not line.strip():
            continue
        parts = line.split()
        tow = float(parts[1])
        if tow < t_min or tow > t_max:
            continue
        hit = ref.get(round(tow, 1))
        if hit is None:
            continue
        ecef = (float(parts[2]), float(parts[3]), float(parts[4]))
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ecef, hit)))
        errs.append(err)
        n += 1
        if int(float(parts[8])) == 4:
            n_fix += 1
    rms = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")
    return {
        "t_min": t_min,
        "t_max": t_max,
        "n_epochs": n,
        "n_fix": n_fix,
        "rms_m": rms,
        "median_m": sorted(errs)[len(errs) // 2] if errs else float("nan"),
    }


def main() -> int:
    ref = ROOT / "datasets/PPC-Dataset-data/tokyo/run1/reference.csv"
    pos = ROOT / "results/wp12d/full_run1.pos"
    stats = canyon_rms(pos, ref, 188990.0, 189070.0)
    out = ROOT / "results/wp12d/run1_canyon_188990_189070.json"
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
