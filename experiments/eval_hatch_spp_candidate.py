#!/usr/bin/env python3
"""Compare Hatch A-lite SPP candidate vs reference + hybrid_v5 baseline.

Reports per-run median/mean 3D error in meters and 0.5m pass rate.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
sys.path.insert(0, str(REPO / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def load_pos(path: Path):
    """Read libgnss++ .pos file → dict[tow -> (x,y,z,status)]."""
    out: dict[float, tuple[float, float, float, int]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            tow = float(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            status = int(float(parts[8]))
            out[round(tow, 3)] = (x, y, z, status)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["tokyo", "nagoya"])
    parser.add_argument("--run", required=True, choices=["run1", "run2", "run3"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"),
    )
    parser.add_argument(
        "--hatch-pos",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--hybrid-pos",
        type=Path,
        default=REPO / "experiments/results/libgnss_rtk_pos_v5",
    )
    args = parser.parse_args()

    run_dir = args.data_root / args.city / args.run
    loader = PPCDatasetLoader(run_dir)
    times, gt_ecef = loader.load_ground_truth()
    print(f"Reference: {len(times)} epochs", flush=True)

    # build tow→gt
    gt_by_tow: dict[float, np.ndarray] = {}
    for t, g in zip(times, gt_ecef):
        gt_by_tow[round(float(t), 3)] = g

    hatch = load_pos(args.hatch_pos)
    hybrid_path = args.hybrid_pos / f"{args.city}_{args.run}_full.pos"
    hybrid = load_pos(hybrid_path)
    print(f"Hatch: {len(hatch)} positions", flush=True)
    print(f"Hybrid_v5: {len(hybrid)} positions", flush=True)

    hatch_errs = []
    hybrid_errs = []
    for tow, gt in gt_by_tow.items():
        if tow in hatch:
            p = np.array(hatch[tow][:3])
            hatch_errs.append(float(np.linalg.norm(p - gt)))
        if tow in hybrid:
            p = np.array(hybrid[tow][:3])
            hybrid_errs.append(float(np.linalg.norm(p - gt)))

    hatch_errs = np.array(hatch_errs)
    hybrid_errs = np.array(hybrid_errs)
    print(f"\n=== {args.city}/{args.run} 3D error ===", flush=True)
    for name, errs in [("hatch_alite", hatch_errs), ("hybrid_v5", hybrid_errs)]:
        if errs.size == 0:
            print(f"  {name}: NO DATA", flush=True)
            continue
        print(
            f"  {name}: n={errs.size}, median={np.median(errs):.3f} m, "
            f"mean={np.mean(errs):.3f} m, p95={np.percentile(errs, 95):.3f} m, "
            f"pass<0.5m={np.mean(errs < 0.5)*100:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
