#!/usr/bin/env python3
"""Diagnose robust_spp on PPC raw PR — verify SPP works before Hatch smoothing."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
sys.path.insert(0, str(REPO / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.robust_spp import robust_spp  # noqa: E402


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="tokyo")
    p.add_argument("--run", default="run1")
    p.add_argument("--systems", default="G,E,J")
    p.add_argument("--n-test", type=int, default=10)
    args = p.parse_args()

    run_dir = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data") / args.city / args.run
    print(f"Loading {run_dir} ...", flush=True)
    loader = PPCDatasetLoader(run_dir)
    systems = tuple(s.strip() for s in args.systems.split(","))
    data = loader.load_experiment_data(systems=systems)
    print(f"  n_epochs={data['n_epochs']}, origin={data['origin_ecef']}", flush=True)

    init_pos = np.asarray(data["origin_ecef"], dtype=np.float64).copy()
    n_pass = 0
    for ep_idx in range(args.n_test):
        sat = data["sat_ecef"][ep_idx]
        pr = data["pseudoranges"][ep_idx]
        w = data["weights"][ep_idx]
        sat_ids = data["used_prns"][ep_idx]
        print(f"\nepoch {ep_idx}: n_sat={len(pr)}, prn_systems={set(s[0] for s in sat_ids)}", flush=True)
        print(f"  pr range: [{pr.min():.1f}, {pr.max():.1f}] m", flush=True)
        print(f"  weights range: [{w.min():.1f}, {w.max():.1f}]", flush=True)

        # SPP with proper init
        pos = robust_spp(sat, pr, weights=w, init_pos=init_pos, weight_func="cauchy", threshold=15.0)
        if pos is None:
            print("  ! robust_spp returned None (init=origin)", flush=True)
        else:
            err = float(np.linalg.norm(pos - data["ground_truth"][ep_idx]))
            print(f"  pos={pos}, gt={data['ground_truth'][ep_idx]}, err={err:.3f} m", flush=True)
            n_pass += 1

        # SPP without init
        pos2 = robust_spp(sat, pr, weights=w, init_pos=None, weight_func="cauchy", threshold=15.0)
        if pos2 is None:
            print("  ! robust_spp returned None (init=None)", flush=True)
        else:
            err = float(np.linalg.norm(pos2 - data["ground_truth"][ep_idx]))
            print(f"  pos(no init)={pos2}, err={err:.3f} m", flush=True)

    print(f"\n=== {n_pass}/{args.n_test} converged with origin init ===", flush=True)


if __name__ == "__main__":
    main()
