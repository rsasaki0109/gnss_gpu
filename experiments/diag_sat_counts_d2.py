#!/usr/bin/env python3
"""TASK_D D2: multi-GNSS satellite-count audit for PPC tokyo/run1.

Loads the run with systems=("G",) (current default) and systems=("G","R","E","C","J")
(all constellations PPCDatasetLoader supports) and reports:
  - per-constellation observation counts (raw RINEX, before nav/ephemeris filtering)
  - epoch-median / mean usable satellite counts for each systems= setting
  - a histogram of usable sat counts per epoch

This is a *data audit* only (no FGO solve) so it runs in a couple of minutes.

Usage:
    set PYTHONPATH=python
    python experiments/diag_sat_counts_d2.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402

RUN_DIR = Path("E:/datasets/PPC-Dataset-data/tokyo/run1")


def raw_constellation_histogram(run_dir: Path) -> Counter:
    rover_obs = read_rinex_obs(run_dir / "rover.obs")
    counter: Counter = Counter()
    for epoch in rover_obs.epochs:
        for sat_id in epoch.satellites:
            if sat_id:
                counter[sat_id[0]] += 1
    return counter


def audit(run_dir: Path, systems: tuple[str, ...]) -> dict:
    t0 = time.time()
    loader = PPCDatasetLoader(run_dir)
    data = loader.load_experiment_data(systems=systems)
    dt = time.time() - t0
    sat_counts = np.asarray(data["satellite_counts"], dtype=np.int64)
    sys_ids = data.get("system_ids")
    sys_hist: Counter = Counter()
    if sys_ids is not None:
        id_to_char = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}
        for t in range(len(sys_ids)):
            for sid in np.asarray(sys_ids[t]).ravel():
                sys_hist[id_to_char.get(int(sid), "?")] += 1
    return {
        "systems": systems,
        "n_epochs": int(data["n_epochs"]),
        "load_time_s": dt,
        "median_sats": float(np.median(sat_counts)),
        "mean_sats": float(np.mean(sat_counts)),
        "min_sats": int(sat_counts.min()),
        "max_sats": int(sat_counts.max()),
        "used_sys_hist": dict(sys_hist),
        "sat_count_hist": dict(Counter(sat_counts.tolist())),
    }


def main() -> None:
    print(f"Run: {RUN_DIR}\n")

    raw_hist = raw_constellation_histogram(RUN_DIR)
    total_raw = sum(raw_hist.values())
    print("Raw RINEX observation-slot histogram (rover.obs, all epochs, before nav/ephemeris filtering):")
    for sys_char, count in sorted(raw_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {sys_char}: {count:8d}  ({100.0 * count / total_raw:5.1f}%)")
    print()

    for systems in (("G",), ("G", "R", "E", "C", "J")):
        r = audit(RUN_DIR, systems)
        print(f"systems={''.join(systems)}  (loaded in {r['load_time_s']:.1f}s)")
        print(f"  n_epochs={r['n_epochs']}  median_sats={r['median_sats']:.1f}  "
              f"mean_sats={r['mean_sats']:.2f}  min={r['min_sats']} max={r['max_sats']}")
        print(f"  used-satellite constellation histogram: {r['used_sys_hist']}")
        hist = r["sat_count_hist"]
        for k in sorted(hist):
            print(f"    sats={k:2d}: {hist[k]:5d} epochs")
        print()


if __name__ == "__main__":
    main()
