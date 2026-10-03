#!/usr/bin/env python3
"""UrbanNav PF-smoother ablation of ParticleFilterDevice resampling (D-039).

Runs ``exp_pf_smoother_eval.main`` with a preset once per condition. Each
time it swaps the particle filter's resampler by patching
``gnss_gpu.pf_smoother_runtime.ParticleFilterDevice``; the smoother's backward
filter inherits the setting via ``clone_pf_device_init_kwargs``. It reports the
forward and smoothed 2D P50/P95/RMS that the evaluator writes.

    PYTHONIOENCODING=utf-8 PYTHONPATH="<libgnsspp>;python;experiments" \\
        python experiments/exp_urbannav_resampler_ablation.py \\
        --data-root E:/datasets/urbannav/Tokyo --preset odaiba_stop_detect
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import exp_pf_smoother_eval as smoother_eval  # noqa: E402

from gnss_gpu import pf_smoother_runtime  # noqa: E402

# name -> (resampling, megopolis_iterations, extra evaluator argv)
CONDITIONS = {
    "megopolis_b15": ("megopolis_legacy", 15, []),
    "megopolis_b15_seed43": ("megopolis_legacy", 15, ["--seed", "43"]),
    "coalesced_b15": ("megopolis_coalesced", 15, []),
    "coalesced_b60": ("megopolis_coalesced", 60, []),
}
METRICS = (
    "forward_p50", "forward_p95", "forward_rms_2d",
    "smoothed_p50", "smoothed_p95", "smoothed_rms_2d", "n_epochs", "ms_per_epoch",
)
RESULTS_CSV = smoother_eval.RESULTS_DIR / "pf_smoother_eval.csv"


def _resampler_class(base, resampling: str, iterations: int):
    class PatchedParticleFilterDevice(base):
        def __init__(self, *args, **kwargs):
            kwargs["resampling"] = resampling
            kwargs["megopolis_iterations"] = iterations
            super().__init__(*args, **kwargs)

    return PatchedParticleFilterDevice


def run_condition(name: str, data_root: str, preset: str, seed: int | None = None) -> dict[str, float]:
    resampling, iterations, extra = CONDITIONS[name]
    if seed is not None:
        extra = [*extra, "--seed", str(seed)]
    original = pf_smoother_runtime.ParticleFilterDevice
    pf_smoother_runtime.ParticleFilterDevice = _resampler_class(original, resampling, iterations)
    try:
        code = smoother_eval.main(["--data-root", data_root, "--preset", preset, *extra])
    finally:
        pf_smoother_runtime.ParticleFilterDevice = original
    if code not in (0, None):
        raise RuntimeError(f"{name}: evaluator exited with {code}")
    with RESULTS_CSV.open(newline="") as fh:
        row = list(csv.DictReader(fh))[-1]
    return {key: float(row[key]) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description="UrbanNav PF-smoother resampler ablation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--preset", default="odaiba_stop_detect")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default="",
                        help="Comma-separated PF seeds; each condition runs once per seed")
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(v) for v in args.seeds.split(",") if v] or [None]
    summary: dict[str, dict[str, float]] = {}
    for name in args.conditions.split(","):
        for seed in seeds:
            key = name if seed is None else f"{name}@seed{seed}"
            summary[key] = run_condition(name, args.data_root, args.preset, seed)
            print(json.dumps({key: summary[key]}), flush=True)
        if args.summary_json is not None:
            args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
