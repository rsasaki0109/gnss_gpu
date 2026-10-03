#!/usr/bin/env python3
"""PPC ablation of ParticleFilterDevice resampling (internal_docs/decisions.md D-039).

Runs ``exp_ppc_ctrbpf_fgo.main`` once per condition. Each time it swaps the
particle filter's resampler (``_build_pf`` is wrapped; the runner itself is
untouched). It records the honest PPC pass rates at 0.5/1/3 m, plus per-epoch
error statistics from the written .pos files against reference.csv. PF-only
variants rarely reach 3 m on PPC, so the error statistics are what separate
the conditions.

Conditions:
  megopolis_b15           historical default (fixed seed every call)
  megopolis_b15_pfseed43  historical default with PF seed 43: run-to-run noise
  megopolis_b15_seedstep  same, but a fresh seed on every resample
  coalesced_b15           Chesser et al. Megopolis, B=15
  coalesced_b60           Chesser et al. Megopolis, B=60
  coalesced_b60_seedstep  B=60 with a fresh seed on every resample

    PYTHONIOENCODING=utf-8 PYTHONPATH=python python experiments/exp_ppc_resampler_ablation.py \\
        --data-root E:/datasets/PPC-Dataset-data --methods pf+pu,rbpf+pu
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import exp_ppc_ctrbpf_fgo as ctrbpf  # noqa: E402
from ppc_ctrbpf_io import _load_full_reference  # noqa: E402

from gnss_gpu.pf_device_config import clone_pf_device_init_kwargs  # noqa: E402

CONDITIONS = {
    "megopolis_b15": ("megopolis", 15, False),
    "megopolis_b15_pfseed43": ("megopolis", 15, False, 43),
    "megopolis_b15_seedstep": ("megopolis", 15, True),
    "coalesced_b15": ("megopolis_coalesced", 15, False),
    "coalesced_b60": ("megopolis_coalesced", 60, False),
    "coalesced_b60_seedstep": ("megopolis_coalesced", 60, True),
}
METRICS = ("honest_ppc_pct", "honest_ppc_1m_pct", "honest_ppc_3m_pct")


def _patched_build_pf(original, resampling: str, iterations: int, seed_step: bool,
                      pf_seed: int | None = None):
    def build(config):
        pf = original(config)
        if pf_seed is not None:
            # Re-create with another seed; predict noise and resampling both change.
            pf = type(pf)(**(clone_pf_device_init_kwargs(pf) | {"seed": pf_seed}))
        if pf.resampling != "megopolis":
            # FFBSi variants use systematic resampling; leave them untouched.
            return pf
        pf.resampling = resampling
        pf.megopolis_iterations = iterations
        if seed_step:
            counter = itertools.count(1)
            base = int(pf.seed)

            def resample():
                seed = base + next(counter)
                if pf.resampling == "megopolis":
                    pf._pf_device_resample_megopolis(pf._state, pf.megopolis_iterations, seed)
                else:
                    pf._pf_device_resample_megopolis_coalesced(pf._state, pf.megopolis_iterations, seed)

            pf._resample = resample
        return pf

    return build


def run_condition(name: str, passthrough: list[str], pos_dir: Path) -> Path:
    original = ctrbpf._build_pf
    ctrbpf._build_pf = _patched_build_pf(original, *CONDITIONS[name])
    prefix = f"resampler_ablation_{name}"
    argv = sys.argv
    try:
        sys.argv = [
            "exp_ppc_ctrbpf_fgo.py", *passthrough,
            "--results-prefix", prefix, "--pos-dir", str(pos_dir / name),
        ]
        ctrbpf.main()
    finally:
        sys.argv = argv
        ctrbpf._build_pf = original
    return ctrbpf.RESULTS_DIR / f"{prefix}_runs.csv"


def _read_pos(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            parts = line.split()
            out[round(float(parts[1]), 1)] = np.array([float(v) for v in parts[2:5]])
    return out


def _horizontal(err: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lat = np.arctan2(ref[:, 2], np.hypot(ref[:, 0], ref[:, 1]))
    lon = np.arctan2(ref[:, 1], ref[:, 0])
    east = -np.sin(lon) * err[:, 0] + np.cos(lon) * err[:, 1]
    north = (-np.sin(lat) * np.cos(lon) * err[:, 0] - np.sin(lat) * np.sin(lon) * err[:, 1]
             + np.cos(lat) * err[:, 2])
    return np.hypot(east, north)


def error_stats(pos_dir: Path, data_root: Path) -> dict[str, dict[str, float]]:
    """Per-method epoch error statistics, averaged over runs (missing = excluded)."""
    per_method: dict[str, list[dict[str, float]]] = {}
    for pos_path in sorted(pos_dir.glob("*.pos")):
        city, run, method = pos_path.stem.split("_", 2)
        reference = dict(_load_full_reference(data_root / city / run / "reference.csv"))
        est = _read_pos(pos_path)
        tows = [t for t in est if t in reference]
        e = np.array([est[t] for t in tows])
        r = np.array([reference[t] for t in tows])
        h = _horizontal(e - r, r)
        per_method.setdefault(method, []).append({
            "coverage": len(tows) / len(reference),
            "p50_2d_m": float(np.median(h)),
            "p90_2d_m": float(np.percentile(h, 90)),
            "rms_2d_m": float(np.sqrt(np.mean(h**2))),
            "lt1m_2d": float(np.mean(h < 1.0)),
            "lt3m_2d": float(np.mean(h < 3.0)),
            "lt10m_2d": float(np.mean(h < 10.0)),
        })
    return {
        method: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} | {"n_runs": len(rows)}
        for method, rows in per_method.items()
    }


def summarize(csv_path: Path) -> dict[str, dict[str, float]]:
    per_method: dict[str, dict[str, list[float]]] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            bucket = per_method.setdefault(row["method"], {m: [] for m in METRICS})
            for metric in METRICS:
                bucket[metric].append(float(row[metric]))
    # OFFICIAL PPC2024 convention: per-run average.
    return {
        method: {metric: sum(vals) / len(vals) for metric, vals in values.items()}
        | {"n_runs": len(values[METRICS[0]])}
        for method, values in per_method.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPC resampler ablation for ParticleFilterDevice.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--methods", default="pf+pu,rbpf+pu")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pos-dir", type=Path, default=ctrbpf.RESULTS_DIR / "resampler_ablation_pos")
    args, extra = parser.parse_known_args()

    passthrough = ["--data-root", args.data_root, "--methods", args.methods, *extra]
    summary = {}
    for name in args.conditions.split(","):
        csv_path = run_condition(name, passthrough, args.pos_dir)
        summary[name] = {
            "honest": summarize(csv_path),
            "error": error_stats(args.pos_dir / name, Path(args.data_root)),
        }
        print(json.dumps({name: summary[name]}, indent=2))
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
