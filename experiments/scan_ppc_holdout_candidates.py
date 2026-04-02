#!/usr/bin/env python3
"""Search nearby holdout candidates around the six tuned positive PPC segments."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate import save_results
from exp_ppc_pf_ablation_sweep import POSITIVE_SEGMENTS, _subset_url_for_city, _run_dir
from exp_urbannav_baseline import load_real_data
from gnss_gpu.io.plateau import load_plateau
from scan_ppc_plateau_segments import (
    RESULTS_DIR,
    compute_nlos_fraction,
    derive_mesh_codes,
    ensure_subset,
)
from fetch_plateau_subset import expand_meshes


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    save_results({key: [row.get(key, np.nan) for row in rows] for key in rows[0]}, path)


def _candidate_starts(anchor: int, n_epochs: int, segment_length: int, radius: int, step: int) -> list[int]:
    low = max(0, anchor - radius)
    high = min(max(0, n_epochs - segment_length), anchor + radius)
    starts = list(range(low, high + 1, max(step, 1)))
    if anchor not in starts and low <= anchor <= high:
        starts.append(anchor)
    return sorted(set(starts))


def _select_specs(raw: str | None):
    if raw is None:
        return POSITIVE_SEGMENTS
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    specs = [spec for spec in POSITIVE_SEGMENTS if f"{spec.city}/{spec.run}" in wanted]
    missing = sorted(wanted - {f"{spec.city}/{spec.run}" for spec in specs})
    if missing:
        raise ValueError(f"unknown segments requested: {', '.join(missing)}")
    return tuple(specs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find nearby PPC holdout candidates around tuned positive segments")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/tmp/PPC-real/PPC-Dataset"),
        help="PPC-Dataset root directory",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("/tmp/plateau_segment_cache"),
        help="PLATEAU subset cache root",
    )
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help="Optional subset like tokyo/run1,nagoya/run2",
    )
    parser.add_argument("--segment-length", type=int, default=100, help="Epochs per candidate segment")
    parser.add_argument("--search-radius", type=int, default=400, help="Epoch radius around tuned anchor")
    parser.add_argument("--step", type=int, default=50, help="Epoch stride within the search window")
    parser.add_argument(
        "--min-offset",
        type=int,
        default=150,
        help="Minimum distance from tuned anchor to count as holdout",
    )
    parser.add_argument("--mesh-radius", type=int, default=1, help="Neighboring mesh expansion radius")
    parser.add_argument("--sample-stride", type=int, default=5, help="Epoch stride for NLOS estimation")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_holdout_candidates",
        help="Output prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    specs = _select_specs(args.segments)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.subset_root.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    print("=" * 72)
    print("  PPC Holdout Candidate Search")
    print("=" * 72)

    for spec in specs:
        run_dir = _run_dir(args.data_root, spec)
        data = load_real_data(run_dir, systems=systems)
        if data is None:
            raise RuntimeError(f"failed to load PPC data: {run_dir}")
        ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
        mesh_codes = derive_mesh_codes(ground_truth)
        sat_ecef = data["sat_ecef"]
        n_epochs = int(data["n_epochs"])
        triangle_cache: dict[str, int] = {}

        starts = _candidate_starts(
            spec.start_epoch,
            n_epochs,
            args.segment_length,
            args.search_radius,
            args.step,
        )
        print(f"\n[{spec.city}/{spec.run}] anchor={spec.start_epoch} candidates={len(starts)}")

        for start in starts:
            stop = min(start + args.segment_length, n_epochs)
            meshes = expand_meshes(sorted(set(mesh_codes[start:stop])), args.mesh_radius)
            subset_dir = ensure_subset(_subset_url_for_city(spec.city), meshes, args.subset_root)
            subset_key = subset_dir.name
            model = load_plateau(subset_dir, zone=spec.plateau_zone)
            triangle_cache[subset_key] = int(model.triangles.shape[0])
            n_nlos, n_total = compute_nlos_fraction(
                model,
                ground_truth,
                sat_ecef,
                start,
                stop,
                max(args.sample_stride, 1),
            )
            frac = float(n_nlos / n_total) if n_total else 0.0
            candidate_rows.append(
                {
                    "city": spec.city,
                    "run": spec.run,
                    "anchor_start_epoch": spec.start_epoch,
                    "start_epoch": start,
                    "offset_epoch": start - spec.start_epoch,
                    "is_anchor": int(start == spec.start_epoch),
                    "is_holdout": int(abs(start - spec.start_epoch) >= args.min_offset),
                    "nlos_fraction": frac,
                    "subset_key": subset_key,
                    "plateau_zone": spec.plateau_zone,
                    "triangle_count": triangle_cache[subset_key],
                    "n_meshes": len(meshes),
                    "n_total_obs": n_total,
                    "n_nlos": n_nlos,
                }
            )
            del model
            gc.collect()

        holdout_rows = [
            row
            for row in candidate_rows
            if row["city"] == spec.city
            and row["run"] == spec.run
            and row["is_holdout"] == 1
        ]
        holdout_rows.sort(
            key=lambda row: (row["nlos_fraction"], abs(int(row["offset_epoch"]))),
            reverse=True,
        )
        if holdout_rows:
            best_rows.append(holdout_rows[0])
            print(
                "  best holdout:"
                f" start={holdout_rows[0]['start_epoch']}"
                f" offset={holdout_rows[0]['offset_epoch']:+d}"
                f" nlos={100.0 * float(holdout_rows[0]['nlos_fraction']):.2f}%"
            )

    candidate_path = RESULTS_DIR / f"{args.results_prefix}_candidates.csv"
    best_path = RESULTS_DIR / f"{args.results_prefix}_best.csv"
    _save_rows(candidate_rows, candidate_path)
    _save_rows(best_rows, best_path)
    print(f"\nSaved candidates to: {candidate_path}")
    print(f"Saved best-per-run holdouts to: {best_path}")


if __name__ == "__main__":
    main()
