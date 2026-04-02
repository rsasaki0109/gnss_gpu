#!/usr/bin/env python3
"""Scan PPC trajectory segments for real-PLATEAU LOS/NLOS interactions.

This script loads one PPC run, derives third-level mesh codes from the usable
ground-truth epochs, extracts matching PLATEAU building tiles over HTTP range
requests, and reports per-segment ray-traced NLOS fractions.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate import ecef_to_lla, save_results
from exp_urbannav_baseline import load_real_data
from fetch_plateau_subset import (
    HTTPRangeReader,
    PRESET_URLS,
    expand_meshes,
    extract_entries,
    mesh3_code,
    select_bldg_entries,
)


RESULTS_DIR = _SCRIPT_DIR / "results"


def derive_mesh_codes(ground_truth_ecef: np.ndarray) -> list[str]:
    codes: list[str] = []
    for pos in ground_truth_ecef:
        lat, lon, _ = ecef_to_lla(pos[0], pos[1], pos[2])
        codes.append(mesh3_code(np.degrees(lat), np.degrees(lon)))
    return codes


def find_segment_starts(mesh_codes: list[str]) -> list[int]:
    if not mesh_codes:
        return []
    starts = [0]
    for i in range(1, len(mesh_codes)):
        if mesh_codes[i] != mesh_codes[i - 1]:
            starts.append(i)
    return starts


def sliding_segment_starts(n_epochs: int, segment_length: int, start_step: int) -> list[int]:
    if n_epochs <= 0:
        return []
    step = max(int(start_step), 1)
    starts = list(range(0, n_epochs, step))
    last_start = max(0, n_epochs - max(int(segment_length), 1))
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def ensure_subset(
    zip_url: str,
    meshes: list[str],
    subset_root: Path,
) -> Path:
    mesh_key = hashlib.sha1(",".join(meshes).encode("utf-8")).hexdigest()[:16]
    out_dir = subset_root / mesh_key
    manifest = out_dir / "manifest.txt"
    gml_files = list(out_dir.glob("*.gml"))
    if gml_files and manifest.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HTTPRangeReader(zip_url)) as zf:
        entries = select_bldg_entries(zf, meshes)
        if not entries:
            raise RuntimeError(f"no PLATEAU building tiles matched meshes: {meshes}")
        extract_entries(zf, entries, out_dir)
        manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return out_dir


def compute_nlos_fraction(
    model,
    ground_truth: np.ndarray,
    sat_ecef: list[np.ndarray],
    start: int,
    stop: int,
    sample_stride: int,
) -> tuple[int, int]:
    n_total = 0
    n_nlos = 0
    for idx in range(start, stop, sample_stride):
        sats = np.asarray(sat_ecef[idx], dtype=np.float64).reshape(-1, 3)
        if sats.size == 0:
            continue
        los = np.asarray(model.check_los(ground_truth[idx], sats), dtype=bool)
        n_total += int(los.size)
        n_nlos += int(np.count_nonzero(~los))
    return n_nlos, n_total


def save_scan_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    save_results(
        {key: [row[key] for row in rows] for key in rows[0].keys()},
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan PPC segments for PLATEAU NLOS interactions")
    parser.add_argument("--run-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESET_URLS),
        default="tokyo23",
        help="Built-in PLATEAU ZIP preset",
    )
    parser.add_argument("--zip-url", type=str, default="", help="Override PLATEAU ZIP URL")
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument("--plateau-zone", type=int, default=9, help="PLATEAU plane-rect zone")
    parser.add_argument("--segment-length", type=int, default=100, help="Epochs per segment")
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="If > 0, scan sliding windows every N epochs instead of mesh-boundary starts",
    )
    parser.add_argument("--mesh-radius", type=int, default=1, help="Neighboring mesh expansion radius")
    parser.add_argument("--sample-stride", type=int, default=5, help="Epoch stride for ray-trace evaluation")
    parser.add_argument("--max-segments", type=int, default=None, help="Optional cap on number of segments")
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("/tmp/plateau_segment_cache"),
        help="Directory for extracted PLATEAU subset caches",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_plateau_segment_scan",
        help="CSV filename prefix under experiments/results/",
    )
    parser.add_argument(
        "--stop-on-positive",
        action="store_true",
        help="Stop once a segment with non-zero NLOS fraction is found",
    )
    args = parser.parse_args()
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    zip_url = args.zip_url or PRESET_URLS[args.preset]
    output_path = RESULTS_DIR / f"{args.results_prefix}.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.subset_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PLATEAU Segment Scan")
    print("=" * 72)

    print("\n[1] Loading PPC run ...")
    data = load_real_data(args.run_dir, systems=systems)
    if data is None:
        raise RuntimeError(f"failed to load PPC data: {args.run_dir}")

    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    sat_ecef = data["sat_ecef"]
    mesh_codes = derive_mesh_codes(ground_truth)
    if args.start_step > 0:
        segment_starts = sliding_segment_starts(data["n_epochs"], args.segment_length, args.start_step)
        start_mode = f"sliding step={args.start_step}"
    else:
        segment_starts = find_segment_starts(mesh_codes)
        start_mode = "mesh-boundary"
    if args.max_segments is not None:
        segment_starts = segment_starts[:args.max_segments]
    print(f"    usable epochs: {data['n_epochs']}")
    print(f"    segment starts ({start_mode}): {len(segment_starts)}")

    rows: list[dict[str, object]] = []
    model_cache: dict[str, object] = {}
    triangle_cache: dict[str, int] = {}

    print("\n[2] Scanning segments ...")
    from gnss_gpu.io.plateau import load_plateau

    for seg_id, start in enumerate(segment_starts):
        stop = min(start + args.segment_length, data["n_epochs"])
        meshes = expand_meshes(sorted(set(mesh_codes[start:stop])), args.mesh_radius)
        print(
            f"    loading seg {seg_id:02d} start={start:5d} "
            f"mesh={mesh_codes[start]} meshes={len(meshes)}",
            flush=True,
        )
        subset_dir = ensure_subset(zip_url, meshes, args.subset_root)
        mesh_key = subset_dir.name
        if mesh_key not in model_cache:
            model = load_plateau(subset_dir, zone=args.plateau_zone)
            model_cache[mesh_key] = model
            triangle_cache[mesh_key] = int(model.triangles.shape[0])
        model = model_cache[mesh_key]
        n_nlos, n_total = compute_nlos_fraction(
            model,
            ground_truth,
            sat_ecef,
            start,
            stop,
            max(args.sample_stride, 1),
        )
        frac = float(n_nlos / n_total) if n_total else 0.0
        row = {
            "segment_id": seg_id,
            "start_epoch": start,
            "end_epoch": stop - 1,
            "n_epochs": stop - start,
            "start_mesh": mesh_codes[start],
            "n_meshes": len(meshes),
            "subset_key": mesh_key,
            "triangle_count": triangle_cache[mesh_key],
            "n_nlos": n_nlos,
            "n_total_obs": n_total,
            "nlos_fraction": frac,
        }
        rows.append(row)
        save_scan_rows(rows, output_path)
        print(
            f"    seg {seg_id:02d} start={start:5d} mesh={mesh_codes[start]} "
            f"tri={triangle_cache[mesh_key]:7d} nlos={n_nlos:4d}/{n_total:4d} "
            f"frac={100.0 * frac:5.2f}%",
            flush=True,
        )
        if args.stop_on_positive and frac > 0.0:
            print("    stopping early on first positive NLOS segment", flush=True)
            break

    if not rows:
        raise RuntimeError("no segments were scanned")

    rows.sort(key=lambda row: (row["nlos_fraction"], row["triangle_count"]), reverse=True)
    save_scan_rows(rows, output_path)
    print(f"\nTop segment: start_epoch={rows[0]['start_epoch']}, "
          f"mesh={rows[0]['start_mesh']}, nlos_fraction={100.0 * rows[0]['nlos_fraction']:.2f}%")


if __name__ == "__main__":
    main()
