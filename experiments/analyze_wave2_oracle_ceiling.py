#!/usr/bin/env python3
"""Oracle ceiling for Wave 2 candidate pool: best-of hybrid + w2 variants per tow."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402


def _load_reference(data_root: Path, city: str, run: str) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    path = data_root / city / run / "reference.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                tow = round(float(row["GPS TOW (s)"]), 1)
                xyz = np.array(
                    [
                        float(row["ECEF X (m)"]),
                        float(row["ECEF Y (m)"]),
                        float(row["ECEF Z (m)"]),
                    ],
                    dtype=np.float64,
                )
                out[tow] = xyz
            except (ValueError, KeyError):
                continue
    return out


def _load_pos(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                tow = round(float(parts[1]), 1)
                xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
                out[tow] = xyz
            except ValueError:
                continue
    return out


def _manifest_pool(manifest_dir: Path, city: str, run: str, project_root: Path) -> list[tuple[str, Path]]:
    manifest_path = manifest_dir / f"{city}_{run}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[tuple[str, Path]] = []
    for label, rel_dir in zip(payload["labels"], payload["dirs"], strict=True):
        pos_path = project_root / rel_dir / f"{city}_{run}_full.pos"
        out.append((str(label), pos_path))
    return out


def _segment_score(
    tows: list[float],
    positions: list[np.ndarray],
    reference: dict[float, np.ndarray],
) -> dict[str, float | int]:
    est: list[np.ndarray] = []
    ref: list[np.ndarray] = []
    for tow, pos in zip(tows, positions, strict=True):
        ref_xyz = reference.get(tow)
        if ref_xyz is None or not np.all(np.isfinite(pos)):
            continue
        est.append(pos)
        ref.append(ref_xyz)
    if not est:
        return {"segment_ppc_pct": 0.0, "segment_pass_m": 0.0, "segment_total_m": 0.0, "n_epochs": 0}
    est_arr = np.asarray(est, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    score = score_ppc2024(est_arr, ref_arr)
    return {
        "segment_ppc_pct": float(score.score_pct),
        "segment_pass_m": float(score.pass_distance_m),
        "segment_total_m": float(score.total_distance_m),
        "n_epochs": int(score.n_epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="nagoya/run2")
    parser.add_argument("--data-root", type=Path, default=Path("E:/datasets/PPC-Dataset-data"))
    parser.add_argument("--hybrid-pos-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "experiments/results/wave2_oracle_ceiling_summary.json",
    )
    args = parser.parse_args()

    city, run = args.run.strip("/").split("/", 1)
    reference = _load_reference(args.data_root, city, run)
    hybrid_path = args.hybrid_pos_dir / f"{city}_{run}_full.pos"
    hybrid = _load_pos(hybrid_path)
    pool = _manifest_pool(args.manifest_dir, city, run, PROJECT_ROOT)
    candidate_maps = {label: _load_pos(path) for label, path in pool}

    # Use hybrid-covered tows (same support as Wave 2 smoke rtkdiag inputs).
    tows = sorted(set(hybrid.keys()) & set(reference.keys()))
    hybrid_positions: list[np.ndarray] = []
    oracle_positions: list[np.ndarray] = []
    oracle_labels: list[str] = []
    for tow in tows:
        ref = reference[tow]
        hyb = hybrid[tow]
        hybrid_positions.append(hyb)
        best_label = "hybrid"
        best_pos = hyb
        best_err = float(np.linalg.norm(hyb - ref))
        for label, pos_map in candidate_maps.items():
            cand = pos_map.get(tow)
            if cand is None:
                continue
            err = float(np.linalg.norm(cand - ref))
            if err < best_err:
                best_err = err
                best_pos = cand
                best_label = label
        oracle_positions.append(best_pos)
        oracle_labels.append(best_label)

    hybrid_score = _segment_score(tows, hybrid_positions, reference)
    oracle_score = _segment_score(tows, oracle_positions, reference)
    headroom_pp = float(oracle_score["segment_ppc_pct"] - hybrid_score["segment_ppc_pct"])

    label_counts: dict[str, int] = {}
    for label in oracle_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    report = {
        "run": args.run,
        "start_epoch": int(args.start_epoch),
        "max_epochs": int(args.max_epochs),
        "n_tows": len(tows),
        "hybrid_only": hybrid_score,
        "oracle_best_of_pool": oracle_score,
        "headroom_pp": headroom_pp,
        "oracle_label_counts": label_counts,
        "pool_labels": [label for label, _path in pool],
        "note": (
            "Segment PPC on hybrid-covered tows; oracle picks min 3D error among "
            "hybrid + Wave 2 candidates per tow."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
