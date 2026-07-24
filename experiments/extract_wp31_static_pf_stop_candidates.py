#!/usr/bin/env python3
"""Extract compact recurring PF candidates for multiple static stops."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_static_reanchor_shadow import recurring_position_candidates  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_segment(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("segment must be START:END") from exc
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("segment must satisfy 0 <= START < END")
    return start, end


def extract_stop_candidates(
    basin_csv: Path,
    segments: list[tuple[int, int]],
    *,
    sample_stride_epochs: int = 5,
    radius_m: float = 0.2,
    dedup_radius_m: float = 0.2,
    max_candidates: int = 24,
    chunksize: int = 250_000,
) -> list[dict[str, Any]]:
    if not segments:
        raise ValueError("at least one static segment is required")
    ordered = sorted(segments)
    if len(set(ordered)) != len(ordered):
        raise ValueError("static segments must be unique")
    positions: list[dict[int, list[np.ndarray]]] = [{} for _ in ordered]
    lower = min(start for start, _end in ordered)
    upper = max(end for _start, end in ordered)
    for chunk in pd.read_csv(
        basin_csv,
        usecols=["epoch", "ecef_x", "ecef_y", "ecef_z"],
        chunksize=int(chunksize),
    ):
        relevant = chunk[(chunk["epoch"] >= lower) & (chunk["epoch"] < upper)]
        if relevant.empty:
            continue
        epochs = relevant["epoch"].to_numpy(dtype=np.int64)
        xyz = relevant[["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=np.float64)
        for epoch, position in zip(epochs, xyz):
            for index, (start, end) in enumerate(ordered):
                if start <= int(epoch) < end:
                    positions[index].setdefault(int(epoch), []).append(position)
                    break
    nodes = []
    for (start, end), by_epoch in zip(ordered, positions):
        arrays = {epoch: np.asarray(rows) for epoch, rows in by_epoch.items()}
        candidates = recurring_position_candidates(
            arrays,
            start,
            end,
            radius_m=float(radius_m),
            sample_stride_epochs=int(sample_stride_epochs),
            dedup_radius_m=float(dedup_radius_m),
            max_candidates=int(max_candidates),
        )
        encoded_candidates = []
        for candidate_id, row in enumerate(candidates):
            encoded = dict(row)
            encoded["candidate_id"] = candidate_id
            encoded["position_ecef"] = np.asarray(row["position_ecef"]).tolist()
            encoded_candidates.append(encoded)
        nodes.append(
            {
                "segment": [start, end],
                "source_epoch_count": len(by_epoch),
                "candidate_count": len(encoded_candidates),
                "candidates": encoded_candidates,
            }
        )
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_csv", type=Path)
    parser.add_argument("--segment", action="append", type=parse_segment, required=True)
    parser.add_argument("--sample-stride-epochs", type=int, default=5)
    parser.add_argument("--radius-m", type=float, default=0.2)
    parser.add_argument("--dedup-radius-m", type=float, default=0.2)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nodes = extract_stop_candidates(
        args.basin_csv,
        list(args.segment),
        sample_stride_epochs=args.sample_stride_epochs,
        radius_m=args.radius_m,
        dedup_radius_m=args.dedup_radius_m,
        max_candidates=args.max_candidates,
        chunksize=args.chunksize,
    )
    digest = _sha256_file(args.basin_csv)
    result = {
        "schema": "wp31_static_pf_stop_candidates_v1",
        "basin_csv": str(args.basin_csv).replace("\\", "/"),
        "basin_csv_sha256": digest,
        "sample_stride_epochs": args.sample_stride_epochs,
        "radius_m": args.radius_m,
        "dedup_radius_m": args.dedup_radius_m,
        "max_candidates": args.max_candidates,
        "nodes": nodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "nodes": [{k: v for k, v in node.items() if k != "candidates"} for node in nodes]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
