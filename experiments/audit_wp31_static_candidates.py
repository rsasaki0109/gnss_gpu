#!/usr/bin/env python3
"""Truth-only post-selection audit of a frozen static candidate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def audit_candidates(
    candidates: list[dict[str, Any]], truth_segment: np.ndarray
) -> list[dict[str, Any]]:
    truth = np.asarray(truth_segment, dtype=np.float64).reshape(-1, 3)
    truth = truth[np.isfinite(truth).all(axis=1)]
    if not len(truth):
        raise ValueError("truth segment has no finite rows")
    center = np.median(truth, axis=0)
    rows = [
        {
            "candidate_id": int(row["candidate_id"]),
            "audit_error_m": float(np.linalg.norm(np.asarray(row["position_ecef"]) - center)),
        }
        for row in candidates
    ]
    rows.sort(key=lambda row: float(row["audit_error_m"]))
    for rank, row in enumerate(rows, start=1):
        row["audit_rank"] = rank
        row["audit_sub50cm"] = int(float(row["audit_error_m"]) < 0.5)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    start, end = (int(value) for value in source["segment"])
    _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
    rows = audit_candidates(list(source["candidates"]), np.asarray(truth[start:end]))
    result = {
        "schema": "wp31_static_candidate_truth_audit_v1",
        "production_input": False,
        "segment": [start, end],
        "candidate_count": len(rows),
        "sub50cm_candidate_count": sum(row["audit_sub50cm"] for row in rows),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
