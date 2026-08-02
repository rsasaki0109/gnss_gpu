#!/usr/bin/env python3
"""Post-estimator score and safety audit for a causal PPC FLOAT selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.evaluate_ppc_official_score import (
        evaluate_route,
        read_estimates,
        read_reference,
    )
except ModuleNotFoundError:
    from evaluate_ppc_official_score import (  # type: ignore[no-redef]
        evaluate_route,
        read_estimates,
        read_reference,
    )
from gnss_gpu.ppc_score import score_ppc2024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aligned(path: Path, reference_tow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    estimates, statuses = read_estimates(path)
    positions = np.full((reference_tow.size, 3), np.nan, dtype=np.float64)
    fixed = np.zeros(reference_tow.size, dtype=bool)
    for index, tow in enumerate(reference_tow):
        key = float(tow)
        if key in estimates:
            positions[index] = estimates[key]
        fixed[index] = statuses.get(key, 0) != 0
    return positions, fixed


def audit_selector(
    safe_output: Path,
    safe_summary: Path,
    selected_output: Path,
    selector_summary: Path,
    reference: Path,
    blocks: int = 5,
) -> dict[str, Any]:
    if blocks < 1:
        raise ValueError("blocks must be positive")
    safe_metadata = json.loads(safe_summary.read_text(encoding="utf-8"))
    selector_metadata = json.loads(selector_summary.read_text(encoding="utf-8"))
    if (
        safe_metadata.get("output_sha256") != _sha256(safe_output)
        or selector_metadata.get("output_sha256") != _sha256(selected_output)
        or selector_metadata.get("production_input_truth") is not False
        or selector_metadata.get("truth_usage") != "none"
        or selector_metadata.get("input_sha256", {}).get("safe_output")
        != _sha256(safe_output)
    ):
        raise ValueError("selector artifact integrity check failed")

    safe_estimates, safe_statuses = read_estimates(safe_output)
    selected_estimates, selected_statuses = read_estimates(selected_output)
    safe_fixed_tows = {tow for tow, status in safe_statuses.items() if status != 0}
    selected_fixed_tows = {
        tow for tow, status in selected_statuses.items() if status != 0
    }
    same_fixed_set = safe_fixed_tows == selected_fixed_tows
    fixed_positions_identical = same_fixed_set and all(
        np.array_equal(safe_estimates[tow], selected_estimates[tow])
        for tow in safe_fixed_tows
    )
    if not same_fixed_set or not fixed_positions_identical:
        raise ValueError("selector changed safe FIX authority or position")

    reference_tow, reference_xyz = read_reference(reference)
    safe_xyz, _ = _aligned(safe_output, reference_tow)
    selected_xyz, _ = _aligned(selected_output, reference_tow)

    block_results = []
    for block_number, indices in enumerate(
        np.array_split(np.arange(reference_tow.size), blocks)
    ):
        safe_score = score_ppc2024(safe_xyz[indices], reference_xyz[indices])
        selected_score = score_ppc2024(selected_xyz[indices], reference_xyz[indices])
        block_results.append(
            {
                "block": block_number,
                "epochs": int(indices.size),
                "safe_score_pct": safe_score.score_pct,
                "selected_score_pct": selected_score.score_pct,
                "delta_pct_points": selected_score.score_pct - safe_score.score_pct,
            }
        )

    safe_score = evaluate_route(safe_output, reference)
    selected_score = evaluate_route(selected_output, reference)
    return {
        "schema": "gnss_gpu_ppc_causal_float_selector_audit_v1",
        "truth_usage": "post_estimator_scoring_only",
        "truth_opened_after_estimator_artifact_hashes": True,
        "forward_only": True,
        "safe_fix_set_identical": same_fixed_set,
        "safe_fix_positions_identical": fixed_positions_identical,
        "safe": safe_score,
        "selected": selected_score,
        "score_delta_pct_points": (
            selected_score["ppc_score_pct"] - safe_score["ppc_score_pct"]
        ),
        "all_blocks_non_degrading": all(
            row["delta_pct_points"] >= 0.0 for row in block_results
        ),
        "blocks": block_results,
        "input_sha256": {
            "safe_output": _sha256(safe_output),
            "safe_summary": _sha256(safe_summary),
            "selected_output": _sha256(selected_output),
            "selector_summary": _sha256(selector_summary),
            "reference": _sha256(reference),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--selector-summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_selector(
            args.safe_output,
            args.safe_summary,
            args.selected_output,
            args.selector_summary,
            args.reference,
            args.blocks,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
