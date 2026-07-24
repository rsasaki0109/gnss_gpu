#!/usr/bin/env python3
"""Apply a promoted WP42 offset and audit the complete PPC denominator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def apply_offset(
    positions: np.ndarray, *, start: int, end: int, offset: np.ndarray
) -> np.ndarray:
    output = np.asarray(positions, dtype=np.float64).copy()
    if output.ndim != 2 or output.shape[1] != 3:
        raise ValueError("trajectory positions must have shape (n, 3)")
    if not 0 <= int(start) < int(end) <= len(output):
        raise ValueError("moving offset segment is outside the trajectory")
    value = np.asarray(offset, dtype=np.float64).reshape(3)
    if not np.isfinite(value).all():
        raise ValueError("moving offset is nonfinite")
    output[int(start) : int(end)] += value
    return output


def apply_linear_bootstrap_profile(
    positions: np.ndarray, *, start: int, end: int, block_offsets: np.ndarray
) -> np.ndarray:
    output = np.asarray(positions, dtype=np.float64).copy()
    offsets = np.asarray(block_offsets, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[0] < 2 or offsets.shape[1] != 3:
        raise ValueError("bootstrap offsets must have shape (blocks, 3)")
    if not np.isfinite(offsets).all() or not 0 <= start < end <= len(output):
        raise ValueError("bootstrap profile inputs are invalid")
    boundaries = np.linspace(start, end, len(offsets) + 1, dtype=int)
    centers = (boundaries[:-1] + boundaries[1:] - 1) / 2.0
    for epoch in range(start, end):
        offset = np.asarray(
            [np.interp(epoch, centers, offsets[:, axis]) for axis in range(3)]
        )
        output[epoch] += offset
    return output


def apply_right_boundary_affine_profile(
    positions: np.ndarray,
    *,
    start: int,
    end: int,
    boundary_epoch: int,
    reference_offset: np.ndarray,
) -> np.ndarray:
    output = np.asarray(positions, dtype=np.float64).copy()
    offset = np.asarray(reference_offset, dtype=np.float64).reshape(3)
    if (
        output.ndim != 2
        or output.shape[1] != 3
        or not 0 <= start < end <= boundary_epoch < len(output)
        or not np.isfinite(offset).all()
    ):
        raise ValueError("right-boundary affine profile inputs are invalid")
    scales = np.asarray(
        [
            (boundary_epoch - epoch) / (boundary_epoch - start)
            for epoch in range(start, end)
        ]
    )
    output[start:end] += scales[:, None] * offset
    return output


def apply_fixed_boundary_affine_profile(
    positions: np.ndarray,
    *,
    start: int,
    end: int,
    reference_offset: np.ndarray,
    boundary_offset: np.ndarray,
) -> np.ndarray:
    output = np.asarray(positions, dtype=np.float64).copy()
    left = np.asarray(reference_offset, dtype=np.float64).reshape(3)
    right = np.asarray(boundary_offset, dtype=np.float64).reshape(3)
    if (
        output.ndim != 2
        or output.shape[1] != 3
        or not 0 <= start < end < len(output)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("fixed-boundary affine profile inputs are invalid")
    scales = np.asarray([(end - epoch) / (end - start) for epoch in range(start, end)])
    output[start:end] += scales[:, None] * left + (1.0 - scales[:, None]) * right
    return output


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("promotion_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    promotion: dict[str, Any] = json.loads(
        args.promotion_json.read_text(encoding="utf-8")
    )
    if not bool(promotion.get("production_promoted", False)) or bool(
        promotion.get("production_input_truth", True)
    ):
        raise RuntimeError("moving offset is not a truth-free production promotion")
    supported_reasons = {
        "unique_moving_temporal_trifrequency_ddpr_rank_consensus",
        "unique_anchor_boundary_identity_profile",
        "unique_cppr_rank_consensus",
        "unique_long_cppr_precursor_boundary",
        "unique_multibasis_two_block_path",
        "multibasis_leading_instability_backward_outage_recovery",
        "unique_affine_multibasis_road_carrier_cluster",
        "unique_singlebasis_road_carrier_cppr_mode",
        "unique_cross_basis_cppr_mode",
        "unique_cppr_anchor_cross_basis_mode",
        "unique_cross_basis_stability_cppr_mode",
    }
    if promotion.get("reason") not in supported_reasons:
        raise RuntimeError("moving offset promotion reason is unsupported")
    rows = _read(args.trajectory)
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
    if not np.array_equal(epochs, np.arange(len(rows))):
        raise RuntimeError("full trajectory epochs are not contiguous from zero")
    positions = np.asarray(
        [
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
            for row in rows
        ]
    )
    start, end = (int(value) for value in promotion["segment"])
    if promotion.get("profile_mode") == "right_boundary_affine_fixed":
        if int(promotion["boundary_epoch"]) != end:
            raise RuntimeError("fixed affine promotion boundary is not segment end")
        output = apply_fixed_boundary_affine_profile(
            positions,
            start=start,
            end=end,
            reference_offset=np.asarray(promotion["offset_ecef_m"]),
            boundary_offset=np.asarray(promotion["boundary_offset_ecef_m"]),
        )
    elif promotion.get("profile_mode") == "right_boundary_affine_zero":
        output = apply_right_boundary_affine_profile(
            positions,
            start=start,
            end=end,
            boundary_epoch=int(promotion["boundary_epoch"]),
            reference_offset=np.asarray(promotion["offset_ecef_m"]),
        )
    elif promotion.get("profile_mode") == "linear_bootstrap_centers":
        output = apply_linear_bootstrap_profile(
            positions,
            start=start,
            end=end,
            block_offsets=np.asarray(promotion["block_offsets_ecef_m"]),
        )
    else:
        output = apply_offset(
            positions,
            start=start,
            end=end,
            offset=np.asarray(promotion["offset_ecef_m"]),
        )

    # Production positions and the applied epoch mask are frozen above. Truth is
    # loaded only now for the complete-denominator audit and never changes output.
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    times = np.asarray([float(row["tow"]) for row in rows])
    truth = np.asarray(
        [truth_positions[int(np.argmin(np.abs(truth_times - tow)))] for tow in times]
    )
    before_errors = np.linalg.norm(positions - truth, axis=1)
    after_errors = np.linalg.norm(output - truth, axis=1)
    before_sub50 = before_errors < 0.5
    after_sub50 = after_errors < 0.5
    for index, row in enumerate(rows):
        row["ecef_x"], row["ecef_y"], row["ecef_z"] = (
            repr(float(value)) for value in output[index]
        )
        row["error_m"] = repr(float(after_errors[index]))
        row["sub50cm"] = str(int(after_sub50[index]))
    args.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.output_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fix = np.asarray([int(row.get("fix", "0")) for row in rows], dtype=np.int64)
    false_fix = np.asarray(
        [int(row.get("false_fix", "0")) for row in rows], dtype=np.int64
    )
    summary_schema = {
        "unique_anchor_boundary_identity_profile": (
            "wp45_anchor_boundary_identity_benchmark_v1"
        ),
        "unique_moving_temporal_trifrequency_ddpr_rank_consensus": (
            "wp43_moving_temporal_trifrequency_benchmark_v1"
        ),
        "unique_cppr_rank_consensus": "wp55_cppr_rank_benchmark_v1",
        "unique_long_cppr_precursor_boundary": ("wp57_precursor_boundary_benchmark_v1"),
        "unique_multibasis_two_block_path": "wp60_two_block_path_benchmark_v1",
        "multibasis_leading_instability_backward_outage_recovery": (
            "wp62_backward_outage_path_benchmark_v1"
        ),
        "unique_affine_multibasis_road_carrier_cluster": (
            "wp76_affine_multibasis_benchmark_v1"
        ),
        "unique_singlebasis_road_carrier_cppr_mode": (
            "wp87_fixed_affine_singlebasis_benchmark_v1"
        ),
        "unique_cross_basis_cppr_mode": "wp131_cross_basis_cppr_benchmark_v1",
        "unique_cppr_anchor_cross_basis_mode": (
            "wp133_cppr_anchor_benchmark_v1"
        ),
        "unique_cross_basis_stability_cppr_mode": (
            "wp138_stability_cppr_benchmark_v1"
        ),
    }[promotion["reason"]]
    if (
        promotion["reason"] == "unique_singlebasis_road_carrier_cppr_mode"
        and promotion.get("profile_mode") == "constant"
    ):
        summary_schema = (
            "wp93_constant_singlebasis_benchmark_v1"
            if promotion.get("schema", "").startswith("wp93_")
            else "wp_constant_singlebasis_benchmark_v1"
        )
    summary = {
        "schema": summary_schema,
        "production_input_truth": False,
        "truth_usage": "post_application_full_denominator_audit_only",
        "production_promoted": True,
        "full_denominator_epochs": len(rows),
        "segment": [start, end],
        "before_sub50cm_epochs": int(np.count_nonzero(before_sub50)),
        "after_sub50cm_epochs": int(np.count_nonzero(after_sub50)),
        "after_sub50cm_pct": float(100.0 * np.mean(after_sub50)),
        "gained_epochs": int(np.count_nonzero(after_sub50 & ~before_sub50)),
        "lost_epochs": int(np.count_nonzero(before_sub50 & ~after_sub50)),
        "fix_epochs": int(np.count_nonzero(fix)),
        "false_fix_epochs": int(np.count_nonzero(false_fix)),
        "declared_false_fix_pct": float(
            100.0 * np.count_nonzero(false_fix) / max(np.count_nonzero(fix), 1)
        ),
        "input_sha256": {
            "trajectory": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
            "promotion": hashlib.sha256(args.promotion_json.read_bytes()).hexdigest(),
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
