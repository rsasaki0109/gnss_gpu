#!/usr/bin/env python3
"""Audit affine recovery of WP163 and fail-closed behavior on WP164."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gnss_gpu.ddpr_profiles import ArcScreenPolicy, fit_offset_profile, score_arc_quality
from gnss_gpu.evaluation_contract import M4_PRESERVED_SHA256, sha256_file, write_json


def _audit_rank(
    path: Path,
    *,
    repo_root: Path,
    spread_gate_m: float,
    ddpr_gate_m: float,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("production_input_truth") is not False:
        raise ValueError(f"source is not truth-free: {path}")
    rows = []
    for hypothesis in payload["hypotheses"]:
        offsets = np.asarray(hypothesis["block_offsets_ecef_m"], dtype=np.float64)
        epochs = np.linspace(
            float(payload["segment"][0]),
            float(payload["segment"][1] - 1),
            offsets.shape[0],
        )
        affine = fit_offset_profile(epochs, offsets, mode="affine")
        passes = (
            affine.accepted
            and affine.weighted_rms_m is not None
            and affine.weighted_rms_m <= spread_gate_m
            and float(hypothesis["ddpr_rms_m"]) <= ddpr_gate_m
        )
        rows.append(
            {
                "seed_id": int(hypothesis["seed_id"]),
                "constant_block_spread_m": float(hypothesis["block_spread_m"]),
                "affine_block_rms_m": affine.weighted_rms_m,
                "ddpr_rms_m": float(hypothesis["ddpr_rms_m"]),
                "carrier_rms_cycles": float(hypothesis["carrier_rms_cycles"]),
                "passes_spread_and_ddpr_gates": passes,
            }
        )
    recovered = [row for row in rows if row["passes_spread_and_ddpr_gates"]]
    return {
        "path": path.resolve().relative_to(repo_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "recovered_hypotheses": len(recovered),
        "recovered_seed_ids": [row["seed_id"] for row in recovered],
        "best_affine_block_rms_m": min(
            row["affine_block_rms_m"] for row in rows if row["affine_block_rms_m"] is not None
        ),
        "best_ddpr_rms_m": min(row["ddpr_rms_m"] for row in rows),
    }


def _screen_v2_legacy_audit(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = ArcScreenPolicy()
    qualities = [
        score_arc_quality(
            sat_id=row["sat"],
            start_epoch=payload["segment"][0],
            end_epoch=payload["segment"][1] - 1,
            epochs_present=int(row["epochs_present"]),
            outlier_fraction=float(row["outlier_fraction"]),
            median_abs_residual_m=float(row["median_abs_residual_m"]),
            policy=policy,
        )
        for row in payload["per_satellite"]
    ]
    return {
        "path": path.resolve().relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "hard_excluded_satellites": sorted(
            quality.sat_id for quality in qualities if quality.hard_excluded
        ),
        "soft_weighted_satellites": {
            quality.sat_id: quality.quality_weight
            for quality in qualities
            if not quality.hard_excluded and quality.quality_weight < 0.999
        },
    }


def audit(
    *,
    repo_root: Path,
    wp163_sources: list[Path],
    wp164_sources: list[Path],
    wp163_screen: Path,
    wp164_screen: Path,
) -> dict[str, Any]:
    wp163 = [
        _audit_rank(path, repo_root=repo_root, spread_gate_m=0.5, ddpr_gate_m=4.0)
        for path in wp163_sources
    ]
    wp164 = [
        _audit_rank(path, repo_root=repo_root, spread_gate_m=0.5, ddpr_gate_m=4.0)
        for path in wp164_sources
    ]
    recovered_ranks = sum(row["recovered_hypotheses"] > 0 for row in wp163)
    wp164_false_passes = sum(row["recovered_hypotheses"] for row in wp164)
    m4 = {
        relative: {
            "expected_sha256": expected,
            "actual_sha256": sha256_file(repo_root / relative),
        }
        for relative, expected in M4_PRESERVED_SHA256.items()
    }
    passed = (
        recovered_ranks >= 2
        and wp164_false_passes == 0
        and all(item["expected_sha256"] == item["actual_sha256"] for item in m4.values())
    )
    return {
        "schema": "gnss_gpu_phase2_structural_audit_v1",
        "production_input_truth": False,
        "wp163": {
            "recovered_reference_ranks": recovered_ranks,
            "minimum_required_recovered_ranks": 2,
            "ranks": wp163,
            "screen_v2": _screen_v2_legacy_audit(wp163_screen, repo_root=repo_root),
        },
        "wp164": {
            "false_passing_hypotheses": wp164_false_passes,
            "ranks": wp164,
            "screen_v2": _screen_v2_legacy_audit(wp164_screen, repo_root=repo_root),
        },
        "m4": m4,
        "passed": passed,
    }


def _paths(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 3:
        raise FileNotFoundError(f"expected three rank sources for {pattern}, found {len(paths)}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", type=Path, default=Path("results/wp31"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    results = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    result = audit(
        repo_root=root,
        wp163_sources=_paths(
            results,
            "tokyo_run1_wp163_ref_rank*_phase1_screened_7095_7150_development.json",
        ),
        wp164_sources=_paths(
            results,
            "tokyo_run1_wp164_ref_rank*_phase1_screened_4785_4840_development.json",
        ),
        wp163_screen=results / "tokyo_run1_wp163_ddpr_screen_7095_7150.json",
        wp164_screen=results / "tokyo_run1_wp164_ddpr_screen_4785_4840.json",
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
