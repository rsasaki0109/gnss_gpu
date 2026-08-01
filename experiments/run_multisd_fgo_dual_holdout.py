#!/usr/bin/env python3
"""Run two PPC-only MultiSD holdout partitions concurrently and audit their union."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from experiments.audit_multisd_fgo_dual_holdout import audit_dual_holdout
    from experiments.run_multisd_fgo_ppc_cv import Policy, _run_one
except ModuleNotFoundError:  # Direct `python experiments/<script>.py` execution.
    from audit_multisd_fgo_dual_holdout import audit_dual_holdout
    from run_multisd_fgo_ppc_cv import Policy, _run_one


def _policy(name: str, holdout_satellites: int) -> Policy:
    return Policy(
        name=name,
        window=10,
        minimum_epochs=10,
        holdout_offset=2,
        top_k=4,
        maximum_seed_separation_m=0.5,
        validation_history_epochs=3,
        minimum_carrier_fraction=0.75,
        minimum_fixed_ambiguities=6,
        holdout_satellites=holdout_satellites,
        constellation_ranked_par=True,
        candidate_ratio=1.0,
        candidate_groups=4,
        fallback_consensus_groups=2,
        fallback_consensus_separation_m=0.1,
        fallback_max_seed_separation_m=0.25,
        quality_ranked_par=True,
        interleave_constellation_par=False,
        minimum_bootstrapped_success_rate=0.0,
        maximum_adop_cycles=0.0,
        fallback_minimum_bootstrapped_success_rate=0.9999,
    )


def run_dual_holdout(
    binary: Path,
    data_root: Path,
    output_dir: Path,
    city: str,
    run: str,
    *,
    baseline_pos: Path | None = None,
    max_epochs: int = 0,
    cuda_mode: str = "auto",
    maximum_conflict_separation_m: float = 0.1,
    resume: bool = False,
) -> dict[str, Any]:
    """Run holdout-four and holdout-three in isolated child processes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = (_policy("qf_h4", 4), _policy("qf_h3", 3))

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_one,
                binary.resolve(),
                data_root.resolve(),
                output_dir.resolve(),
                city,
                run,
                policy,
                max_epochs,
                cuda_mode,
                resume,
                False,
            )
            for policy in policies
        ]
        completed = [future.result() for future in futures]
    wall_time_ms = (perf_counter() - start) * 1000.0

    primary_pos, primary_shadow, primary_command = completed[0]
    secondary_pos, secondary_shadow, secondary_command = completed[1]
    reference = data_root / city / run / "reference.csv"
    audit = audit_dual_holdout(
        primary_shadow,
        secondary_shadow,
        reference,
        baseline_pos_path=baseline_pos,
        maximum_conflict_separation_m=maximum_conflict_separation_m,
    )
    return {
        "schema": "gnss_gpu_multisd_fgo_dual_holdout_run_v1",
        "city": city,
        "run": run,
        "max_epochs": max_epochs,
        "cuda_mode": cuda_mode,
        "estimator_inputs": "PPC rover.obs, base.obs, base.nav only",
        "excluded_estimator_inputs": ["imu", "lidar", "camera", "reference"],
        "truth_usage": "reference.csv opened only after both solvers exit",
        "process_isolation": True,
        "concurrent_wall_time_ms": wall_time_ms,
        "commands": [primary_command, secondary_command],
        "partition_artifacts": {
            "holdout_four": {
                "pos": str(primary_pos),
                "shadow": str(primary_shadow),
            },
            "holdout_three": {
                "pos": str(secondary_pos),
                "shadow": str(secondary_shadow),
            },
        },
        "audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", choices=("tokyo", "nagoya"), required=True)
    parser.add_argument("--run", choices=("run1", "run2", "run3"), required=True)
    parser.add_argument("--baseline-pos", type=Path)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--cuda-mode", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--maximum-conflict-separation", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    payload = run_dual_holdout(
        args.binary,
        args.data_root,
        args.output_dir,
        args.city,
        args.run,
        baseline_pos=args.baseline_pos,
        max_epochs=args.max_epochs,
        cuda_mode=args.cuda_mode,
        maximum_conflict_separation_m=args.maximum_conflict_separation,
        resume=args.resume,
    )
    output = args.output_dir / "dual_run_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
