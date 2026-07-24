#!/usr/bin/env python3
"""Replay and lock the common truth-free PF-only Tokyo production finalizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_base_summary(summary: dict[str, Any], selector: dict[str, Any]) -> None:
    keys = (
        "transition_sigma_m",
        "transition_loss",
        "emission_weight",
        "current_seed_consensus_bonus",
        "current_seed_consensus_min_support",
        "external_seed_bonus",
        "external_seed_max_age_epochs",
        "anchor_stride_epochs",
        "path_mode",
        "tdcp_fallback",
        "apply_static_anchor_offset_auto",
    )
    for key in keys:
        if summary.get(key) != selector.get(key):
            raise RuntimeError(
                f"base selector setting {key!r} differs: "
                f"{summary.get(key)!r} != {selector.get(key)!r}"
            )
    if int(summary.get("n_epochs_full_denominator", 0)) != 1200:
        raise RuntimeError("base selector does not use the 1200-epoch denominator")
    if float(summary.get("false_fix_pct", 100.0)) > 1.0:
        raise RuntimeError("base selector exceeds the false-FIX gate")


def _validate_gpu_scale(config: dict[str, Any]) -> dict[str, Any]:
    gpu_config = config.get("gpu_scale")
    if not isinstance(gpu_config, dict):
        raise RuntimeError("locked configuration lacks the WP29 GPU scale gate")
    report_path = _ROOT / gpu_config["report_path"]
    report = _load_json(report_path)
    if report.get("gate") != "WP29_GPU_SCALE" or report.get("status") != "pass":
        raise RuntimeError("WP29 GPU scale report is not a passing locked gate")
    if not all(report.get("checks", {}).values()):
        raise RuntimeError("WP29 GPU scale report contains a failed check")
    if float(report["runtime_total_epochs_per_second"]) < float(
        gpu_config["min_total_epochs_per_second"]
    ):
        raise RuntimeError("WP29 end-to-end throughput is below the locked gate")
    if float(report["position_tolerance_m"]) != float(
        gpu_config["position_tolerance_m"]
    ):
        raise RuntimeError("WP29 parity tolerance differs from locked configuration")
    max_growth = float(gpu_config["max_steady_memory_growth_mib"]) * 1024**2
    if float(report["max_steady_memory_growth_bytes"]) != max_growth:
        raise RuntimeError("WP29 memory tolerance differs from locked configuration")
    return {
        "report_path": str(report_path.relative_to(_ROOT)),
        "report_sha256": _sha256(report_path),
        "status": "pass",
        "lambda_engine": gpu_config["lambda_engine"],
        "runtime_mode": gpu_config["runtime_mode"],
        "runtime_total_epochs_per_second": report[
            "runtime_total_epochs_per_second"
        ],
        "epoch_compute_p99_seconds": report["epoch_compute_p99_seconds"],
        "fix_state_mismatches": report["cpu_vs_gpu_audit"][
            "fix_state_mismatches"
        ],
        "max_position_delta_m": report["cpu_vs_gpu_audit"][
            "max_position_delta_m"
        ],
    }


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "python")
    subprocess.run(command, cwd=_ROOT, env=env, check=True)


def _score_trajectory(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1200 or [int(row["epoch"]) for row in rows] != list(range(1200)):
        raise RuntimeError(f"{path} is not a full ordered 1200-epoch trajectory")
    correct = sum(int(row["sub50cm"]) for row in rows)
    fixed = [row for row in rows if int(row.get("fix", "0"))]
    false = sum(int(row.get("false_fix", "0")) for row in fixed)
    return {
        "n_epochs_full_denominator": 1200,
        "sub50cm_full_epochs": correct,
        "sub50cm_full_pct": 100.0 * correct / 1200.0,
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": false,
        "false_fix_pct": 100.0 * false / max(len(fixed), 1),
    }


def _artifact_paths(stage: dict[str, Any]) -> list[Path]:
    return [
        _ROOT / value
        for key, value in stage.items()
        if key.endswith("_path") and isinstance(value, str)
    ]


def replay_run(
    run_name: str,
    run: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    base_summary_path = _ROOT / run["base_summary_path"]
    base_trajectory_path = _ROOT / run["base_trajectory_path"]
    _validate_base_summary(_load_json(base_summary_path), config["selector"])
    current = output_dir / f"{run_name}_stage0_base.csv"
    shutil.copyfile(base_trajectory_path, current)
    evidence_hashes: dict[str, str] = {
        run["base_summary_path"]: _sha256(base_summary_path),
        run["base_trajectory_path"]: _sha256(base_trajectory_path),
    }
    stage_records: list[dict[str, Any]] = []
    for index, stage_name in enumerate(config["stage_order"], start=1):
        evidence = run.get("stages", {}).get(stage_name)
        if evidence is None:
            stage_records.append(
                {"stage": stage_name, "status": "fail_closed_no_evidence"}
            )
            continue
        for path in _artifact_paths(evidence):
            if not path.is_file():
                raise FileNotFoundError(path)
            evidence_hashes[str(path.relative_to(_ROOT))] = _sha256(path)
        output_trajectory = output_dir / f"{run_name}_stage{index}_{stage_name}.csv"
        output_summary = output_dir / f"{run_name}_stage{index}_{stage_name}.json"
        stage_config = config["stages"][stage_name]
        if stage_name == "moving_offset":
            command = [
                sys.executable,
                "experiments/apply_wp29_moving_offset_shadow.py",
                str(current),
                "--candidates-json",
                evidence["candidates_path"],
                "--selection-json",
                evidence["selection_path"],
                "--data-dir",
                run["data_dir"],
                "--out-summary",
                str(output_summary),
                "--out-trajectory",
                str(output_trajectory),
            ]
        elif stage_name == "route_template_bridge":
            command = [
                sys.executable,
                "experiments/apply_wp29_route_template_bridge_shadow.py",
                str(current),
                "--template-trajectory",
                evidence["template_trajectory_path"],
                "--start",
                str(evidence["start"]),
                "--end",
                str(evidence["end"]),
                "--max-endpoint-distance-m",
                str(stage_config["max_endpoint_distance_m"]),
                "--max-arc-relative-error",
                str(stage_config["max_arc_relative_error"]),
                "--data-dir",
                run["data_dir"],
                "--out-summary",
                str(output_summary),
                "--out-trajectory",
                str(output_trajectory),
            ]
        elif stage_name == "carrier_runner_block":
            command = [
                sys.executable,
                "experiments/apply_wp29_carrier_runner_block_shadow.py",
                str(current),
                evidence["candidate_audit_path"],
                evidence["absolute_evidence_path"],
                "--start",
                str(stage_config["start_epoch"]),
                "--anchor-stride",
                str(stage_config["anchor_stride"]),
                "--min-block-anchors",
                str(stage_config["min_block_anchors"]),
                "--min-carrier-rows",
                str(stage_config["min_carrier_rows"]),
                "--data-dir",
                run["data_dir"],
                "--out-summary",
                str(output_summary),
                "--out-trajectory",
                str(output_trajectory),
            ]
        elif stage_name == "post_jump_route_bridge":
            command = [
                sys.executable,
                "experiments/apply_wp29_post_jump_route_bridge_shadow.py",
                str(current),
                evidence["alternate_route_path"],
                evidence["candidate_audit_path"],
                evidence["route_summary_path"],
                evidence["late_anchor_result_path"],
                "--jump-residual-m",
                str(stage_config["jump_residual_m"]),
                "--min-tail-anchors",
                str(stage_config["min_tail_anchors"]),
                "--data-dir",
                run["data_dir"],
                "--out-summary",
                str(output_summary),
                "--out-trajectory",
                str(output_trajectory),
            ]
        else:
            raise RuntimeError(f"unknown production stage: {stage_name}")
        _run(command)
        stage_summary = _load_json(output_summary)
        if float(stage_summary.get("false_fix_pct", 100.0)) > 1.0:
            raise RuntimeError(f"{run_name}/{stage_name} exceeds false-FIX gate")
        stage_records.append(
            {
                "stage": stage_name,
                "status": "applied",
                "summary_path": str(output_summary.relative_to(_ROOT)),
                "trajectory_path": str(output_trajectory.relative_to(_ROOT)),
                "sub50cm_full_epochs": stage_summary["sub50cm_full_epochs"],
                "false_fix_pct": stage_summary["false_fix_pct"],
            }
        )
        current = output_trajectory
    final_path = output_dir / f"{run_name}_locked_final.csv"
    shutil.copyfile(current, final_path)
    metrics = _score_trajectory(final_path)
    target = float(run["target_pct"])
    if not metrics["sub50cm_full_pct"] > target:
        raise RuntimeError(
            f"{run_name} misses target: {metrics['sub50cm_full_pct']} <= {target}"
        )
    if metrics["false_fix_pct"] > float(config["false_fix_max_pct"]):
        raise RuntimeError(f"{run_name} final false-FIX gate failed")
    return {
        "run": run_name,
        "target_pct": target,
        "final_trajectory_path": str(final_path.relative_to(_ROOT)),
        "final_trajectory_sha256": _sha256(final_path),
        "metrics": metrics,
        "stages": stage_records,
        "evidence_sha256": evidence_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    def resolve_workspace_path(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()

    args.config = resolve_workspace_path(args.config)
    args.ledger = resolve_workspace_path(args.ledger)
    args.output_dir = resolve_workspace_path(args.output_dir)
    args.out_report = resolve_workspace_path(args.out_report)
    config = _load_json(args.config)
    ledger = _load_json(args.ledger)
    if config.get("runtime_fgo") is not False or config.get("pf_only") is not True:
        raise RuntimeError("locked configuration is not explicitly PF-only/no-FGO")
    gpu_scale = _validate_gpu_scale(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        replay_run(name, run, config, args.output_dir)
        for name, run in ledger["runs"].items()
    ]
    report = {
        "gate": "WP30_M4_LOCKED",
        "status": "pass",
        "config_path": str(args.config.relative_to(_ROOT)),
        "config_sha256": _sha256(args.config),
        "ledger_path": str(args.ledger.relative_to(_ROOT)),
        "ledger_sha256": _sha256(args.ledger),
        "same_config_all_runs": True,
        "pf_only": True,
        "runtime_fgo": False,
        "gpu_scale": gpu_scale,
        "runs": results,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
