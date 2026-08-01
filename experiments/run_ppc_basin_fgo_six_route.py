#!/usr/bin/env python3
"""Run the truth-free basin FGO pipeline, then audit all six PPC routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROUTES = tuple(
    (city, f"run{number}")
    for city in ("tokyo", "nagoya")
    for number in range(1, 4)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_pre_run_manifest(
    path: Path,
    *,
    binary: Path,
    route_dir: Path,
    command: list[str],
    native_imu: bool,
) -> None:
    """Freeze estimator provenance before the process can create artifacts."""

    input_names = ["rover.obs", "base.obs", "base.nav"]
    if native_imu:
        input_names.append("imu.csv")
    inputs = {name: route_dir / name for name in input_names}
    missing = [str(value) for value in inputs.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"missing estimator input(s): {', '.join(missing)}")
    payload = {
        "schema": "gnss_gpu_ppc_basin_fgo_pre_run_manifest_v1",
        "production_input_truth": False,
        "reference_in_command": any("reference" in value.lower() for value in command),
        "command": command,
        "binary": {"path": str(binary), "sha256": _sha256(binary)},
        "inputs": {
            name: {"path": str(value), "sha256": _sha256(value)}
            for name, value in inputs.items()
        },
    }
    if payload["reference_in_command"]:
        raise ValueError("estimator command must not contain a reference path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _solver_command(
    binary: Path,
    route_dir: Path,
    output_dir: Path,
    stem: str,
    max_epochs: int,
    top_k: int = 4,
    native_imu: bool = False,
    skip_epochs: int = 0,
) -> tuple[list[str], dict[str, Path]]:
    artifacts = {
        "pos": output_dir / f"{stem}.pos",
        "shadow": output_dir / f"{stem}.shadow.csv",
        "basins": output_dir / f"{stem}.basins.jsonl",
        "tracker": output_dir / f"{stem}.tracker.csv",
        "tracker_summary": output_dir / f"{stem}.tracker.json",
        "audit": output_dir / f"{stem}.audit.json",
        "candidate_audit": output_dir / f"{stem}.candidate_audit.json",
    }
    command = [
        str(binary),
        "--rover", str(route_dir / "rover.obs"),
        "--base", str(route_dir / "base.obs"),
        "--nav", str(route_dir / "base.nav"),
        "--preset", "low-cost",
        "--no-kml",
        "--out", str(artifacts["pos"]),
        "--multisd-fgo-shadow-csv", str(artifacts["shadow"]),
        "--multisd-fgo-basin-jsonl", str(artifacts["basins"]),
        "--multisd-fgo-shadow-window", "10",
        "--multisd-fgo-shadow-min-epochs", "10",
        "--multisd-fgo-shadow-holdout-offset", "2",
        "--multisd-fgo-shadow-top-k", str(top_k),
        "--multisd-fgo-shadow-max-seed-separation", "0.5",
        "--multisd-fgo-shadow-validation-history", "3",
        "--multisd-fgo-shadow-min-carrier-fraction", "0.75",
        "--multisd-fgo-shadow-min-fixed-ambiguities", "6",
        "--multisd-fgo-shadow-holdout-satellites", "4",
        "--multisd-fgo-shadow-candidate-ratio", "1.5",
        "--multisd-fgo-shadow-candidate-groups", "1",
        "--multisd-fgo-shadow-fallback-consensus-groups", "1",
        "--multisd-fgo-shadow-fallback-consensus-separation", "0",
        "--multisd-fgo-shadow-fallback-max-seed-separation", "0",
        "--multisd-fgo-shadow-min-bsr", "0",
        "--multisd-fgo-shadow-max-adop", "0",
        "--multisd-fgo-shadow-fallback-min-bsr", "0",
    ]
    if max_epochs > 0:
        command.extend(("--max-epochs", str(max_epochs)))
    if skip_epochs > 0:
        command.extend(("--skip-epochs", str(skip_epochs)))
    if native_imu:
        city = route_dir.parent.name.lower()
        lever_by_city = {
            "tokyo": (0.31, 0.0, 0.55),
            "nagoya": (0.593, 0.670, 1.216),
        }
        if city not in lever_by_city:
            raise ValueError(f"no audited PPC IMU lever arm for city: {city}")
        command.extend(("--multisd-fgo-imu", str(route_dir / "imu.csv")))
        command.append("--multisd-fgo-imu-lever-arm-flu")
        command.extend(str(value) for value in lever_by_city[city])
        command.extend(("--multisd-fgo-imu-fixed-lag", "5"))
    return command, artifacts


def _shadow_runtime(path: Path) -> dict[str, float | int | None]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row["runtime_ms"]) for row in rows]
    result: dict[str, float | int | None] = {
        "epochs": len(rows),
        "mean_ms": sum(values) / len(values) if values else None,
        "p95_ms": _quantile(values, 0.95),
        "maximum_ms": max(values, default=None),
    }
    if rows and "imu_fgo_runtime_ms" in rows[0]:
        imu_values = [float(row["imu_fgo_runtime_ms"]) for row in rows]
        result.update(
            {
                "imu_available_epochs": sum(
                    row["imu_fgo_available"] == "1" for row in rows
                ),
                "imu_warm_started_epochs": sum(
                    row["imu_fgo_warm_started"] == "1" for row in rows
                ),
                "imu_mean_ms": sum(imu_values) / len(imu_values),
                "imu_p95_ms": _quantile(imu_values, 0.95),
                "imu_maximum_ms": max(imu_values),
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--skip-epochs", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cuda-mode", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--route", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--imu", action="store_true",
        help="enable the legacy tracker-only IMU bridge ablation",
    )
    parser.add_argument(
        "--native-imu", action="store_true",
        help="enable native Pose3/velocity/bias fixed-lag IMU FGO",
    )
    parser.add_argument(
        "--native-imu-aperture", type=float, default=0.0,
        help="PF aperture in metres among GNSS-holdout-passing basins (0 disables)",
    )
    parser.add_argument(
        "--native-imu-fix-min-streak", type=int, default=0,
        help="IMU-consistent GNSS pass streak for accelerated reacquisition (0 disables)",
    )
    args = parser.parse_args(argv)
    if not 2 <= args.top_k <= 32:
        parser.error("--top-k must be in [2, 32]")
    if args.skip_epochs < 0:
        parser.error("--skip-epochs must be non-negative")
    binary = args.binary.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        [tuple(value.lower().split("/", 1)) for value in args.route]
        if args.route
        else list(ROUTES)
    )
    if any(route not in ROUTES for route in selected):
        parser.error("--route must be tokyo/run1..3 or nagoya/run1..3")
    if args.native_imu_aperture < 0.0:
        parser.error("--native-imu-aperture must be non-negative")
    if args.native_imu_aperture > 0.0 and not args.native_imu:
        parser.error("--native-imu-aperture requires --native-imu")
    if args.native_imu_fix_min_streak not in (0,) and args.native_imu_fix_min_streak < 2:
        parser.error("--native-imu-fix-min-streak must be 0 or at least 2")
    if args.native_imu_fix_min_streak > 0 and not args.native_imu:
        parser.error("--native-imu-fix-min-streak requires --native-imu")

    environment = os.environ.copy()
    environment["GNSSPP_FGO_CUDA_SOLVER"] = args.cuda_mode
    route_results = []
    for city, run in selected:
        stem = (
            f"{city}_{run}_basin_fgo_k{args.top_k}_e{args.max_epochs or 'full'}"
            f"_s{args.skip_epochs}"
            if args.skip_epochs
            else f"{city}_{run}_basin_fgo_k{args.top_k}_e{args.max_epochs or 'full'}"
        )
        route_dir = data_root / city / run
        solver_command, artifacts = _solver_command(
            binary, route_dir, output_dir, stem, args.max_epochs, args.top_k,
            args.native_imu, args.skip_epochs,
        )
        pre_run_manifest = output_dir / f"{stem}.run_manifest.json"
        _write_pre_run_manifest(
            pre_run_manifest,
            binary=binary,
            route_dir=route_dir,
            command=solver_command,
            native_imu=args.native_imu,
        )
        solver_complete = all(
            artifacts[name].is_file() and artifacts[name].stat().st_size > 0
            for name in ("pos", "shadow", "basins")
        )
        if not (args.resume and solver_complete):
            subprocess.run(solver_command, check=True, env=environment)

        tracker_command = [
            sys.executable,
            str(Path(__file__).with_name("run_ppc_basin_fgo_tracker.py")),
            "--basin-jsonl", str(artifacts["basins"]),
            "--output", str(artifacts["tracker"]),
            "--summary", str(artifacts["tracker_summary"]),
        ]
        if args.imu:
            tracker_command.extend(("--imu-csv", str(route_dir / "imu.csv")))
        if args.native_imu:
            tracker_command.append("--native-imu-fgo")
        if args.native_imu_aperture > 0.0:
            tracker_command.extend(
                ("--native-imu-aperture", str(args.native_imu_aperture))
            )
        if args.native_imu_fix_min_streak > 0:
            tracker_command.extend(
                (
                    "--native-imu-fix-min-streak",
                    str(args.native_imu_fix_min_streak),
                )
            )
        subprocess.run(tracker_command, check=True)

        # Only this final subprocess receives a reference path. Both estimator
        # processes have exited and their content hashes are already fixed.
        audit_command = [
            sys.executable,
            str(Path(__file__).with_name("audit_ppc_basin_fgo_tracker.py")),
            "--tracker-csv", str(artifacts["tracker"]),
            "--tracker-summary", str(artifacts["tracker_summary"]),
            "--baseline-pos", str(artifacts["pos"]),
            "--reference", str(route_dir / "reference.csv"),
            "--output", str(artifacts["audit"]),
        ]
        subprocess.run(audit_command, check=True)
        candidate_audit_command = [
            sys.executable,
            str(Path(__file__).with_name("audit_ppc_basin_fgo_candidate_supply.py")),
            "--basin-jsonl", str(artifacts["basins"]),
            "--reference", str(route_dir / "reference.csv"),
            "--output", str(artifacts["candidate_audit"]),
        ]
        subprocess.run(candidate_audit_command, check=True)
        audit = json.loads(artifacts["audit"].read_text(encoding="utf-8"))
        candidate_audit = json.loads(
            artifacts["candidate_audit"].read_text(encoding="utf-8")
        )
        route_results.append(
            {
                "route": f"{city}/{run}",
                "runtime": _shadow_runtime(artifacts["shadow"]),
                "audit": audit,
                "candidate_supply_audit": candidate_audit,
                "commands": {
                    "solver_without_truth": solver_command,
                    "tracker_without_truth": tracker_command,
                    "post_estimator_audit": audit_command,
                    "post_estimator_candidate_audit": candidate_audit_command,
                },
                "artifact_sha256": {
                    name: _sha256(path) for name, path in artifacts.items()
                },
                "pre_run_manifest": {
                    "path": str(pre_run_manifest),
                    "sha256": _sha256(pre_run_manifest),
                },
            }
        )

    result = {
        "schema": "gnss_gpu_ppc_basin_fgo_six_route_v1",
        "binary_sha256": _sha256(binary),
        "max_epochs": args.max_epochs,
        "skip_epochs": args.skip_epochs,
        "top_k": args.top_k,
        "cuda_mode": args.cuda_mode,
        "production_input_truth": False,
        "imu_enabled": args.imu,
        "native_imu_enabled": args.native_imu,
        "native_imu_aperture_m": args.native_imu_aperture,
        "native_imu_fix_min_streak": args.native_imu_fix_min_streak,
        "truth_usage": "separate_post_estimator_audit_subprocess_only",
        "routes": route_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
