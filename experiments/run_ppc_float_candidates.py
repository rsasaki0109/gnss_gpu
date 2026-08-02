#!/usr/bin/env python3
"""Generate frozen, truth-free gnss_fuse FLOAT candidates for PPC routes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROUTES = tuple(
    f"{city}_run{run_number}"
    for city in ("tokyo", "nagoya")
    for run_number in range(1, 4)
)
LEVER_ARMS = {
    "tokyo": "0.31,0,0.55",
    "nagoya": "0.593,0.670,1.216",
}
FROZEN_POLICY_ARGUMENTS = (
    "--preset",
    "low-cost",
    "--library-fix-integrity-gate",
    "--integrity-disjoint-ensemble",
    "--integrity-satellite-par-consensus-promotion",
    "--integrity-satellite-par-surplus-validation",
    "--integrity-satellite-par-surplus-min-fixed-pairs",
    "8",
    "--integrity-satellite-par-surplus-aperture-lt1",
    "0.1",
    "--integrity-satellite-par-surplus-aperture-1to2",
    "0.1",
    "--integrity-satellite-par-surplus-aperture-gt2",
    "0.1",
    "--integrity-satellite-par-acquisition-streak",
    "1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(
    binary: Path,
    dataset_root: Path,
    output_root: Path,
    route: str,
    max_epochs: int = 0,
) -> tuple[list[str], dict[str, Path]]:
    if route not in ROUTES:
        raise ValueError(f"unsupported route {route}")
    city, run = route.split("_", maxsplit=1)
    route_dir = dataset_root / city / run
    route_output = output_root / route
    artifacts = {
        "position": route_output / "float_candidate.pos",
        "integrity": route_output / "float_candidate_integrity.csv",
        "stdout": route_output / "gnss_fuse.stdout.log",
        "stderr": route_output / "gnss_fuse.stderr.log",
        "pre_manifest": route_output / "pre_run_manifest.json",
        "manifest": route_output / "run_manifest.json",
    }
    command = [
        str(binary),
        "--data-dir",
        str(route_dir),
        "--lever-arm",
        LEVER_ARMS[city],
        "--out",
        str(artifacts["position"]),
        "--library-fix-integrity-csv",
        str(artifacts["integrity"]),
        *FROZEN_POLICY_ARGUMENTS,
    ]
    if max_epochs > 0:
        command.extend(("--max-epochs", str(max_epochs)))
    return command, artifacts


def _run_route(
    binary: Path,
    dataset_root: Path,
    output_root: Path,
    route: str,
    max_epochs: int,
) -> dict[str, Any]:
    command, artifacts = build_command(
        binary, dataset_root, output_root, route, max_epochs
    )
    artifacts["position"].parent.mkdir(parents=True, exist_ok=True)
    city, run = route.split("_", maxsplit=1)
    route_dir = dataset_root / city / run
    inputs = {
        name: route_dir / filename
        for name, filename in (
            ("rover", "rover.obs"),
            ("base", "base.obs"),
            ("navigation", "base.nav"),
            ("imu", "imu.csv"),
        )
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{route} missing inputs: {', '.join(missing)}")
    binary_hash = _sha256(binary)
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    pre_manifest = {
        "schema": "gnss_gpu_ppc_float_candidate_pre_run_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "forward_only": True,
        "route": route,
        "command": command,
        "binary_sha256": binary_hash,
        "input_sha256": input_hashes,
    }
    artifacts["pre_manifest"].write_text(
        json.dumps(pre_manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (
        artifacts["stdout"].open("w", encoding="utf-8") as stdout,
        artifacts["stderr"].open("w", encoding="utf-8") as stderr,
    ):
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)
    for name in ("position", "integrity"):
        if not artifacts[name].is_file() or artifacts[name].stat().st_size == 0:
            raise RuntimeError(f"{route} did not produce {name}")
    result = {
        "schema": "gnss_gpu_ppc_float_candidate_run_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "forward_only": True,
        "route": route,
        "command": command,
        "binary_sha256": binary_hash,
        "input_sha256": input_hashes,
        "pre_run_manifest_sha256": _sha256(artifacts["pre_manifest"]),
        "output_sha256": {
            name: _sha256(artifacts[name]) for name in ("position", "integrity")
        },
    }
    artifacts["manifest"].write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256")
    parser.add_argument("--route", action="append", choices=ROUTES)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.binary.is_file():
        parser.error("binary does not exist")
    if not args.dataset_root.is_dir():
        parser.error("dataset root does not exist")
    if args.max_epochs < 0:
        parser.error("max epochs must be non-negative")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if (
        args.expected_binary_sha256 is not None
        and _sha256(args.binary).lower() != args.expected_binary_sha256.lower()
    ):
        parser.error("binary SHA-256 does not match the frozen executable")
    routes = tuple(args.route or ROUTES)
    if len(set(routes)) != len(routes):
        parser.error("routes must be unique")

    def run(route: str) -> dict[str, Any]:
        return _run_route(
            args.binary.resolve(),
            args.dataset_root.resolve(),
            args.output_root.resolve(),
            route,
            args.max_epochs,
        )

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(routes))) as executor:
        results = list(executor.map(run, routes))
    summary = {
        "schema": "gnss_gpu_ppc_float_candidate_suite_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "forward_only": True,
        "routes": results,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
