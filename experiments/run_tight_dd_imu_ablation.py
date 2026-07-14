#!/usr/bin/env python3
"""Run resumable baseline vs tight-DD/IMU PPC ablations through gnss_fuse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")
DEFAULT_MANIFEST = Path(__file__).with_name("blocked_span_manifest.csv")
RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)
VARIANTS = {"baseline": (), "tight_dd_imu": ("--tight-dd-imu",)}


def _binary_provenance(binary: str, *, use_wsl: bool) -> dict[str, object]:
    """Fingerprint the exact executable used for a matched ablation."""
    if use_wsl:
        return {"binary_path": str(binary), "binary_sha256": None}
    path = Path(binary).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "binary_path": str(path),
        "binary_sha256": digest.hexdigest(),
        "binary_mtime_ns": stat.st_mtime_ns,
        "binary_size_bytes": stat.st_size,
    }


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            times.append(round(float(row[0]), 2))
            positions.append([float(row[5]), float(row[6]), float(row[7])])
    return np.asarray(times), np.asarray(positions, dtype=np.float64)


def _load_positions(path: Path) -> dict[float, np.ndarray]:
    positions: dict[float, np.ndarray] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            positions[round(float(fields[1]), 2)] = np.asarray(
                [float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64
            )
    return positions


def _diagnostic_int(log: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s*(\d+)", log, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0


def _summarize(
    position_path: Path,
    reference_path: Path,
    log: str,
    runtime_s: float,
    start: int,
    end: int | None,
    *,
    include_diagnostics: bool = True,
) -> dict[str, object]:
    times, reference = _load_reference(reference_path)
    stop = reference.shape[0] if end is None else min(end, reference.shape[0])
    times = times[start:stop]
    reference = reference[start:stop]
    output = _load_positions(position_path)
    estimated = np.full_like(reference, np.nan)
    emitted = np.zeros(reference.shape[0], dtype=bool)
    for index, tow in enumerate(times):
        if float(tow) in output:
            estimated[index] = output[float(tow)]
            emitted[index] = True
    errors = np.linalg.norm(estimated[emitted] - reference[emitted], axis=1)
    honest_errors = np.full(reference.shape[0], np.inf, dtype=np.float64)
    honest_errors[emitted] = errors
    distances = np.zeros(reference.shape[0], dtype=np.float64)
    if reference.shape[0] > 1:
        distances[1:] = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    total_distance = float(distances.sum())
    honest_pass_distance = float(distances[honest_errors <= 0.5].sum())
    result: dict[str, object] = {
        "requested_epochs": int(reference.shape[0]),
        "emitted_epochs": int(emitted.sum()),
        "coverage": float(emitted.mean()) if emitted.size else 0.0,
        "honest_ppc_score_pct": (
            100.0 * honest_pass_distance / total_distance if total_distance > 0.0 else 0.0
        ),
        "pass_distance_m": honest_pass_distance,
        "total_distance_m": total_distance,
        "runtime_s": float(runtime_s),
        "runtime_ms_per_requested_epoch": (
            1000.0 * runtime_s / reference.shape[0] if reference.shape[0] else float("nan")
        ),
        "tight_dd_epochs": (
            _diagnostic_int(log, "Tight DD/IMU epochs") if include_diagnostics else None
        ),
        "tight_dd_rows": _diagnostic_int(log, "Tight DD rows") if include_diagnostics else None,
        "carrier_to_code_fallbacks": (
            _diagnostic_int(log, "Carrier-to-code fallbacks")
            if include_diagnostics
            else None
        ),
        "tight_dd_soft_resets": (
            _diagnostic_int(log, "Innovation-gated soft resets")
            if include_diagnostics
            else None
        ),
    }
    tight_match = re.search(
        r"^Tight DD/IMU epochs:\s*\d+\s*\(accepted=(\d+),\s*innovation_rejected=(\d+)\)",
        log,
        flags=re.MULTILINE,
    )
    par_match = re.search(
        r"^Partial-AR epochs/fixed ambiguities:\s*(\d+)/(\d+)",
        log,
        flags=re.MULTILINE,
    )
    result["tight_dd_accepted"] = (
        int(tight_match.group(1)) if include_diagnostics and tight_match else None
    )
    result["tight_dd_rejected"] = (
        int(tight_match.group(2)) if include_diagnostics and tight_match else None
    )
    result["partial_ar_epochs"] = (
        int(par_match.group(1)) if include_diagnostics and par_match else None
    )
    result["fixed_ambiguities"] = (
        int(par_match.group(2)) if include_diagnostics and par_match else None
    )
    if include_diagnostics:
        result["tight_dd_accepted"] = result["tight_dd_accepted"] or 0
        result["tight_dd_rejected"] = result["tight_dd_rejected"] or 0
        result["partial_ar_epochs"] = result["partial_ar_epochs"] or 0
        result["fixed_ambiguities"] = result["fixed_ambiguities"] or 0
    if errors.size:
        result.update(
            {
                "pass_0_5m": float(np.mean(errors <= 0.5)),
                "pass_1m": float(np.mean(errors <= 1.0)),
                "pass_3m": float(np.mean(errors <= 3.0)),
                "error_p50_m": float(np.quantile(errors, 0.50)),
                "error_p95_m": float(np.quantile(errors, 0.95)),
                "error_p99_m": float(np.quantile(errors, 0.99)),
            }
        )
    return result


def _run_one(
    *,
    binary: str,
    use_wsl: bool,
    data_dir: Path,
    output_path: Path,
    log_path: Path,
    max_epochs: int,
    variant: str,
) -> float:
    def path_arg(path: Path) -> str:
        return _wsl_path(path) if use_wsl else str(path.resolve())

    args = [
        binary,
        "--rover",
        path_arg(data_dir / "rover.obs"),
        "--base",
        path_arg(data_dir / "base.obs"),
        "--nav",
        path_arg(data_dir / "base.nav"),
        "--imu",
        path_arg(data_dir / "imu.csv"),
        "--preset",
        "low-cost",
        "--out",
        path_arg(output_path),
        *VARIANTS[variant],
    ]
    if max_epochs > 0:
        args.extend(("--max-epochs", str(max_epochs)))
    command = ["wsl", "-e", *args] if use_wsl else args
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=REPO, check=True, text=True, capture_output=True, encoding="utf-8"
    )
    runtime_s = time.perf_counter() - started
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return runtime_s


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "scope_id",
        "city",
        "run",
        "evaluation_role",
        "variant",
        "binary_path",
        "binary_sha256",
        "binary_mtime_ns",
        "binary_size_bytes",
        "diagnostics_scope",
        "runtime_scope",
        "requested_epochs",
        "emitted_epochs",
        "coverage",
        "honest_ppc_score_pct",
        "pass_distance_m",
        "total_distance_m",
        "pass_0_5m",
        "pass_1m",
        "pass_3m",
        "error_p50_m",
        "error_p95_m",
        "error_p99_m",
        "tight_dd_epochs",
        "tight_dd_accepted",
        "tight_dd_rejected",
        "tight_dd_rows",
        "carrier_to_code_fallbacks",
        "partial_ar_epochs",
        "fixed_ambiguities",
        "tight_dd_soft_resets",
        "runtime_s",
        "runtime_ms_per_requested_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    metrics = (
        "honest_ppc_score_pct",
        "coverage",
        "pass_0_5m",
        "error_p50_m",
        "error_p95_m",
        "error_p99_m",
        "runtime_ms_per_requested_epoch",
    )
    by_scope = {(str(row["scope_id"]), str(row["variant"])): row for row in rows}
    comparisons: list[dict[str, object]] = []
    for scope_id in sorted({str(row["scope_id"]) for row in rows}):
        baseline = by_scope.get((scope_id, "baseline"))
        tight = by_scope.get((scope_id, "tight_dd_imu"))
        if baseline is None or tight is None:
            continue
        binary_match = (
            baseline.get("binary_sha256") == tight.get("binary_sha256")
            and baseline.get("binary_sha256") not in {None, ""}
        )
        comparison: dict[str, object] = {
            "scope_id": scope_id,
            "city": tight["city"],
            "run": tight["run"],
            "evaluation_role": tight["evaluation_role"],
            "comparison_status": "matched" if binary_match else "binary_mismatch",
            "binary_sha256_match": binary_match,
        }
        for metric in metrics:
            baseline_value = baseline.get(metric, "")
            tight_value = tight.get(metric, "")
            comparison[f"baseline_{metric}"] = baseline_value
            comparison[f"tight_{metric}"] = tight_value
            comparison[f"tight_minus_baseline_{metric}"] = (
                float(tight_value) - float(baseline_value)
                if binary_match and baseline_value != "" and tight_value != ""
                else ""
            )
        comparisons.append(comparison)
    fields = list(comparisons[0]) if comparisons else ["scope_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="/tmp/gnsspp-tight-build/apps/gnss_fuse")
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scope", choices=("full", "blocked", "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/results/tight_dd_imu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-scope",
        help="evaluate just one scope_id for a targeted reproducibility refresh",
    )
    parser.add_argument(
        "--rerun-baseline",
        action="store_true",
        help="refresh baseline results while preserving completed tight results",
    )
    parser.add_argument(
        "--rerun-tight",
        action="store_true",
        help="refresh tight_dd_imu results while preserving completed baselines",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    binary_provenance = _binary_provenance(args.binary, use_wsl=args.wsl)

    scopes: list[dict[str, object]] = []
    if args.scope in {"full", "all"}:
        scopes.extend(
            {
                "scope_id": f"{city}_{run}_full",
                "city": city,
                "run": run,
                "start": 0,
                "end": None,
                "evaluation_role": "development" if (city, run) == ("tokyo", "run1") else "holdout",
            }
            for city, run in RUNS
        )
    if args.scope in {"blocked", "all"}:
        for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8")):
            scopes.append(
                {
                    "scope_id": row["span_id"],
                    "city": row["city"],
                    "run": row["run"],
                    "start": int(row["start_epoch"]),
                    "end": int(row["end_epoch_exclusive"]),
                    "evaluation_role": row.get("tcfgo_evaluation_role", row["evaluation_role"]),
                }
            )

    if args.only_scope:
        scopes = [scope for scope in scopes if scope["scope_id"] == args.only_scope]
        if not scopes:
            raise ValueError(f"unknown --only-scope: {args.only_scope}")

    summaries: list[dict[str, object]] = []
    for scope in scopes:
        data_dir = args.data_root / str(scope["city"]) / str(scope["run"])
        for variant in VARIANTS:
            stem = f"{scope['scope_id']}_{variant}"
            position_path = args.out_dir / f"{stem}.pos"
            log_path = args.out_dir / f"{stem}.log"
            summary_path = args.out_dir / f"{stem}.json"
            end = scope["end"]
            if (
                args.force
                or (args.rerun_baseline and variant == "baseline")
                or (args.rerun_tight and variant == "tight_dd_imu")
                or not summary_path.exists()
            ):
                print(f"[{scope['scope_id']}/{variant}]", flush=True)
                full_stem = f"{scope['city']}_{scope['run']}_full_{variant}"
                full_position_path = args.out_dir / f"{full_stem}.pos"
                full_summary_path = args.out_dir / f"{full_stem}.json"
                reuse_full = (
                    not args.force
                    and end is not None
                    and full_position_path.exists()
                    and full_summary_path.exists()
                )
                if reuse_full:
                    full_payload = json.loads(full_summary_path.read_text(encoding="utf-8"))
                    requested = int(end) - int(scope["start"])
                    runtime_s = (
                        float(full_payload["runtime_ms_per_requested_epoch"])
                        * requested
                        / 1000.0
                    )
                    payload = _summarize(
                        full_position_path,
                        data_dir / "reference.csv",
                        "",
                        runtime_s,
                        int(scope["start"]),
                        int(end),
                        include_diagnostics=False,
                    )
                    payload["diagnostics_scope"] = "unavailable_from_full_position_slice"
                    payload["runtime_scope"] = "scaled_full_run_average"
                else:
                    runtime_s = _run_one(
                        binary=args.binary,
                        use_wsl=args.wsl,
                        data_dir=data_dir,
                        output_path=position_path,
                        log_path=log_path,
                        max_epochs=0 if end is None else int(end),
                        variant=variant,
                    )
                    payload = _summarize(
                        position_path,
                        data_dir / "reference.csv",
                        log_path.read_text(encoding="utf-8"),
                        runtime_s,
                        int(scope["start"]),
                        None if end is None else int(end),
                    )
                    payload["diagnostics_scope"] = "evaluated_scope"
                    payload["runtime_scope"] = "measured_wall_clock"
                payload.update(binary_provenance)
                summary_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            else:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                # Results written after this executable can be stamped without
                # rerunning. Pre-build results remain deliberately unmatched.
                binary_mtime = binary_provenance.get("binary_mtime_ns")
                if (
                    not payload.get("binary_sha256")
                    and binary_mtime is not None
                    and position_path.exists()
                    and position_path.stat().st_mtime_ns >= int(binary_mtime)
                ):
                    payload.update(binary_provenance)
                    summary_path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            payload.update(
                {
                    "scope_id": scope["scope_id"],
                    "city": scope["city"],
                    "run": scope["run"],
                    "evaluation_role": scope["evaluation_role"],
                    "variant": variant,
                }
            )
            summaries.append(payload)

    output = args.out_dir / "tight_dd_imu_ablation_summary.csv"
    _write_csv(output, summaries)
    comparison = args.out_dir / "tight_dd_imu_ablation_comparison.csv"
    _write_comparison(comparison, summaries)
    print(f"saved: {output}")
    print(f"saved: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
