#!/usr/bin/env python3
"""Run and score GNSS-only MultiSD FGO policies on the six PPC routes.

The solver subprocess receives rover/base/navigation RINEX only. Reference CSV
is opened only after the subprocess exits and is used exclusively for scoring.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROUTES = tuple(
    (city, f"run{run}")
    for city in ("tokyo", "nagoya")
    for run in range(1, 4)
)


@dataclass(frozen=True)
class Policy:
    name: str
    window: int
    minimum_epochs: int
    holdout_offset: int
    top_k: int
    maximum_seed_separation_m: float
    validation_history_epochs: int
    minimum_carrier_fraction: float
    minimum_fixed_ambiguities: int
    holdout_satellites: int
    constellation_ranked_par: bool
    candidate_ratio: float
    candidate_groups: int
    fallback_consensus_groups: int
    fallback_consensus_separation_m: float
    fallback_max_seed_separation_m: float
    quality_ranked_par: bool = False
    interleave_constellation_par: bool = False
    minimum_bootstrapped_success_rate: float = 0.0
    maximum_adop_cycles: float = 0.0
    fallback_minimum_bootstrapped_success_rate: float = 0.0


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference(path: Path) -> dict[float, tuple[float, float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, skipinitialspace=True)
        return {
            _tow(row["GPS TOW (s)"]): tuple(
                float(row[f"ECEF {axis} (m)"]) for axis in "XYZ"
            )
            for row in rows
        }


def read_solution_tows(path: Path) -> list[float]:
    return list(read_solutions(path))


def read_solutions(path: Path) -> dict[float, dict[str, float | int]]:
    output: dict[float, dict[str, float | int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"malformed POS row in {path}")
            tow = _tow(fields[1])
            output[tow] = {
                "x": float(fields[2]),
                "y": float(fields[3]),
                "z": float(fields[4]),
                "status": int(fields[8]),
            }
    return output


def read_shadow(path: Path) -> dict[float, dict[str, str]]:
    runtime_fields = {
        "build_runtime_ms",
        "optimize_wall_ms",
        "optimizer_cpu_ms",
        "runtime_ms",
    }
    with path.open(encoding="utf-8", newline="") as stream:
        output: dict[float, dict[str, str]] = {}
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            if row.get(None) or not row.get("tow") or not row.get("epoch_index"):
                raise ValueError(
                    f"malformed shadow row {line_number} in {path}"
                )
            tow = _tow(row["tow"])
            if tow in output:
                previous = output[tow]
                scientific_fields = set(row) - runtime_fields
                if any(previous.get(key) != row.get(key) for key in scientific_fields):
                    raise ValueError(
                        f"conflicting duplicate shadow TOW {tow} in {path}"
                    )
                # A killed parent process can leave its solver child finishing
                # the same final rows as a resumed run. Accept only an exact
                # scientific duplicate and retain the conservative runtime.
                for key in runtime_fields & set(row):
                    try:
                        previous[key] = str(
                            max(float(previous[key]), float(row[key]))
                        )
                    except (KeyError, TypeError, ValueError):
                        raise ValueError(
                            f"invalid duplicate runtime at TOW {tow} in {path}"
                        ) from None
                continue
            output[tow] = row
        return output


def artifacts_complete(
    pos_path: Path,
    shadow_path: Path,
    max_epochs: int,
) -> bool:
    try:
        if not read_solution_tows(pos_path):
            return False
        shadow = read_shadow(shadow_path)
        if not shadow:
            return False
        if max_epochs > 0:
            last_epoch = max(int(row["epoch_index"]) for row in shadow.values())
            if last_epoch != max_epochs - 1:
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def _score_tows(
    tows: list[float],
    shadow: dict[float, dict[str, str]],
    truth: dict[float, tuple[float, float, float]],
) -> dict[str, object]:
    fixed_errors: list[float] = []
    missing_truth_fixed = 0
    runtimes: list[float] = []
    for tow in tows:
        row = shadow.get(tow)
        if row is None:
            continue
        try:
            runtime = float(row["runtime_ms"])
        except (KeyError, TypeError, ValueError):
            runtime = math.nan
        if math.isfinite(runtime):
            runtimes.append(runtime)
        if row.get("shadow_fixed") != "1":
            continue
        reference = truth.get(tow)
        if reference is None:
            missing_truth_fixed += 1
            continue
        try:
            estimate = tuple(float(row[axis]) for axis in "xyz")
        except (KeyError, TypeError, ValueError):
            missing_truth_fixed += 1
            continue
        if not all(math.isfinite(value) for value in estimate):
            missing_truth_fixed += 1
            continue
        fixed_errors.append(math.dist(estimate, reference))

    correct = sum(error < 0.5 for error in fixed_errors)
    false = sum(error >= 0.5 for error in fixed_errors) + missing_truth_fixed
    above_1m = sum(error > 1.0 for error in fixed_errors) + missing_truth_fixed
    fixed = len(fixed_errors) + missing_truth_fixed
    epochs = len(tows)
    return {
        "epochs": epochs,
        "evaluated_epochs": sum(tow in shadow for tow in tows),
        "fixed_epochs": fixed,
        "correct_fixed_epochs": correct,
        "false_fixed_epochs": false,
        "false_fixed_above_1m_epochs": above_1m,
        "missing_truth_fixed_epochs": missing_truth_fixed,
        "correct_fix_rate": correct / epochs if epochs else 0.0,
        "false_per_fixed": false / fixed if fixed else 0.0,
        "fixed_error_p95_m": _quantile(fixed_errors, 0.95),
        "fixed_error_max_m": max(fixed_errors) if fixed_errors else None,
        "runtime_p95_ms": _quantile(runtimes, 0.95),
        "runtime_max_ms": max(runtimes) if runtimes else None,
    }


def _score_solution_authority(
    tows: list[float],
    solutions: dict[float, dict[str, float | int]],
    shadow: dict[float, dict[str, str]],
    truth: dict[float, tuple[float, float, float]],
    *,
    include_shadow_rescue: bool,
) -> dict[str, object]:
    errors: list[float] = []
    missing_truth_fixed = 0
    shadow_rescues = 0
    for tow in tows:
        solution = solutions[tow]
        estimate: tuple[float, float, float] | None = None
        if int(solution["status"]) == 4:
            estimate = tuple(float(solution[axis]) for axis in "xyz")
        elif include_shadow_rescue:
            row = shadow.get(tow)
            if row is not None and row.get("shadow_fixed") == "1":
                try:
                    estimate = tuple(float(row[axis]) for axis in "xyz")
                except (KeyError, TypeError, ValueError):
                    estimate = None
                shadow_rescues += 1
        if estimate is None:
            continue
        reference = truth.get(tow)
        if reference is None or not all(math.isfinite(value) for value in estimate):
            missing_truth_fixed += 1
            continue
        errors.append(math.dist(estimate, reference))
    correct = sum(error < 0.5 for error in errors)
    false = sum(error >= 0.5 for error in errors) + missing_truth_fixed
    above_1m = sum(error > 1.0 for error in errors) + missing_truth_fixed
    fixed = len(errors) + missing_truth_fixed
    epochs = len(tows)
    return {
        "epochs": epochs,
        "fixed_epochs": fixed,
        "correct_fixed_epochs": correct,
        "false_fixed_epochs": false,
        "false_fixed_above_1m_epochs": above_1m,
        "missing_truth_fixed_epochs": missing_truth_fixed,
        "shadow_rescue_epochs": shadow_rescues,
        "correct_fix_rate": correct / epochs if epochs else 0.0,
        "false_per_fixed": false / fixed if fixed else 0.0,
        "fixed_error_p95_m": _quantile(errors, 0.95),
        "fixed_error_max_m": max(errors) if errors else None,
    }


def score_artifact(
    city: str,
    run: str,
    policy: Policy,
    pos_path: Path,
    shadow_path: Path,
    reference_path: Path,
    *,
    block_count: int = 5,
) -> dict[str, object]:
    solutions = read_solutions(pos_path)
    tows = list(solutions)
    shadow = read_shadow(shadow_path)
    truth = read_reference(reference_path)
    route = _score_tows(tows, shadow, truth)
    baseline = _score_solution_authority(
        tows, solutions, shadow, truth, include_shadow_rescue=False
    )
    baseline_priority_union = _score_solution_authority(
        tows, solutions, shadow, truth, include_shadow_rescue=True
    )
    blocks = []
    for index in range(block_count):
        start = index * len(tows) // block_count
        stop = (index + 1) * len(tows) // block_count
        block_tows = tows[start:stop]
        metrics = _score_tows(block_tows, shadow, truth)
        union_metrics = _score_solution_authority(
            block_tows,
            solutions,
            shadow,
            truth,
            include_shadow_rescue=True,
        )
        blocks.append(
            {
                "block": index,
                **metrics,
                "baseline_priority_union": union_metrics,
            }
        )
    return {
        "city": city,
        "run": run,
        "policy": asdict(policy),
        "truth_usage": "post_solver_scoring_only",
        "pos_sha256": _sha256(pos_path),
        "shadow_sha256": _sha256(shadow_path),
        "route": route,
        "baseline": baseline,
        # Diagnostic only: baseline Status 4 has precedence, so this union is
        # not an integrity-safe authority when the baseline itself false-fixes.
        "baseline_priority_union": baseline_priority_union,
        "contiguous_time_blocks": blocks,
    }


def _aggregate(
    scores: list[dict[str, object]], route_key: str = "route"
) -> dict[str, object]:
    routes = [score[route_key] for score in scores]
    epochs = sum(int(route["epochs"]) for route in routes)  # type: ignore[index]
    fixed = sum(int(route["fixed_epochs"]) for route in routes)  # type: ignore[index]
    correct = sum(
        int(route["correct_fixed_epochs"]) for route in routes  # type: ignore[index]
    )
    false = sum(
        int(route["false_fixed_epochs"]) for route in routes  # type: ignore[index]
    )
    above_1m = sum(
        int(route["false_fixed_above_1m_epochs"])  # type: ignore[index]
        for route in routes
    )
    return {
        "epochs": epochs,
        "fixed_epochs": fixed,
        "correct_fixed_epochs": correct,
        "false_fixed_epochs": false,
        "false_fixed_above_1m_epochs": above_1m,
        "correct_fix_rate": correct / epochs if epochs else 0.0,
        "false_per_fixed": false / fixed if fixed else 0.0,
    }


def nested_leave_one_run_out(
    scores: list[dict[str, object]],
    *,
    stratify_city: bool = False,
) -> dict[str, object]:
    policies = sorted(
        {str(score["policy"]["name"]) for score in scores}  # type: ignore[index]
    )
    by_key = {
        (
            str(score["city"]),
            str(score["run"]),
            str(score["policy"]["name"]),  # type: ignore[index]
        ): score
        for score in scores
    }
    folds = []
    selected_holdouts = []
    for holdout_city, holdout_run in ROUTES:
        training_routes = [
            (city, run)
            for city, run in ROUTES
            if (city, run) != (holdout_city, holdout_run)
            and (not stratify_city or city == holdout_city)
        ]
        ranked = []
        for policy in policies:
            if (holdout_city, holdout_run, policy) not in by_key:
                continue
            training = [
                by_key[(city, run, policy)]
                for city, run in training_routes
                if (city, run, policy) in by_key
            ]
            if len(training) != len(training_routes):
                continue
            aggregate = _aggregate(training)
            training_cities = sorted({str(score["city"]) for score in training})
            city_rates = [
                float(
                    _aggregate(
                        [score for score in training if score["city"] == city]
                    )["correct_fix_rate"]
                )
                for city in training_cities
            ]
            block_false = max(
                int(block["false_fixed_epochs"])
                for score in training
                for block in score["contiguous_time_blocks"]  # type: ignore[index]
            )
            ranked.append(
                (
                    int(aggregate["false_fixed_above_1m_epochs"]) != 0,
                    int(aggregate["false_fixed_epochs"]) != 0,
                    block_false != 0,
                    float(aggregate["false_per_fixed"]),
                    -min(city_rates),
                    -float(aggregate["correct_fix_rate"]),
                    policy,
                    aggregate,
                )
            )
        if not ranked:
            continue
        selected = min(ranked)
        policy = selected[-2]
        holdout = by_key[(holdout_city, holdout_run, policy)]
        selected_holdouts.append(holdout)
        folds.append(
            {
                "holdout_city": holdout_city,
                "holdout_run": holdout_run,
                "selected_policy": policy,
                "training": selected[-1],
                "holdout": holdout["route"],
            }
        )
    city_aggregates = {
        city: _aggregate([score for score in selected_holdouts if score["city"] == city])
        for city in ("tokyo", "nagoya")
    }
    return {
        "selection": (
            ("city-stratified " if stratify_city else "") +
            "outer leave-one-run-out; training policies ranked by zero >1m "
            "false, zero false, zero contiguous-block false, false/FIX, "
            "worst-city correct FIX, then aggregate correct FIX"
        ),
        "folds": folds,
        "aggregate": _aggregate(selected_holdouts),
        "cities": city_aggregates,
        "complete": len(folds) == len(ROUTES),
    }


def _parse_policy(raw: str) -> Policy:
    fields = raw.split(":")
    if len(fields) not in (16, 17, 18, 20, 21):
        raise argparse.ArgumentTypeError(
            "policy must be NAME:WINDOW:MIN_EPOCHS:HOLDOUT_OFFSET:TOP_K:"
            "MAX_SEED_M:HISTORY:MIN_CARRIER_FRACTION:MIN_FIXED_AMBIGUITIES:"
            "HOLDOUT_SATELLITES:CONSTELLATION_PAR_0_OR_1:CANDIDATE_RATIO:"
            "CANDIDATE_GROUPS:FALLBACK_CONSENSUS_GROUPS:"
            "FALLBACK_CONSENSUS_SEPARATION_M:FALLBACK_MAX_SEED_M:"
            "QUALITY_RANKED_PAR_0_OR_1:INTERLEAVE_CONSTELLATION_PAR_0_OR_1:"
            "MIN_BSR:MAX_ADOP_CYCLES:FALLBACK_MIN_BSR"
        )
    if fields[10] not in ("0", "1"):
        raise argparse.ArgumentTypeError(
            "CONSTELLATION_PAR_0_OR_1 must be 0 or 1"
        )
    if len(fields) >= 17 and fields[16] not in ("0", "1"):
        raise argparse.ArgumentTypeError(
            "QUALITY_RANKED_PAR_0_OR_1 must be 0 or 1"
        )
    if len(fields) >= 18 and fields[17] not in ("0", "1"):
        raise argparse.ArgumentTypeError(
            "INTERLEAVE_CONSTELLATION_PAR_0_OR_1 must be 0 or 1"
        )
    policy = Policy(
        fields[0],
        *(int(value) for value in fields[1:5]),
        float(fields[5]),
        int(fields[6]),
        float(fields[7]),
        int(fields[8]),
        int(fields[9]),
        bool(int(fields[10])),
        float(fields[11]),
        int(fields[12]),
        int(fields[13]),
        float(fields[14]),
        float(fields[15]),
        bool(int(fields[16])) if len(fields) >= 17 else False,
        bool(int(fields[17])) if len(fields) >= 18 else False,
        float(fields[18]) if len(fields) >= 20 else 0.0,
        float(fields[19]) if len(fields) >= 20 else 0.0,
        float(fields[20]) if len(fields) == 21 else 0.0,
    )
    if (
        policy.window < 2
        or policy.minimum_epochs < 2
        or policy.minimum_epochs > policy.window
        or policy.holdout_offset < 0
        or not 2 <= policy.top_k <= 32
        or not math.isfinite(policy.maximum_seed_separation_m)
        or policy.maximum_seed_separation_m < 0.0
        or policy.validation_history_epochs < 1
        or not math.isfinite(policy.minimum_carrier_fraction)
        or not 0.0 < policy.minimum_carrier_fraction <= 1.0
        or not 2 <= policy.minimum_fixed_ambiguities <= 16
        or not 2 <= policy.holdout_satellites <= 16
        or not math.isfinite(policy.candidate_ratio)
        or policy.candidate_ratio < 1.0
        or not 1 <= policy.candidate_groups <= 32
        or not 1 <= policy.fallback_consensus_groups <= 32
        or not math.isfinite(policy.fallback_consensus_separation_m)
        or policy.fallback_consensus_separation_m < 0.0
        or (
            policy.fallback_consensus_groups > 1
            and policy.fallback_consensus_separation_m <= 0.0
        )
        or not math.isfinite(policy.fallback_max_seed_separation_m)
        or policy.fallback_max_seed_separation_m < 0.0
        or not math.isfinite(policy.minimum_bootstrapped_success_rate)
        or not 0.0 <= policy.minimum_bootstrapped_success_rate <= 1.0
        or not math.isfinite(policy.maximum_adop_cycles)
        or policy.maximum_adop_cycles < 0.0
        or not math.isfinite(
            policy.fallback_minimum_bootstrapped_success_rate
        )
        or not 0.0 <= policy.fallback_minimum_bootstrapped_success_rate <= 1.0
    ):
        raise argparse.ArgumentTypeError(f"invalid policy: {raw}")
    return policy


def _run_one(
    binary: Path,
    data_root: Path,
    output_dir: Path,
    city: str,
    run: str,
    policy: Policy,
    max_epochs: int,
    cuda_mode: str,
    resume: bool,
    analyze_only: bool,
    tdcp_slip_repair: bool = False,
    tdcp_slip_repair_max_cycles: int = 32,
    tdcp_slip_repair_tolerance_cycles: float = 0.20,
    wcmc: bool = False,
    wcmc_warmup_epochs: int = 5,
    wcmc_baseline_alpha: float = 0.05,
    wcmc_min_correction_m: float = 0.5,
    wcmc_max_correction_m: float = 10.0,
    fallback_integer_aperture: bool = False,
    fallback_ia_covariance_scale: float = 16.0,
) -> tuple[Path, Path, list[str]]:
    route_dir = data_root / city / run
    stem = f"{city}_{run}_{policy.name}"
    pos_path = output_dir / f"{stem}.pos"
    shadow_path = output_dir / f"{stem}.shadow.csv"
    metadata_path = output_dir / f"{stem}.run.json"
    command = [
        str(binary),
        "--rover", str(route_dir / "rover.obs"),
        "--base", str(route_dir / "base.obs"),
        "--nav", str(route_dir / "base.nav"),
        "--preset", "low-cost",
        "--no-kml",
        "--out", str(pos_path),
        "--multisd-fgo-shadow-csv", str(shadow_path),
        "--multisd-fgo-shadow-window", str(policy.window),
        "--multisd-fgo-shadow-min-epochs", str(policy.minimum_epochs),
        "--multisd-fgo-shadow-holdout-offset", str(policy.holdout_offset),
        "--multisd-fgo-shadow-top-k", str(policy.top_k),
        "--multisd-fgo-shadow-max-seed-separation",
        str(policy.maximum_seed_separation_m),
        "--multisd-fgo-shadow-validation-history",
        str(policy.validation_history_epochs),
        "--multisd-fgo-shadow-min-carrier-fraction",
        str(policy.minimum_carrier_fraction),
        "--multisd-fgo-shadow-min-fixed-ambiguities",
        str(policy.minimum_fixed_ambiguities),
        "--multisd-fgo-shadow-holdout-satellites",
        str(policy.holdout_satellites),
    ]
    if max_epochs > 0:
        command.extend(("--max-epochs", str(max_epochs)))
    if policy.constellation_ranked_par:
        command.append("--multisd-fgo-shadow-constellation-par")
    if policy.quality_ranked_par:
        command.append("--multisd-fgo-shadow-quality-ranked-par")
    if policy.interleave_constellation_par:
        command.append("--multisd-fgo-shadow-interleave-constellation-par")
    command.extend(
        (
            "--multisd-fgo-shadow-candidate-ratio",
            str(policy.candidate_ratio),
            "--multisd-fgo-shadow-candidate-groups",
            str(policy.candidate_groups),
            "--multisd-fgo-shadow-fallback-consensus-groups",
            str(policy.fallback_consensus_groups),
            "--multisd-fgo-shadow-fallback-consensus-separation",
            str(policy.fallback_consensus_separation_m),
            "--multisd-fgo-shadow-fallback-max-seed-separation",
            str(policy.fallback_max_seed_separation_m),
            "--multisd-fgo-shadow-min-bsr",
            str(policy.minimum_bootstrapped_success_rate),
            "--multisd-fgo-shadow-max-adop",
            str(policy.maximum_adop_cycles),
            "--multisd-fgo-shadow-fallback-min-bsr",
            str(policy.fallback_minimum_bootstrapped_success_rate),
        )
    )
    if tdcp_slip_repair:
        command.extend(
            (
                "--multisd-fgo-shadow-tdcp-slip-repair",
                "--multisd-fgo-shadow-tdcp-slip-repair-max-cycles",
                str(tdcp_slip_repair_max_cycles),
                "--multisd-fgo-shadow-tdcp-slip-repair-tolerance",
                str(tdcp_slip_repair_tolerance_cycles),
            )
        )
    if wcmc:
        command.extend(
            (
                "--multisd-fgo-shadow-wcmc",
                "--multisd-fgo-shadow-wcmc-warmup",
                str(wcmc_warmup_epochs),
                "--multisd-fgo-shadow-wcmc-alpha",
                str(wcmc_baseline_alpha),
                "--multisd-fgo-shadow-wcmc-min-correction",
                str(wcmc_min_correction_m),
                "--multisd-fgo-shadow-wcmc-max-correction",
                str(wcmc_max_correction_m),
            )
        )
    if fallback_integer_aperture:
        command.extend(
            (
                "--multisd-fgo-shadow-fallback-integer-aperture",
                "--multisd-fgo-shadow-fallback-ia-covariance-scale",
                str(fallback_ia_covariance_scale),
            )
        )
    if analyze_only:
        if not artifacts_complete(pos_path, shadow_path, max_epochs):
            raise ValueError(
                f"cannot analyze incomplete artifact pair: {pos_path}, {shadow_path}"
            )
        return pos_path, shadow_path, command
    expected_metadata = {
        "command": command,
        "binary_sha256": _sha256(binary),
    }
    metadata_matches = False
    if metadata_path.is_file():
        try:
            metadata_matches = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                == expected_metadata
            )
        except (OSError, json.JSONDecodeError):
            metadata_matches = False
    complete = (
        resume
        and pos_path.is_file()
        and shadow_path.is_file()
        and artifacts_complete(pos_path, shadow_path, max_epochs)
        and metadata_matches
    )
    if not complete:
        environment = os.environ.copy()
        environment["GNSSPP_FGO_CUDA_SOLVER"] = cuda_mode
        subprocess.run(command, check=True, env=environment)
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return pos_path, shadow_path, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        type=_parse_policy,
        default=[],
        help=(
            "NAME:WINDOW:MIN_EPOCHS:HOLDOUT_OFFSET:TOP_K:MAX_SEED_M:"
            "HISTORY:MIN_CARRIER_FRACTION:MIN_FIXED_AMBIGUITIES:"
            "HOLDOUT_SATELLITES:CONSTELLATION_PAR_0_OR_1:CANDIDATE_RATIO:"
            "CANDIDATE_GROUPS:FALLBACK_CONSENSUS_GROUPS:"
            "FALLBACK_CONSENSUS_SEPARATION_M:FALLBACK_MAX_SEED_M:"
            "QUALITY_RANKED_PAR_0_OR_1:INTERLEAVE_CONSTELLATION_PAR_0_OR_1:"
            "MIN_BSR:MAX_ADOP_CYCLES:FALLBACK_MIN_BSR"
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--cuda-mode", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tdcp-slip-repair",
        action="store_true",
        help="enable fail-closed Doppler-conditioned integer SD-TDCP slip repair",
    )
    parser.add_argument("--tdcp-slip-repair-max-cycles", type=int, default=32)
    parser.add_argument(
        "--tdcp-slip-repair-tolerance-cycles", type=float, default=0.20
    )
    parser.add_argument("--wcmc", action="store_true")
    parser.add_argument("--wcmc-warmup-epochs", type=int, default=5)
    parser.add_argument("--wcmc-baseline-alpha", type=float, default=0.05)
    parser.add_argument("--wcmc-min-correction-m", type=float, default=0.5)
    parser.add_argument("--wcmc-max-correction-m", type=float, default=10.0)
    parser.add_argument("--fallback-integer-aperture", action="store_true")
    parser.add_argument(
        "--fallback-ia-covariance-scale", type=float, default=16.0
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="score existing complete artifacts without running or sidecar checks",
    )
    parser.add_argument(
        "--route",
        action="append",
        choices=tuple(f"{city}/{run}" for city, run in ROUTES),
    )
    args = parser.parse_args()
    if not 1 <= args.tdcp_slip_repair_max_cycles <= 10000:
        parser.error("--tdcp-slip-repair-max-cycles must be in [1, 10000]")
    if not math.isfinite(args.tdcp_slip_repair_tolerance_cycles) or not (
        0.0 < args.tdcp_slip_repair_tolerance_cycles <= 0.5
    ):
        parser.error(
            "--tdcp-slip-repair-tolerance-cycles must be in (0, 0.5]"
        )
    if args.wcmc_warmup_epochs < 1:
        parser.error("--wcmc-warmup-epochs must be >= 1")
    if not math.isfinite(args.wcmc_baseline_alpha) or not (
        0.0 <= args.wcmc_baseline_alpha <= 1.0
    ):
        parser.error("--wcmc-baseline-alpha must be in [0, 1]")
    if not math.isfinite(args.wcmc_max_correction_m) or not (
        args.wcmc_max_correction_m > 0.0
    ):
        parser.error("--wcmc-max-correction-m must be > 0")
    if not math.isfinite(args.wcmc_min_correction_m) or not (
        0.0 <= args.wcmc_min_correction_m <= args.wcmc_max_correction_m
    ):
        parser.error("--wcmc-min-correction-m must be in [0, max]")
    if not math.isfinite(args.fallback_ia_covariance_scale) or not (
        args.fallback_ia_covariance_scale > 0.0
    ):
        parser.error("--fallback-ia-covariance-scale must be > 0")
    policies = args.policy or [
        Policy(
            "locked_w10_o2_k4_s05_h3_f075_m6",
            10, 10, 2, 4, 0.5, 3, 0.75, 6, 4, False,
            1.5, 1, 1, 0.0, 0.0,
        )
    ]
    routes = (
        [tuple(route.split("/", 1)) for route in args.route]
        if args.route
        else list(ROUTES)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    commands = []
    for policy in policies:
        for city, run in routes:
            pos, shadow, command = _run_one(
                args.binary.resolve(),
                args.data_root.resolve(),
                args.output_dir.resolve(),
                city,
                run,
                policy,
                args.max_epochs,
                args.cuda_mode,
                args.resume,
                args.analyze_only,
                args.tdcp_slip_repair,
                args.tdcp_slip_repair_max_cycles,
                args.tdcp_slip_repair_tolerance_cycles,
                args.wcmc,
                args.wcmc_warmup_epochs,
                args.wcmc_baseline_alpha,
                args.wcmc_min_correction_m,
                args.wcmc_max_correction_m,
                args.fallback_integer_aperture,
                args.fallback_ia_covariance_scale,
            )
            commands.append(command)
            scores.append(
                score_artifact(
                    city,
                    run,
                    policy,
                    pos,
                    shadow,
                    args.data_root / city / run / "reference.csv",
                )
            )
    payload = {
        "schema": "gnss_gpu_multisd_fgo_ppc_nested_cv_v1",
        "estimator_inputs": "PPC rover.obs, base.obs, base.nav only",
        "excluded_estimator_inputs": ["imu", "lidar", "camera", "reference"],
        "truth_usage": "reference.csv opened only by post-subprocess scorer",
        "cuda_mode": args.cuda_mode,
        "max_epochs": args.max_epochs,
        "tdcp_slip_repair": {
            "enabled": args.tdcp_slip_repair,
            "maximum_cycles": args.tdcp_slip_repair_max_cycles,
            "tolerance_cycles": args.tdcp_slip_repair_tolerance_cycles,
        },
        "wcmc": {
            "enabled": args.wcmc,
            "warmup_epochs": args.wcmc_warmup_epochs,
            "baseline_alpha": args.wcmc_baseline_alpha,
            "minimum_correction_m": args.wcmc_min_correction_m,
            "maximum_correction_m": args.wcmc_max_correction_m,
        },
        "fallback_integer_aperture": {
            "enabled": args.fallback_integer_aperture,
            "covariance_scale": args.fallback_ia_covariance_scale,
            "tolerable_failure_rate": 0.001,
        },
        "commands": commands,
        "scores": scores,
        "nested_leave_one_run_out": nested_leave_one_run_out(scores),
        "nested_city_leave_one_run_out": nested_leave_one_run_out(
            scores, stratify_city=True
        ),
    }
    output_path = args.output_dir / "audit.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload["nested_leave_one_run_out"], indent=2))
    print(f"audit: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
