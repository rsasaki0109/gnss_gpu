"""Fail-closed evaluation contracts for urban-navigation promotion.

This module deliberately contains no estimator logic.  It defines the stable
campaign vocabulary, immutable negative holdouts, and reproducibility records
that every selector/estimator must satisfy before a production promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA = "gnss_gpu_reproducibility_manifest_v1"
EVALUATION_INPUT_SCHEMA = "gnss_gpu_campaign_evaluation_input_v1"
EVALUATION_RESULT_SCHEMA = "gnss_gpu_campaign_evaluation_result_v1"


class FailureCategory(str, Enum):
    """Stable, mutually useful failure categories for campaign reporting."""

    OBSERVATION_NLOS_MULTIPATH = "observation_nlos_multipath"
    BASIN_IDENTITY = "basin_identity"
    OFFSET_OR_DRIFT_MODEL = "offset_or_drift_model"
    EVIDENCE_THINNING = "evidence_thinning"
    OUTAGE_OR_REACQUISITION = "outage_or_reacquisition"
    MAP_CONSTRAINT_MISLEAD = "map_constraint_mislead"
    MISSING_EVIDENCE = "missing_evidence"
    UNSAFE_ACCEPTANCE = "unsafe_acceptance"
    RUNTIME_OR_RESOURCE = "runtime_or_resource"
    DATA_INTEGRITY = "data_integrity"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HoldoutSpec:
    holdout_id: str
    city: str
    dataset: str
    segment: tuple[int, int]
    failure_category: FailureCategory
    expected_disposition: str
    lock_path: str
    lock_schema: str
    lock_sha256: str


MANDATORY_NEGATIVE_HOLDOUTS: tuple[HoldoutSpec, ...] = (
    HoldoutSpec(
        holdout_id="nagoya_wp53",
        city="nagoya",
        dataset="nagoya_run1",
        segment=(1436, 1656),
        failure_category=FailureCategory.MISSING_EVIDENCE,
        expected_disposition="abstained",
        lock_path="internal_docs/wp53_alternate_reference_supply_posterior_rejection_2026_07_22.json",
        lock_schema="wp53_alternate_reference_supply_posterior_rejection_lock_v1",
        lock_sha256="82DB26EEFB298A7A9EA5C8F8DA23871183E869B838C877C9A091793D84728E1C",
    ),
    HoldoutSpec(
        holdout_id="tokyo_wp129",
        city="tokyo",
        dataset="tokyo_run1",
        segment=(5225, 5280),
        failure_category=FailureCategory.BASIN_IDENTITY,
        expected_disposition="rejected",
        lock_path="internal_docs/wp129_tokyo_posterior_rejection_2026_07_23.json",
        lock_schema="wp129_tokyo_constant_singlebasis_posterior_rejection_lock_v1",
        lock_sha256="770FFB13929B80810874F3530C14B02D72FDAE78257B9A1C4E01F24A162DEB9B",
    ),
    HoldoutSpec(
        holdout_id="tokyo_wp156",
        city="tokyo",
        dataset="tokyo_run1",
        segment=(10890, 10945),
        failure_category=FailureCategory.UNSAFE_ACCEPTANCE,
        expected_disposition="rejected",
        lock_path="internal_docs/wp156_tokyo_zero_gain_acceptance_rejection_2026_07_23.json",
        lock_schema="wp156_tokyo_zero_gain_acceptance_rejection_lock_v1",
        lock_sha256="E300D66325977E3F2585D77228EF76940C2D199CF826347936734A08EA5DD5B4",
    ),
    HoldoutSpec(
        holdout_id="tokyo_wp168",
        city="tokyo",
        dataset="tokyo_run1",
        segment=(1320, 1375),
        failure_category=FailureCategory.UNSAFE_ACCEPTANCE,
        expected_disposition="rejected",
        lock_path="internal_docs/wp171_tokyo_screened_zero_gain_acceptance_rejection_2026_07_24.json",
        lock_schema="wp171_tokyo_screened_zero_gain_acceptance_rejection_lock_v1",
        lock_sha256="3711AAA219F92156217F7E33A9DF56BE3FE04337A203ECF97068882C1B4D28F8",
    ),
)

M4_PRESERVED_SHA256: Mapping[str, str] = {
    "internal_docs/wp30_m4_production_config.json": (
        "66A5FF3F1919C4B0F9ED95A5EFA38865B518C9E03E6FD2652B7A0456A1F89486"
    ),
    "internal_docs/wp30_m4_tokyo_evidence_ledger.json": (
        "9D756F447304C30B73694225F1CEEA1A82DE864F8D968D449928662582DF098C"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def verify_locked_contract(repo_root: Path) -> dict[str, Any]:
    """Verify the mandatory holdout and M4 files byte-for-byte."""

    checks: list[dict[str, Any]] = []
    for spec in MANDATORY_NEGATIVE_HOLDOUTS:
        path = repo_root / spec.lock_path
        actual_hash = sha256_file(path) if path.is_file() else None
        schema = None
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema = payload.get("schema") if isinstance(payload, dict) else None
        passed = actual_hash == spec.lock_sha256 and schema == spec.lock_schema
        checks.append(
            {
                "id": spec.holdout_id,
                "path": spec.lock_path,
                "expected_sha256": spec.lock_sha256,
                "actual_sha256": actual_hash,
                "expected_schema": spec.lock_schema,
                "actual_schema": schema,
                "passed": passed,
            }
        )

    m4_checks: list[dict[str, Any]] = []
    for relative, expected in M4_PRESERVED_SHA256.items():
        path = repo_root / relative
        actual = sha256_file(path) if path.is_file() else None
        m4_checks.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )
    return {
        "passed": all(item["passed"] for item in checks + m4_checks),
        "mandatory_negative_holdouts": checks,
        "m4": m4_checks,
    }


def build_reproducibility_manifest(
    *,
    repo_root: Path,
    input_paths: Iterable[Path],
    config: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    """Create a deterministic, verifiable description of an evaluation run."""

    resolved_inputs: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path) for path in input_paths), key=lambda item: str(item)):
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"evaluation input does not exist: {path}")
        resolved_inputs.append(
            {
                "path": _repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "tracked_worktree_clean": status == "" if status is not None else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "command": list(command),
        "config": dict(config),
        "config_sha256": _canonical_sha256(config),
        "inputs": resolved_inputs,
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def verify_reproducibility_manifest(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for item in manifest.get("inputs", []):
        relative = item.get("path")
        path = Path(relative)
        if not path.is_absolute():
            path = repo_root / path
        actual = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": relative,
                "expected_sha256": item.get("sha256"),
                "actual_sha256": actual,
                "passed": actual == item.get("sha256"),
            }
        )
    expected_content = manifest.get("content_sha256")
    without_content = dict(manifest)
    without_content.pop("content_sha256", None)
    content_ok = expected_content == _canonical_sha256(without_content)
    return {
        "passed": (
            manifest.get("schema") == MANIFEST_SCHEMA
            and bool(checks)
            and all(item["passed"] for item in checks)
            and content_ok
        ),
        "content_hash_passed": content_ok,
        "inputs": checks,
    }


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_campaign(payload: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Evaluate a normalized campaign summary; missing evidence always fails."""

    if payload.get("schema") != EVALUATION_INPUT_SCHEMA:
        raise ValueError(f"expected schema {EVALUATION_INPUT_SCHEMA!r}")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    holdouts = payload.get("holdouts")
    if not isinstance(holdouts, Mapping):
        holdouts = {}

    locked = verify_locked_contract(repo_root)
    gates = [
        _gate(
            "truth_free_production_input",
            candidate.get("production_input_truth") is False,
            "production input must not use truth",
        ),
        _gate(
            "positive_gain",
            isinstance(candidate.get("gained_epochs"), int) and candidate["gained_epochs"] > 0,
            "gained_epochs must be an integer greater than zero",
        ),
        _gate(
            "no_loss",
            candidate.get("lost_epochs") == 0,
            "lost_epochs must be zero",
        ),
        _gate(
            "false_fix_zero",
            candidate.get("false_fix_epochs") == 0,
            "false_fix_epochs must be zero",
        ),
        _gate("locked_contract", locked["passed"], "holdout locks and M4 hashes must match"),
    ]

    for spec in MANDATORY_NEGATIVE_HOLDOUTS:
        result = holdouts.get(spec.holdout_id)
        complete = isinstance(result, Mapping) and result.get("evidence_complete") is True
        safe = (
            complete
            and result.get("accepted") is False
            and result.get("disposition") == spec.expected_disposition
        )
        gates.append(
            _gate(
                f"holdout:{spec.holdout_id}",
                safe,
                f"requires complete evidence and disposition={spec.expected_disposition}",
            )
        )

    manifest = payload.get("reproducibility_manifest")
    manifest_check = (
        verify_reproducibility_manifest(manifest, repo_root)
        if isinstance(manifest, Mapping)
        else {"passed": False, "content_hash_passed": False, "inputs": []}
    )
    gates.append(
        _gate(
            "reproducibility_manifest",
            manifest_check["passed"],
            "manifest and all input hashes must verify",
        )
    )

    required_metrics = (
        "total_epochs",
        "sub50cm_epochs",
        "p50_error_m",
        "p95_error_m",
        "max_contiguous_failure_s",
        "latency_p50_ms",
        "latency_p95_ms",
        "normal_latency_max_ms",
        "search_latency_max_ms",
        "peak_gpu_memory_mb",
    )
    missing_metrics = [name for name in required_metrics if candidate.get(name) is None]
    gates.append(
        _gate(
            "kpi_completeness",
            not missing_metrics,
            "missing: " + ", ".join(missing_metrics) if missing_metrics else "all Phase 0 KPIs reported",
        )
    )
    normal_latency = candidate.get("normal_latency_max_ms")
    search_latency = candidate.get("search_latency_max_ms")
    peak_memory = candidate.get("peak_gpu_memory_mb")
    runtime_values = (normal_latency, search_latency, peak_memory)
    runtime_pass = (
        all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in runtime_values
        )
        and normal_latency <= 100.0
        and search_latency <= 1000.0
        and peak_memory <= 4096.0
    )
    gates.append(
        _gate(
            "realtime_budget",
            runtime_pass,
            "normal<=100ms, search<=1000ms, peak GPU memory<=4096MB",
        )
    )

    total = candidate.get("total_epochs")
    sub50 = candidate.get("sub50cm_epochs")
    sub50_rate = (
        float(sub50) / float(total)
        if isinstance(total, int) and total > 0 and isinstance(sub50, int)
        else None
    )
    return {
        "schema": EVALUATION_RESULT_SCHEMA,
        "candidate_id": candidate.get("id"),
        "promoted": all(item["passed"] for item in gates),
        "gates": gates,
        "metrics": {
            **{name: candidate.get(name) for name in required_metrics},
            "sub50cm_rate": sub50_rate,
            "false_fix_epochs": candidate.get("false_fix_epochs"),
            "gained_epochs": candidate.get("gained_epochs"),
            "lost_epochs": candidate.get("lost_epochs"),
        },
        "failure_counts": {
            category.value: int(payload.get("failure_counts", {}).get(category.value, 0))
            for category in FailureCategory
        },
        "locked_contract": locked,
        "manifest_verification": manifest_check,
        "mandatory_holdouts": [
            {
                "holdout_id": spec.holdout_id,
                "city": spec.city,
                "dataset": spec.dataset,
                "segment": list(spec.segment),
                "failure_category": spec.failure_category.value,
                "expected_disposition": spec.expected_disposition,
                "lock_path": spec.lock_path,
                "lock_schema": spec.lock_schema,
                "lock_sha256": spec.lock_sha256,
            }
            for spec in MANDATORY_NEGATIVE_HOLDOUTS
        ],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_command() -> list[str]:
    return [sys.executable, *sys.argv]
