#!/usr/bin/env python3
"""Hash-verify WP34 holdouts and promote the Tokyo relative-secondary anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _checked(root: Path, path: str, expected: str) -> tuple[Path, dict[str, Any]]:
    resolved = root / path
    if _sha256(resolved) != expected.upper():
        raise RuntimeError(f"hash mismatch: {path}")
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def three_radius_ranking(
    selection: dict[str, Any],
    secondary: dict[str, Any],
    audit: dict[str, Any],
    *,
    direction_count: int,
) -> list[dict[str, Any]]:
    secondary_by_id = {
        int(row["candidate_id"]): row for row in secondary["candidates"]
    }
    audit_by_id = {int(row["candidate_id"]): row for row in audit["candidates"]}
    eligible = {int(value) for value in selection["road_eligible_candidate_ids"]}
    groups: list[dict[str, Any]] = []
    for base in range(direction_count):
        ids = [base, base + direction_count, base + 2 * direction_count]
        if not all(candidate_id in eligible for candidate_id in ids):
            continue
        medians = [
            float(secondary_by_id[candidate_id]["ddpr_median_abs_m"])
            for candidate_id in ids
        ]
        audits = [
            float(audit_by_id[candidate_id]["audit_error_m"])
            for candidate_id in ids
        ]
        groups.append(
            {
                "ids": ids,
                "secondary_mean_m": float(np.mean(medians)),
                "secondary_max_m": float(max(medians)),
                "audit_mean_m": float(np.mean(audits)),
            }
        )
    if len(groups) < 2:
        raise RuntimeError("holdout has fewer than two complete road-eligible groups")
    return sorted(groups, key=lambda row: row["secondary_mean_m"])


def _relative_margin(ranking: list[dict[str, Any]]) -> float:
    return float(
        (ranking[1]["secondary_mean_m"] - ranking[0]["secondary_mean_m"])
        / ranking[0]["secondary_mean_m"]
    )


def validate_and_promote(root: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "wp34_relative_secondary_validation_v1":
        raise RuntimeError("unexpected validation schema")
    if not lock.get("production_approved") or lock.get("production_input_truth"):
        raise RuntimeError("validation lock is not production-approved and truth-free")
    config = lock["selection_config"]
    expected_config = {
        "top_k": 3,
        "min_evidence_epochs": 10,
        "min_relative_margin": 0.075,
        "max_primary_spread_m": 0.5,
        "max_audit_error_m": 0.5,
    }
    if config != expected_config:
        raise RuntimeError("selection config differs from the locked production gate")
    for item in lock["m4_baseline"]:
        _checked(root, item["path"], item["sha256"])

    development = lock["development"]
    _dev_path, dev = _checked(root, development["path"], development["sha256"])
    _audit_path, dev_audit = _checked(
        root, development["audit_path"], development["audit_sha256"]
    )
    candidate_hashes = []
    for item in development["candidate_inputs"]:
        path, _document = _checked(root, item["path"], item["sha256"])
        candidate_hashes.append(_sha256(path).lower())
    secondary_hashes = []
    for item in development["secondary_inputs"]:
        path, document = _checked(root, item["path"], item["sha256"])
        if document.get("production_input_truth", True):
            raise RuntimeError("development secondary input is truth-tainted")
        secondary_hashes.append(_sha256(path).lower())
    if dev.get("candidate_sha256") != candidate_hashes:
        raise RuntimeError("development candidate provenance differs from lock")
    if dev.get("secondary_sha256") != secondary_hashes:
        raise RuntimeError("development secondary provenance differs from lock")
    if dev.get("production_promoted") or dev.get("production_input_truth", True):
        raise RuntimeError("development selector state is invalid")
    if dev.get("config") != {key: config[key] for key in expected_config if key != "max_audit_error_m"}:
        raise RuntimeError("development selector config differs from lock")
    if dev.get("reason") != development["required_reason"]:
        raise RuntimeError("development selector did not pass")
    if float(dev["secondary_relative_margin"]) < config["min_relative_margin"]:
        raise RuntimeError("development relative margin is below gate")
    if float(dev["primary_spread_m"]) > config["max_primary_spread_m"]:
        raise RuntimeError("development primary support is not compact")
    audit_rows = dev_audit.get("candidates", [])
    if len(audit_rows) != 1 or float(audit_rows[0]["audit_error_m"]) > config["max_audit_error_m"]:
        raise RuntimeError("development truth-only audit exceeds promotion limit")

    positive_reports = []
    for item in lock["positive_holdouts"]:
        candidate_path, _candidate = _checked(
            root, item["candidate_path"], item["candidate_sha256"]
        )
        _selection_path, selection = _checked(
            root, item["selection_path"], item["selection_sha256"]
        )
        _audit_path, audit = _checked(root, item["audit_path"], item["audit_sha256"])
        _secondary_path, secondary = _checked(
            root, item["secondary_path"], item["secondary_sha256"]
        )
        if secondary.get("candidate_source_sha256") != _sha256(candidate_path).lower():
            raise RuntimeError("positive holdout candidate provenance differs")
        if secondary.get("production_input_truth", True):
            raise RuntimeError("positive holdout is truth-tainted")
        if int(secondary.get("evidence_epochs", 0)) < config["min_evidence_epochs"]:
            raise RuntimeError("positive holdout evidence is below gate")
        ranking = three_radius_ranking(
            selection, secondary, audit, direction_count=int(item["direction_count"])
        )
        margin = _relative_margin(ranking)
        if ranking[0]["ids"] != item["expected_winner_ids"]:
            raise RuntimeError("positive holdout did not recover production cluster")
        if selection.get("selected_cluster_candidate_ids") != item["expected_winner_ids"]:
            raise RuntimeError("positive holdout selection provenance differs")
        if margin < config["min_relative_margin"]:
            raise RuntimeError("positive holdout relative margin is below gate")
        if ranking[0]["audit_mean_m"] > config["max_audit_error_m"]:
            raise RuntimeError("positive holdout winner exceeds audit limit")
        positive_reports.append(
            {"segment": item["segment"], "winner_ids": ranking[0]["ids"], "margin": margin}
        )

    negative = lock["relative_negative_holdout"]
    candidate_path, _candidate = _checked(
        root, negative["candidate_path"], negative["candidate_sha256"]
    )
    _selection_path, selection = _checked(
        root, negative["selection_path"], negative["selection_sha256"]
    )
    _audit_path, audit = _checked(root, negative["audit_path"], negative["audit_sha256"])
    _secondary_path, secondary = _checked(
        root, negative["secondary_path"], negative["secondary_sha256"]
    )
    if secondary.get("candidate_source_sha256") != _sha256(candidate_path).lower():
        raise RuntimeError("negative holdout candidate provenance differs")
    ranking = three_radius_ranking(
        selection, secondary, audit, direction_count=int(negative["direction_count"])
    )
    negative_margin = _relative_margin(ranking)
    if selection.get("reason") != negative["required_selection_reason"]:
        raise RuntimeError("relative negative selection reason differs")
    if negative_margin >= config["min_relative_margin"]:
        raise RuntimeError("relative negative holdout unexpectedly passes")
    if ranking[0]["audit_mean_m"] <= config["max_audit_error_m"]:
        raise RuntimeError("relative negative winner is not a meaningful negative")
    for item in lock["proposal_negative_holdouts"]:
        _path, document = _checked(root, item["path"], item["sha256"])
        if document.get("reason") != item["required_reason"]:
            raise RuntimeError("proposal negative reason differs")
        if document.get("selected_candidate_ids"):
            raise RuntimeError("proposal negative unexpectedly selected candidates")

    promoted = {
        **dev,
        "schema": "wp34_relative_secondary_parent_production_v1",
        "production_promoted": True,
        "validation_lock_sha256": _sha256(lock_path),
        "validation_reports": {
            "positive_holdouts": positive_reports,
            "relative_negative": {
                "segment": negative["segment"],
                "margin": negative_margin,
                "winner_audit_mean_m": ranking[0]["audit_mean_m"],
            },
        },
    }
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promoted = validate_and_promote(args.repo_root, args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(promoted, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(promoted, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
