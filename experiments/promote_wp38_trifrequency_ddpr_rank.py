#!/usr/bin/env python3
"""Hash-verify WP38 trifrequency holdouts and promote the 6073 anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _checked(root: Path, item: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = root / item["path"]
    if _sha256(path) != str(item["sha256"]).upper():
        raise RuntimeError(f"hash mismatch: {item['path']}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_and_promote(root: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "wp38_trifrequency_ddpr_rank_validation_v1":
        raise RuntimeError("unexpected validation schema")
    if not lock.get("production_approved") or lock.get("production_input_truth"):
        raise RuntimeError("validation lock is not production-approved and truth-free")
    expected_config = {
        "min_evidence_epochs": 10,
        "max_family_rank_fraction": 0.2,
        "min_runner_margin": 0.2,
        "max_audit_error_m": 0.5,
    }
    if lock.get("selection_config") != expected_config:
        raise RuntimeError("selection config differs from the fixed production gate")
    for item in lock["m4_baseline"]:
        _checked(root, item)

    reports: list[dict[str, Any]] = []
    target_selection: dict[str, Any] | None = None
    for case in lock["cases"]:
        source_path, _source = _checked(root, case["candidate_source"])
        family_docs: dict[str, dict[str, Any]] = {}
        family_hashes: dict[str, str] = {}
        for family in ("primary", "secondary", "tertiary"):
            path, document = _checked(root, case[family])
            if document.get("production_input_truth", True):
                raise RuntimeError(f"{case['name']} {family} input is truth-tainted")
            if document.get("candidate_source_sha256") != _sha256(source_path).lower():
                raise RuntimeError(f"{case['name']} {family} candidate provenance differs")
            family_docs[family] = document
            family_hashes[family] = _sha256(path).lower()
        _selection_path, selection = _checked(root, case["selection"])
        if selection.get("production_input_truth", True) or selection.get("production_promoted"):
            raise RuntimeError(f"{case['name']} development state is invalid")
        if selection.get("config") != {
            key: expected_config[key] for key in expected_config if key != "max_audit_error_m"
        }:
            raise RuntimeError(f"{case['name']} selector config differs")
        if selection.get("input_sha256") != family_hashes:
            raise RuntimeError(f"{case['name']} selector input provenance differs")
        if int(selection["winner"]["candidate_id"]) != int(case["expected_winner_id"]):
            raise RuntimeError(f"{case['name']} winner differs")
        winner_id = int(selection["winner"]["candidate_id"])
        primary_winner = next(
            row
            for row in family_docs["primary"]["candidates"]
            if int(row["candidate_id"]) == winner_id
        )
        audit_error = float(primary_winner["final_error_m"])
        role = str(case["role"])
        if role in ("target", "positive"):
            if selection.get("reason") != "unique_trifrequency_ddpr_rank_consensus":
                raise RuntimeError(f"{case['name']} positive case did not select")
            if int(selection["selected_candidate_id"]) != winner_id:
                raise RuntimeError(f"{case['name']} selected ID differs from winner")
            if float(selection["runner_margin"]) < expected_config["min_runner_margin"]:
                raise RuntimeError(f"{case['name']} runner margin is below gate")
            if not selection.get("family_rank_pass"):
                raise RuntimeError(f"{case['name']} family rank gate failed")
            if audit_error > expected_config["max_audit_error_m"]:
                raise RuntimeError(f"{case['name']} positive audit exceeds limit")
        elif role == "negative":
            if selection.get("reason") != "trifrequency_ddpr_rank_gate_failed":
                raise RuntimeError(f"{case['name']} negative case unexpectedly selected")
            if selection.get("selected_candidate_id") is not None:
                raise RuntimeError(f"{case['name']} negative case has a selected ID")
            if audit_error <= expected_config["max_audit_error_m"]:
                raise RuntimeError(f"{case['name']} negative winner is not unsafe")
        else:
            raise RuntimeError(f"unsupported validation role: {role}")
        reports.append(
            {
                "name": case["name"],
                "role": role,
                "winner_id": winner_id,
                "runner_margin": float(selection["runner_margin"]),
                "winner_audit_error_m": audit_error,
                "selected": selection.get("selected_candidate_id") is not None,
            }
        )
        if role == "target":
            target_selection = selection

    if target_selection is None:
        raise RuntimeError("validation lock has no target case")
    return {
        **target_selection,
        "schema": "wp38_trifrequency_ddpr_rank_production_v1",
        "production_promoted": True,
        "validation_lock_sha256": _sha256(lock_path),
        "validation_reports": reports,
    }


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
