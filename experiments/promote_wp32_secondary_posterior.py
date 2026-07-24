#!/usr/bin/env python3
"""Promote a secondary-code posterior only after locked holdout validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _verify_file(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = root / str(entry["path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest().upper() != str(entry["sha256"]).upper():
        raise RuntimeError(f"artifact hash mismatch: {entry['path']}")
    return json.loads(content)


def validate_and_promote(
    manifest: dict[str, Any], artifacts: dict[str, Any]
) -> dict[str, Any]:
    if manifest.get("schema") != "wp32_secondary_posterior_validation_v1":
        raise RuntimeError("unsupported validation manifest")
    if not bool(manifest.get("production_approved", False)):
        raise RuntimeError("validation manifest is not production-approved")
    development = artifacts["development"]
    dev_entry = manifest["development"]
    if development.get("reason") != dev_entry["required_reason"]:
        raise RuntimeError("development selector reason mismatch")
    if bool(development.get("production_input_truth", True)):
        raise RuntimeError("development selector used truth as production input")
    if float(development.get("selected_audit_error_m", float("inf"))) >= float(
        dev_entry["max_audit_error_m"]
    ):
        raise RuntimeError("development audit gate failed")
    expected_config = {
        "min_evidence_epochs": manifest["selection_config"][
            "secondary_min_evidence_epochs"
        ],
        "top_k": manifest["selection_config"]["secondary_top_k"],
        "max_secondary_median_m": manifest["selection_config"][
            "secondary_max_median_m"
        ],
        "max_support_spread_m": manifest["selection_config"][
            "secondary_max_support_spread_m"
        ],
    }
    if development.get("config") != expected_config:
        raise RuntimeError("development selection config mismatch")

    positive = artifacts["positive_holdout"]
    positive_entry = manifest["positive_holdout"]
    if positive.get("reason") != positive_entry["required_reason"]:
        raise RuntimeError("positive holdout reason mismatch")
    if len(positive.get("selected_candidate_ids", [])) != int(
        positive_entry["required_support_members"]
    ):
        raise RuntimeError("positive holdout support count mismatch")
    if float(positive.get("selected_audit_error_m", float("inf"))) >= float(
        positive_entry["max_audit_error_m"]
    ):
        raise RuntimeError("positive holdout audit gate failed")

    for entry, artifact in zip(
        manifest["late_negative_holdouts"],
        artifacts["late_negative_holdouts"],
        strict=True,
    ):
        if artifact.get("reason") != entry["required_reason"]:
            raise RuntimeError("late negative holdout did not fail closed")
    proposal_min_members = int(manifest["selection_config"]["proposal_min_members"])
    proposal_min_score = float(manifest["selection_config"]["proposal_min_score"])
    for artifact in artifacts["early_negative_holdouts"]:
        if any(
            int(row["members"]) >= proposal_min_members
            and float(row["score"]) >= proposal_min_score
            for row in artifact.get("clusters", [])
        ):
            raise RuntimeError("early negative holdout passes proposal gate")

    return {
        "selected_candidate_id": int(development["selected_candidate_ids"][0]),
        "selected_candidate_ids": [
            int(value) for value in development["selected_candidate_ids"]
        ],
        "position_ecef": development["position_ecef"],
        "reason": "unique_secondary_topk_posterior",
        "production_promoted": True,
        "production_input_truth": False,
        "segment": development["segment"],
        "support_secondary_median_m": development[
            "support_secondary_median_m"
        ],
        "support_spread_m": development["support_spread_m"],
        "selected_audit_error_m": development["selected_audit_error_m"],
        "validation": {
            "positive_holdouts": 1,
            "late_negative_holdouts": len(artifacts["late_negative_holdouts"]),
            "early_negative_holdouts": len(artifacts["early_negative_holdouts"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.validation_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    root = Path(__file__).resolve().parents[1]
    artifacts = {
        "development": _verify_file(root, manifest["development"]),
        "positive_holdout": _verify_file(root, manifest["positive_holdout"]),
        "late_negative_holdouts": [
            _verify_file(root, entry) for entry in manifest["late_negative_holdouts"]
        ],
        "early_negative_holdouts": [
            _verify_file(root, entry) for entry in manifest["early_negative_holdouts"]
        ],
    }
    result = validate_and_promote(manifest, artifacts)
    result.update(
        {
            "schema": "wp32_secondary_posterior_production_v1",
            "validation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
