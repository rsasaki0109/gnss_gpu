"""Fail-closed cross-city and cross-device validation contracts.

The evaluator is deliberately metric-agnostic enough to cover positioning and
post-solver QA campaigns, but it never pools unlike metrics.  Every campaign is
checked independently before its geographic coverage contributes to Phase 5.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "gnss_gpu_cross_domain_validation_input_v1"
RESULT_SCHEMA = "gnss_gpu_cross_domain_validation_result_v1"


@dataclass(frozen=True)
class DomainRecord:
    record_id: str
    city: str
    site: str
    date: str
    receiver: str
    epochs: int
    baseline: float
    candidate: float
    baseline_safety: float
    candidate_safety: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DomainRecord:
        return cls(
            record_id=str(value["record_id"]),
            city=str(value["city"]).strip().lower(),
            site=str(value["site"]).strip().lower(),
            date=str(value["date"]),
            receiver=str(value["receiver"]).strip().lower(),
            epochs=int(value["epochs"]),
            baseline=float(value["baseline"]),
            candidate=float(value["candidate"]),
            baseline_safety=float(value["baseline_safety"]),
            candidate_safety=float(value["candidate_safety"]),
        )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def file_sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest().upper()


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def leave_one_group_out(
    records: Sequence[DomainRecord],
    attribute: str,
) -> list[dict[str, Any]]:
    """Build deterministic fold membership without leaking the held-out group."""

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, attribute))].append(record.record_id)
    all_ids = {record.record_id for record in records}
    folds = []
    for group, test_ids in sorted(groups.items()):
        test = sorted(test_ids)
        folds.append(
            {
                "held_out_group": group,
                "test_record_ids": test,
                "training_record_ids": sorted(all_ids - set(test)),
            }
        )
    return folds


def _verify_provenance(
    provenance: Any,
    repo_root: Path,
) -> tuple[bool, list[dict[str, Any]]]:
    if not isinstance(provenance, list) or not provenance:
        return False, []
    checks: list[dict[str, Any]] = []
    for raw in provenance:
        if not isinstance(raw, Mapping):
            checks.append({"passed": False, "reason": "entry_not_object"})
            continue
        relative = raw.get("path")
        expected = raw.get("sha256")
        path = repo_root / str(relative) if relative else Path("")
        actual = file_sha256(path) if relative and path.is_file() else None
        checks.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )
    return all(check["passed"] for check in checks), checks


def _evaluate_campaign(
    campaign: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    campaign_id = str(campaign.get("id", ""))
    raw_records = campaign.get("records")
    parse_error = None
    try:
        records = (
            [DomainRecord.from_mapping(item) for item in raw_records]
            if isinstance(raw_records, list)
            else []
        )
    except (KeyError, TypeError, ValueError) as exc:
        records = []
        parse_error = str(exc)

    ids = [record.record_id for record in records]
    metrics_valid = bool(records) and all(
        record.epochs > 0
        and _finite_nonnegative(record.baseline)
        and _finite_nonnegative(record.candidate)
        and _finite_nonnegative(record.baseline_safety)
        and _finite_nonnegative(record.candidate_safety)
        for record in records
    )
    primary_non_degraded = metrics_valid and all(
        record.candidate <= record.baseline for record in records
    )
    primary_improved = metrics_valid and any(
        record.candidate < record.baseline for record in records
    )
    safety_non_degraded = metrics_valid and all(
        record.candidate_safety <= record.baseline_safety for record in records
    )

    config = campaign.get("config")
    declared_config_hash = campaign.get("config_sha256")
    config_ok = (
        isinstance(config, Mapping)
        and declared_config_hash == canonical_sha256(config)
    )
    overrides = campaign.get("city_or_device_overrides")
    no_tuning = (
        campaign.get("model_selection_scope") == "development_only"
        and isinstance(overrides, Mapping)
        and not overrides
    )
    provenance_ok, provenance = _verify_provenance(
        campaign.get("provenance"),
        repo_root,
    )

    city_folds = leave_one_group_out(records, "city")
    all_receiver_folds = leave_one_group_out(records, "receiver")
    device_out_required = campaign.get("device_out_required", True) is True
    receiver_folds = all_receiver_folds if len(all_receiver_folds) >= 2 else []
    fold_contract = (
        len(city_folds) >= 2
        and (not device_out_required or len(receiver_folds) >= 2)
        and all(fold["test_record_ids"] for fold in city_folds + receiver_folds)
    )
    gates = [
        _gate("records_parse", parse_error is None and bool(records), parse_error or "ok"),
        _gate("unique_record_ids", len(ids) == len(set(ids)), "record IDs must be unique"),
        _gate("metrics_valid", metrics_valid, "metrics must be finite and epochs positive"),
        _gate(
            "primary_non_degraded",
            primary_non_degraded,
            "candidate must be no worse on every held-out record",
        ),
        _gate(
            "primary_improved",
            primary_improved,
            "candidate must improve at least one held-out record",
        ),
        _gate(
            "safety_non_degraded",
            safety_non_degraded,
            "candidate safety metric must be no worse on every held-out record",
        ),
        _gate("config_locked", config_ok, "declared global config hash must match"),
        _gate(
            "no_city_or_device_tuning",
            no_tuning,
            "selection must be development-only with no domain overrides",
        ),
        _gate("fold_contract", fold_contract, "leave-one-group-out folds must be complete"),
        _gate("provenance", provenance_ok, "all source artifacts must hash-match"),
    ]
    total_epochs = sum(record.epochs for record in records)
    weighted_baseline = (
        sum(record.baseline * record.epochs for record in records) / total_epochs
        if total_epochs
        else None
    )
    weighted_candidate = (
        sum(record.candidate * record.epochs for record in records) / total_epochs
        if total_epochs
        else None
    )
    return {
        "id": campaign_id,
        "passed": all(gate["passed"] for gate in gates),
        "primary_metric": campaign.get("primary_metric"),
        "safety_metric": campaign.get("safety_metric"),
        "gates": gates,
        "coverage": {
            "cities": sorted({record.city for record in records}),
            "sites": sorted({f"{record.city}/{record.site}" for record in records}),
            "dates": sorted({record.date for record in records}),
            "receivers": sorted({record.receiver for record in records}),
        },
        "leave_one_city_out": city_folds,
        "leave_one_device_out": receiver_folds,
        "device_out_verified": len(receiver_folds) >= 2,
        "weighted": {
            "epochs": total_epochs,
            "baseline": weighted_baseline,
            "candidate": weighted_candidate,
        },
        "records": [
            {
                **record.__dict__,
                "delta": record.candidate - record.baseline,
                "safety_delta": record.candidate_safety - record.baseline_safety,
            }
            for record in records
        ],
        "provenance": provenance,
    }


def evaluate_cross_domain(
    payload: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate all campaigns and enforce Phase 5 geographic coverage."""

    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"expected schema {INPUT_SCHEMA!r}")
    raw_campaigns = payload.get("campaigns")
    if not isinstance(raw_campaigns, list) or not raw_campaigns:
        raise ValueError("campaigns must be a non-empty array")
    campaigns = [
        _evaluate_campaign(campaign, repo_root)
        if isinstance(campaign, Mapping)
        else {"id": "", "passed": False, "coverage": {}}
        for campaign in raw_campaigns
    ]
    cities = sorted(
        {
            city
            for campaign in campaigns
            for city in campaign.get("coverage", {}).get("cities", [])
        }
    )
    sites = sorted(
        {
            site
            for campaign in campaigns
            for site in campaign.get("coverage", {}).get("sites", [])
        }
    )
    dates = sorted(
        {
            date
            for campaign in campaigns
            for date in campaign.get("coverage", {}).get("dates", [])
        }
    )
    receivers = sorted(
        {
            receiver
            for campaign in campaigns
            for receiver in campaign.get("coverage", {}).get("receivers", [])
        }
    )
    gates = [
        _gate("all_campaigns_pass", all(item["passed"] for item in campaigns), "all pass"),
        _gate("at_least_three_cities", len(cities) >= 3, f"observed {len(cities)}"),
        _gate("at_least_four_sites", len(sites) >= 4, f"observed {len(sites)}"),
        _gate("multiple_dates", len(dates) >= 2, f"observed {len(dates)}"),
        _gate("multiple_receivers", len(receivers) >= 2, f"observed {len(receivers)}"),
        _gate(
            "device_out_campaign",
            any(item.get("device_out_verified") is True for item in campaigns),
            "at least one campaign must contain two or more receiver holdouts",
        ),
    ]
    return {
        "schema": RESULT_SCHEMA,
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "coverage": {
            "cities": cities,
            "sites": sites,
            "dates": dates,
            "receivers": receivers,
        },
        "campaigns": campaigns,
    }
