from __future__ import annotations

import copy
import json
from pathlib import Path

from gnss_gpu.cross_domain_validation import (
    INPUT_SCHEMA,
    canonical_sha256,
    evaluate_cross_domain,
    file_sha256,
)


def _campaign(
    source: Path,
    campaign_id: str,
    records: list[dict],
    *,
    device_out_required: bool = True,
) -> dict:
    config = {"candidate": "fixed", "seed": 0}
    return {
        "id": campaign_id,
        "primary_metric": "error",
        "safety_metric": "failure_rate",
        "model_selection_scope": "development_only",
        "device_out_required": device_out_required,
        "city_or_device_overrides": {},
        "config": config,
        "config_sha256": canonical_sha256(config),
        "provenance": [
            {
                "path": source.name,
                "sha256": file_sha256(source),
            }
        ],
        "records": records,
    }


def _record(
    record_id: str,
    city: str,
    site: str,
    date: str,
    receiver: str,
) -> dict:
    return {
        "record_id": record_id,
        "city": city,
        "site": site,
        "date": date,
        "receiver": receiver,
        "epochs": 100,
        "baseline": 10.0,
        "candidate": 9.0,
        "baseline_safety": 1.0,
        "candidate_safety": 0.0,
    }


def _payload(tmp_path: Path) -> dict:
    source = tmp_path / "source.csv"
    source.write_text("locked\n", encoding="utf-8")
    first = [
        _record("a", "tokyo", "odaiba", "2020-12-17", "trimble"),
        _record("b", "hong-kong", "kowloon", "2019-04-28", "ublox"),
    ]
    second = [
        _record("c", "nagoya", "run1", "2024-01-01", "novatel"),
        _record("d", "tokyo", "run1", "2024-01-02", "novatel"),
    ]
    return {
        "schema": INPUT_SCHEMA,
        "campaigns": [
            _campaign(source, "position", first),
            _campaign(source, "qa", second, device_out_required=False),
        ],
    }


def test_cross_domain_contract_passes_complete_locked_evidence(tmp_path: Path) -> None:
    result = evaluate_cross_domain(_payload(tmp_path), tmp_path)
    assert result["passed"] is True
    assert result["coverage"]["cities"] == ["hong-kong", "nagoya", "tokyo"]
    assert len(result["campaigns"][0]["leave_one_device_out"]) == 2


def test_cross_domain_contract_fails_on_domain_override(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["campaigns"][0]["city_or_device_overrides"] = {"hong-kong": {"seed": 4}}
    result = evaluate_cross_domain(payload, tmp_path)
    assert result["passed"] is False
    gates = {gate["name"]: gate["passed"] for gate in result["campaigns"][0]["gates"]}
    assert gates["no_city_or_device_tuning"] is False


def test_cross_domain_contract_fails_on_one_city_regression(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["campaigns"][0]["records"][1]["candidate"] = 11.0
    result = evaluate_cross_domain(payload, tmp_path)
    assert result["passed"] is False
    gates = {gate["name"]: gate["passed"] for gate in result["campaigns"][0]["gates"]}
    assert gates["primary_non_degraded"] is False


def test_cross_domain_contract_fails_on_tampered_provenance(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("changed\n", encoding="utf-8")
    result = evaluate_cross_domain(payload, tmp_path)
    assert result["passed"] is False


def test_config_hash_is_canonical_and_enforced(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    reordered = json.loads(json.dumps(payload))
    assert copy.deepcopy(reordered)
    payload["campaigns"][0]["config"]["seed"] = 1
    result = evaluate_cross_domain(payload, tmp_path)
    assert result["passed"] is False


def test_required_device_out_fails_with_only_one_receiver(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["campaigns"][1]["device_out_required"] = True
    result = evaluate_cross_domain(payload, tmp_path)
    assert result["passed"] is False
    gates = {gate["name"]: gate["passed"] for gate in result["campaigns"][1]["gates"]}
    assert gates["fold_contract"] is False


def test_locked_phase5_result_recomputes_exactly() -> None:
    repo_root = Path(__file__).parents[1]
    input_path = repo_root / "internal_docs" / "phase5_cross_domain_input_2026_07_29.json"
    result_path = repo_root / "internal_docs" / "phase5_cross_domain_result_2026_07_29.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    locked = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed = evaluate_cross_domain(payload, repo_root)
    assert recomputed == locked
    assert recomputed["passed"] is True
    assert recomputed["coverage"]["cities"] == ["hong-kong", "nagoya", "tokyo"]
    assert len(recomputed["coverage"]["receivers"]) == 3
