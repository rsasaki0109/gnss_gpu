import hashlib
import json
from pathlib import Path

import pytest

from experiments.promote_wp38_trifrequency_ddpr_rank import validate_and_promote
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "internal_docs/wp38_trifrequency_ddpr_rank_validation_2026_07_22.json"


def test_locked_wp38_validation_promotes_target():
    promoted = validate_and_promote(ROOT, LOCK)

    assert promoted["production_promoted"] is True
    assert promoted["selected_candidate_id"] == 59
    assert promoted["reason"] == "unique_trifrequency_ddpr_rank_consensus"
    assert len(promoted["validation_reports"]) == 5
    assert sum(report["selected"] for report in promoted["validation_reports"]) == 3


def test_wp38_promotion_rejects_changed_lock_hash(tmp_path: Path):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock["cases"][0]["primary"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_and_promote(ROOT, changed)


def test_m4_hashes_remain_exact():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for item in lock["m4_baseline"]:
        actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest().upper()
        assert actual == item["sha256"]


def test_wp38_production_reason_is_accepted_by_smoother(tmp_path: Path):
    path = tmp_path / "anchor.json"
    path.write_text(
        json.dumps(
            {
                "segment": [6073, 6539],
                "selected_candidate_id": 59,
                "reason": "unique_trifrequency_ddpr_rank_consensus",
                "position_ecef": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    assert _load_static_position_override(path)[3:] == (
        59,
        "unique_trifrequency_ddpr_rank_consensus",
    )
