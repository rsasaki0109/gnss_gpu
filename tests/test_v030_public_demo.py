from __future__ import annotations

import json
from pathlib import Path

from experiments.build_v030_public_demo import build_snapshot


REPO_ROOT = Path(__file__).parents[1]


def test_public_snapshot_matches_locked_release_evidence() -> None:
    generated = build_snapshot(REPO_ROOT)
    locked = json.loads(
        (REPO_ROOT / "docs/assets/data/v030_release_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == locked
    assert generated["version"] == "0.3.0"
    assert generated["runtime"]["deadline_misses"] == 0
    assert generated["coverage"]["cities"] == ["hong-kong", "nagoya", "tokyo"]
    assert len(generated["negative_controls"]) == 4
    assert all(result["accepted"] is False for result in generated["negative_controls"])
    assert generated["promotion"]["passed_gates"] == 10
    assert generated["promotion"]["gate_count"] == 11
    assert generated["promotion"]["allowed"] is False
    assert generated["promotion"]["tokyo_epoch_gap"] == 1622
    assert generated["soak"]["simulated_duration_s"] == 7200.0
    assert generated["soak"]["final_mode"] == "normal"


def test_public_release_page_has_required_audit_sections() -> None:
    html = (REPO_ROOT / "docs/v0.3.0.html").read_text(encoding="utf-8")
    assert "Deterministic anomaly replay" in html
    assert "Mandatory negative controls" in html
    assert "Production promotion" in html
    assert "Limits, not footnotes" in html
    assert "v030_release_snapshot.json" in html
