from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.audit_ppc_causal_float_selector import audit_selector


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        "1,0,0,0\n2,1,0,0\n3,2,0,0\n4,3,0,0\n",
        encoding="utf-8",
    )
    safe = tmp_path / "safe.csv"
    safe.write_text(
        "tow,status,x,y,z\n1,4,0,0,0\n2,3,3,0,0\n3,3,4,0,0\n4,3,5,0,0\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected.csv"
    selected.write_text(
        "tow,status,x,y,z\n1,4,0,0,0\n2,3,1,0,0\n3,3,2,0,0\n4,3,3,0,0\n",
        encoding="utf-8",
    )
    safe_summary = tmp_path / "safe.json"
    safe_summary.write_text(
        json.dumps({"output_sha256": _digest(safe)}), encoding="utf-8"
    )
    selector_summary = tmp_path / "selector.json"
    selector_summary.write_text(
        json.dumps(
            {
                "production_input_truth": False,
                "truth_usage": "none",
                "input_sha256": {"safe_output": _digest(safe)},
                "output_sha256": _digest(selected),
            }
        ),
        encoding="utf-8",
    )
    return safe, safe_summary, selected, selector_summary, reference


def test_audit_proves_safe_fix_unchanged_and_scores_afterward(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = audit_selector(*args, blocks=2)

    assert result["safe_fix_set_identical"] is True
    assert result["safe_fix_positions_identical"] is True
    assert result["score_delta_pct_points"] > 0.0
    assert result["truth_usage"] == "post_estimator_scoring_only"


def test_audit_rejects_changed_safe_fix(tmp_path: Path) -> None:
    safe, safe_summary, selected, selector_summary, reference = _fixture(tmp_path)
    selected.write_text(
        selected.read_text(encoding="utf-8").replace("1,4,0,0,0", "1,3,0,0,0"),
        encoding="utf-8",
    )
    metadata = json.loads(selector_summary.read_text(encoding="utf-8"))
    metadata["output_sha256"] = _digest(selected)
    selector_summary.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="safe FIX"):
        audit_selector(
            safe, safe_summary, selected, selector_summary, reference, blocks=2
        )
