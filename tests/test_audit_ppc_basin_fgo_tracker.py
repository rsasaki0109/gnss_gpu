from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.audit_ppc_basin_fgo_tracker import audit_tracker


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tracker(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch_index", "tow", "shadow_fixed", "x", "y", "z",
                "ffbsi_valid", "ffbsi_tow", "ffbsi_x", "ffbsi_y", "ffbsi_z",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"epoch_index": 0, "tow": 1, "shadow_fixed": 1, "x": 10.1, "y": 0, "z": 0, "ffbsi_valid": 0},
                {"epoch_index": 1, "tow": 2, "shadow_fixed": 0, "x": 20, "y": 0, "z": 0, "ffbsi_valid": 1, "ffbsi_tow": 1, "ffbsi_x": 10.05, "ffbsi_y": 0, "ffbsi_z": 0},
                {"epoch_index": 2, "tow": 3, "shadow_fixed": 1, "x": 31.2, "y": 0, "z": 0, "ffbsi_valid": 1, "ffbsi_tow": 2, "ffbsi_x": 21.2, "ffbsi_y": 0, "ffbsi_z": 0},
            ]
        )


def test_audit_uses_full_reference_denominator_and_checks_summary(tmp_path: Path) -> None:
    tracker = tmp_path / "tracker.csv"
    reference = tmp_path / "reference.csv"
    summary = tmp_path / "tracker.json"
    _write_tracker(tracker)
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        "1,10,0,0\n2,20,0,0\n3,30,0,0\n4,40,0,0\n",
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "schema": "gnss_gpu_ppc_basin_fgo_tracker_v1",
                "production_input_truth": False,
                "truth_usage": "none",
                "output_sha256": _sha256(tracker),
            }
        ),
        encoding="utf-8",
    )

    result = audit_tracker(tracker, reference, tracker_summary=summary)

    assert result["total_epochs"] == 4
    assert result["fixed"] == 2
    assert result["correct_fix"] == 1
    assert result["false_fix"] == 1
    assert result["false_fix_above_1m"] == 1
    assert result["fix_rate_full_denominator"] == 0.25
    assert result["integrity"]["tracker_summary_valid"] is True
    assert result["integrity"]["passed"] is False
    assert result["delayed_ffbsi"]["evaluated_epochs"] == 2
    assert result["delayed_ffbsi"]["below_0_5m"] == 1
    assert result["delayed_ffbsi"]["above_1m"] == 1


def test_audit_rejects_duplicate_tracker_tow(tmp_path: Path) -> None:
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "epoch_index,tow,shadow_fixed,x,y,z\n0,1,0,0,0,0\n1,1,0,0,0,0\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n1,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate tracker TOW"):
        audit_tracker(tracker, reference)


def test_audit_baseline_priority_union_only_rescues_nonfix(tmp_path: Path) -> None:
    tracker = tmp_path / "tracker.csv"
    _write_tracker(tracker)
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        "1,10,0,0\n2,20,0,0\n3,30,0,0\n4,40,0,0\n5,50,0,0\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.pos"
    baseline.write_text(
        "% test\n0 1 10.2 0 0 0 0 0 4\n0 2 20 0 0 0 0 0 3\n"
        "0 3 30 0 0 0 0 0 4\n0 4 40 0 0 0 0 0 3\n",
        encoding="utf-8",
    )
    result = audit_tracker(tracker, reference, baseline_pos=baseline)
    union = result["baseline_priority_union"]
    assert result["total_epochs"] == 4
    assert result["denominator_contract"] == "baseline_solution_epochs"
    assert union["baseline_fixed"] == 2
    assert union["tracker_rescue_fixed"] == 0
    assert union["fixed"] == 2
    assert union["correct_fix"] == 2
