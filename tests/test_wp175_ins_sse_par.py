import csv
from pathlib import Path

from experiments.analyze_wp175_ins_sse_par import analyze


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ins_sse_audit_uses_truth_only_after_pass(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    reference = tmp_path / "reference.csv"
    base = {
        "gps_week": 1,
        "available": 1,
        "attempted": 4,
        "subsets_evaluated": 2,
        "ratio_passed_subsets": 1,
        "separation_rejected_subsets": 0,
        "fixed_count": 4,
        "dropped_count": 0,
        "ratio": 3,
        "bsr_qscale16": 0.99,
        "ffrt_min_ratio": 2,
        "sse_statistic_per_dof": 1,
        "position_separation_m": 0.02,
        "candidate_ecef_y": 2,
        "candidate_ecef_z": 3,
    }
    _write(
        audit,
        [
            {
                **base,
                "tow": 10,
                "passed": 1,
                "candidate_ecef_x": 1.1,
            },
            {
                **base,
                "tow": 11,
                "passed": 0,
                "candidate_ecef_x": 100,
            },
        ],
    )
    _write(
        reference,
        [
            {
                "GPS TOW (s)": 10,
                "ECEF X (m)": 1,
                "ECEF Y (m)": 2,
                "ECEF Z (m)": 3,
            },
            {
                "GPS TOW (s)": 11,
                "ECEF X (m)": 1,
                "ECEF Y (m)": 2,
                "ECEF Z (m)": 3,
            },
        ],
    )

    candidates, summary = analyze(audit, reference)
    assert len(candidates) == 1
    assert summary["sub50cm_candidate_epochs"] == 1
    assert summary["single_source_fix_authority"] is False
    assert summary["truth_usage"] == "post_selection_audit_only"
