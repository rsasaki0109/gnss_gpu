import json
from pathlib import Path

from experiments.audit_ppc_basin_fgo_candidate_supply import audit_candidate_supply


def test_candidate_supply_separates_oracle_and_validation_recall(tmp_path: Path) -> None:
    basins = tmp_path / "basins.jsonl"
    rows = [
        {
            "schema": "gnsspp_multisd_basin_v1",
            "epoch_index": 0,
            "tow": 1,
            "rank": 0,
            "group_index": 0,
            "evaluated": True,
            "pass": False,
            "position_ecef": [10.1, 0, 0],
            "validation_residuals": [
                {
                    "satellite": "G02",
                    "reference_satellite": "G01",
                    "signal": 0,
                    "kind": "carrier",
                    "normalized_residual": 1.0,
                    "pass": True,
                }
            ],
        },
        {
            "schema": "gnsspp_multisd_basin_v1",
            "epoch_index": 0,
            "tow": 1,
            "rank": 1,
            "group_index": 0,
            "evaluated": True,
            "pass": True,
            "position_ecef": [12, 0, 0],
            "validation_residuals": [
                {
                    "satellite": "G02",
                    "reference_satellite": "G01",
                    "signal": 0,
                    "kind": "carrier",
                    "normalized_residual": -5.0,
                    "pass": False,
                }
            ],
        },
        {"schema": "gnsspp_multisd_basin_v1", "epoch_index": 1, "tow": 2, "rank": 0, "group_index": 0, "evaluated": True, "pass": True, "position_ecef": [20.2, 0, 0]},
        {"schema": "gnsspp_multisd_basin_v1", "epoch_index": 2, "tow": 3, "rank": -1, "group_index": -1, "evaluated": False, "pass": False},
    ]
    basins.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n1,10,0,0\n2,20,0,0\n3,30,0,0\n",
        encoding="utf-8",
    )
    result = audit_candidate_supply(basins, reference)
    assert result["stream_epochs"] == 3
    assert result["evaluated_epochs"] == 2
    assert result["oracle_correct_epochs"] == 2
    assert result["passed_correct_epochs"] == 1
    assert result["unique_pass_epochs"] == 2
    assert result["unique_pass_correct_epochs"] == 1
    assert result["correct_candidate_rank_histogram"] == {"0": 2}
    diagnostics = result["validation_residual_diagnostics"]
    assert diagnostics["rows"] == 2
    group = diagnostics["satellite_reference_signal_groups"]["carrier:G02:G01:0"]
    assert group["correct_rows"] == 1
    assert group["wrong_rows"] == 1
    assert group["wrong_failed_rows"] == 1
