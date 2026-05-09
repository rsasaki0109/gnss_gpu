from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_residual_mask_drop import residual_mask_drop_audit


def test_residual_mask_drop_audit_marks_rows_recovered_without_mask(tmp_path: Path) -> None:
    def fake_compare(
        _trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
    ):
        assert max_epochs == 200
        assert multi_gnss is False
        row = {
            "field": "D",
            "freq": "L1",
            "epoch_index": 2,
            "utcTimeMillis": 2000,
            "sys": 1,
            "svid": 32,
            "matlab_sat_elevation": 4.5,
        }
        if apply_observation_mask:
            merged = pd.DataFrame([{**row, "side": "matlab_only"}])
            payload = {"total_matlab_only": 1}
        else:
            merged = pd.DataFrame([{**row, "side": "both"}])
            payload = {"total_matlab_only": 0}
        return merged, pd.DataFrame(), payload

    by_scope, by_satellite, detail, payload = residual_mask_drop_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=200,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is True
    assert payload["recovered_without_observation_mask"] == 1
    assert by_scope.loc[0, "drop_reason"] == "recovered_without_observation_mask"
    assert by_scope.loc[0, "mask_reason"] == "elevation_below_bridge_threshold"
    assert by_satellite.loc[0, "svid"] == 32
    assert detail.loc[0, "unmasked_total_matlab_only"] == 0


def test_residual_mask_drop_audit_reports_still_missing_rows(tmp_path: Path) -> None:
    def fake_compare(_trip_dir: Path, *, apply_observation_mask: bool, **_kwargs):
        masked_row = {
            "side": "matlab_only",
            "field": "P",
            "freq": "L5",
            "epoch_index": 3,
            "utcTimeMillis": 3000,
            "sys": 1,
            "svid": 32,
            "matlab_sat_elevation": 15.0,
        }
        if apply_observation_mask:
            return pd.DataFrame([masked_row]), pd.DataFrame(), {"total_matlab_only": 1}
        return pd.DataFrame(columns=list(masked_row)), pd.DataFrame(), {"total_matlab_only": 1}

    _by_scope, _by_satellite, detail, payload = residual_mask_drop_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["still_missing_without_observation_mask"] == 1
    assert detail.loc[0, "drop_reason"] == "still_missing_without_observation_mask"
    assert detail.loc[0, "mask_reason"] == "not_recovered"
