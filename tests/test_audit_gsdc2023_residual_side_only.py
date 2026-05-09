from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_residual_side_only import residual_side_only_audit


def test_residual_side_only_audit_groups_scope_and_satellites(tmp_path: Path) -> None:
    def fake_compare(trip_dir: Path, *, max_epochs: int, multi_gnss: bool):
        assert max_epochs == 50
        assert multi_gnss is True
        trip = trip_dir.name
        merged = pd.DataFrame(
            [
                {
                    "side": "both",
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                },
                {
                    "side": "bridge_only",
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 2,
                    "utcTimeMillis": 2000,
                    "sys": 1,
                    "svid": 4,
                },
                {
                    "side": "bridge_only",
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 3,
                    "utcTimeMillis": 3000,
                    "sys": 1,
                    "svid": 4,
                },
                {
                    "side": "matlab_only",
                    "field": "P",
                    "freq": "L5",
                    "epoch_index": 4,
                    "utcTimeMillis": 4000,
                    "sys": 6,
                    "svid": 8,
                },
            ],
        )
        if trip == "phone-b":
            merged.loc[1:, "side"] = "both"
        return merged, pd.DataFrame(), {}

    by_scope, by_satellite, examples, payload = residual_side_only_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=50,
        multi_gnss=True,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["total_bridge_only"] == 2
    assert payload["total_matlab_only"] == 1
    assert payload["largest_scope"]["field"] == "D"
    assert by_scope["count"].tolist() == [2, 1]
    assert by_satellite.loc[by_satellite["side"].eq("bridge_only"), "svid"].iloc[0] == 4
    assert set(examples["side"]) == {"bridge_only", "matlab_only"}


def test_residual_side_only_audit_passes_when_no_side_only_rows(tmp_path: Path) -> None:
    def fake_compare(_trip_dir: Path, **_kwargs):
        return (
            pd.DataFrame(
                [
                    {
                        "side": "both",
                        "field": "D",
                        "freq": "L1",
                        "epoch_index": 1,
                        "utcTimeMillis": 1000,
                        "sys": 1,
                        "svid": 3,
                    },
                ],
            ),
            pd.DataFrame(),
            {},
        )

    by_scope, by_satellite, examples, payload = residual_side_only_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert by_scope.empty
    assert by_satellite.empty
    assert examples.empty
    assert payload["passed"] is True
    assert payload["total_side_only"] == 0
