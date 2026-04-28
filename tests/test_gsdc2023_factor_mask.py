from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_factor_mask import (
    append_factor_rows,
    build_factor_mask_from_residual_diagnostics,
    factor_mask_side_summary,
    merge_factor_mask_keys,
    normalize_factor_mask_frame,
)


def test_append_factor_rows_builds_matlab_style_keys() -> None:
    rows: list[dict[str, object]] = []

    append_factor_rows(
        rows,
        field_names=("P", "resPc"),
        freq="L1",
        epoch_indices=np.array([0, 1], dtype=np.int64),
        slot_indices=np.array([0, 1], dtype=np.int64),
        times_ms=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"), (1, 4, "GPS_L1_CA")),
        next_epoch_indices=np.array([1, 0], dtype=np.int64),
        epoch_offset=10,
    )

    frame = normalize_factor_mask_frame(pd.DataFrame(rows))

    assert frame["field"].tolist() == ["P", "P", "resPc", "resPc"]
    assert frame["epoch_index"].tolist() == [11, 12, 11, 12]
    assert frame["utcTimeMillis"].tolist() == [1000, 2000, 1000, 2000]
    assert frame["next_epoch_index"].tolist() == [12, 0, 12, 0]
    assert frame["nextUtcTimeMillis"].tolist() == [2000, 0, 2000, 0]
    assert frame["sys"].tolist() == [1, 1, 1, 1]
    assert frame["svid"].tolist() == [3, 4, 3, 4]
    assert frame["signal_type"].tolist() == ["GPS_L1_CA", "GPS_L1_CA", "GPS_L1_CA", "GPS_L1_CA"]


def test_factor_mask_merge_and_side_summary_counts_symmetric_parity() -> None:
    left = normalize_factor_mask_frame(
        pd.DataFrame(
            [
                _mask_row("P", 1, 1000, 3),
                _mask_row("D", 2, 2000, 4),
            ],
        ),
        keep_extra_columns=False,
    )
    right = normalize_factor_mask_frame(
        pd.DataFrame(
            [
                _mask_row("P", 1, 1000, 3),
                _mask_row("L", 1, 1000, 5, next_epoch_index=2, next_utc=2000),
            ],
        ),
        keep_extra_columns=False,
    )

    merged = merge_factor_mask_keys(
        left,
        right,
        left_only_side="matlab_only",
        right_only_side="bridge_only",
    )
    summary, payload = factor_mask_side_summary(
        merged,
        left_name="matlab",
        right_name="bridge",
        left_only_side="matlab_only",
        right_only_side="bridge_only",
        include_jaccard=True,
    )

    assert merged["side"].value_counts().to_dict() == {"both": 1, "matlab_only": 1, "bridge_only": 1}
    assert payload["total_matlab_count"] == 2
    assert payload["total_bridge_count"] == 2
    assert payload["total_matched_count"] == 1
    assert payload["total_matlab_only"] == 1
    assert payload["total_bridge_only"] == 1
    assert payload["jaccard"] == 1.0 / 3.0
    assert payload["symmetric_parity"] == 0.5
    assert set(summary["field"]) == {"D", "L", "P"}


def test_build_factor_mask_from_residual_diagnostics_rebuilds_l_pairs(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_factor_finite": 1,
                "d_factor_finite": 0,
                "l_factor_finite": 1,
            },
            {
                "freq": "L1",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 3,
                "p_factor_finite": 0,
                "d_factor_finite": 1,
                "l_factor_finite": 1,
            },
        ],
    ).to_csv(diagnostics_path, index=False)

    mask = build_factor_mask_from_residual_diagnostics(diagnostics_path)

    assert mask["field"].tolist() == ["D", "L", "P", "resD", "resL", "resPc"]
    assert mask.set_index("field").loc["L", "next_epoch_index"] == 2
    assert mask.set_index("field").loc["L", "nextUtcTimeMillis"] == 2000
    assert mask.set_index("field").loc["P", "epoch_index"] == 1
    assert mask.set_index("field").loc["D", "epoch_index"] == 2


def _mask_row(
    field: str,
    epoch_index: int,
    utc_time_ms: int,
    svid: int,
    *,
    next_epoch_index: int = 0,
    next_utc: int = 0,
) -> dict[str, object]:
    return {
        "field": field,
        "freq": "L1",
        "epoch_index": epoch_index,
        "utcTimeMillis": utc_time_ms,
        "next_epoch_index": next_epoch_index,
        "nextUtcTimeMillis": next_utc,
        "sys": 1,
        "svid": svid,
    }
