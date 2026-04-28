from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_output import (
    BridgeResult,
    bridge_position_columns,
    ecef_to_llh_deg,
    export_bridge_outputs,
    format_metrics_line,
    has_valid_bridge_outputs,
    load_bridge_metrics,
    metrics_summary,
    score_from_metrics,
    validate_position_source,
)


def _metrics(*, rms_2d: float = 1.0) -> dict[str, float | int]:
    return {
        "rms_2d": rms_2d,
        "rms_3d": 1.2,
        "mean_2d": 0.9,
        "mean_3d": 1.0,
        "std_2d": 0.2,
        "p50": 0.8,
        "p67": 0.9,
        "p95": 1.4,
        "max_2d": 1.5,
        "n_epochs": 2,
    }


def _sample_result(*, with_truth: bool = True, with_metrics: bool = True) -> BridgeResult:
    times_ms = np.array([1000, 2000], dtype=np.float64)
    ecef = np.array(
        [
            [6378137.0, 0.0, 0.0],
            [6378138.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    raw_wls = np.column_stack([ecef, np.zeros(times_ms.size)])
    fgo_state = np.column_stack([ecef + np.array([0.0, 1.0, 0.0]), np.zeros(times_ms.size)])
    selected_state = np.column_stack([ecef, np.zeros(times_ms.size)])
    metrics = _metrics() if with_metrics else None
    return BridgeResult(
        trip="train/course/phone",
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        selected_source_mode="auto",
        times_ms=times_ms,
        kaggle_wls=ecef,
        raw_wls=raw_wls,
        fgo_state=fgo_state,
        selected_state=selected_state,
        selected_sources=np.array(["baseline", "fgo"], dtype=object),
        truth=ecef - 1.0 if with_truth else None,
        max_sats=7,
        fgo_iters=3,
        failed_chunks=0,
        selected_mse_pr=12.5,
        baseline_mse_pr=10.0,
        raw_wls_mse_pr=11.0,
        fgo_mse_pr=12.5,
        selected_source_counts={"baseline": 1, "raw_wls": 0, "fgo": 1},
        metrics_selected=metrics,
        metrics_kaggle=metrics,
        metrics_raw_wls=_metrics(rms_2d=2.0) if with_metrics else None,
        metrics_fgo=metrics,
        chunk_selection_records=[{"start": 0, "end": 2, "source": "fgo"}],
        parity_audit={"base_correction_ready": True, "base_correction_status": "ok"},
    )


def test_bridge_result_positions_payload_and_summary() -> None:
    result = _sample_result()

    positions = result.positions_table()
    payload = result.metrics_payload()
    summary = "\n".join(result.summary_lines())

    assert list(positions["UnixTimeMillis"]) == [1000, 2000]
    assert list(positions["SelectedSource"]) == ["baseline", "fgo"]
    np.testing.assert_allclose(positions["LatitudeDegrees"].to_numpy(), np.zeros(2), atol=1e-9)
    np.testing.assert_allclose(positions["LongitudeDegrees"].to_numpy(), np.zeros(2), atol=1e-9)
    assert payload["n_clock"] == 1
    assert payload["selected_score_m"] == 1.1
    assert payload["raw_wls_metrics"]["rms_2d_m"] == 2.0
    assert payload["chunk_selection_records"] == [{"start": 0, "end": 2, "source": "fgo"}]
    assert payload["parity_audit"]["base_correction_status"] == "ok"
    assert "parity" in summary
    assert "improves raw WLS" in summary


def test_bridge_result_without_truth_or_metrics_exports_nan_ground_truth(tmp_path) -> None:
    result = _sample_result(with_truth=False, with_metrics=False)

    export_bridge_outputs(tmp_path, result)

    positions = pd.read_csv(tmp_path / "bridge_positions.csv")
    payload = load_bridge_metrics(tmp_path)

    assert has_valid_bridge_outputs(tmp_path)
    assert np.isnan(positions.loc[0, "GroundTruthLatitudeDegrees"])
    assert payload["selected_score_m"] is None
    assert payload["selected_metrics"] is None
    assert "ground truth: unavailable" in "\n".join(result.summary_lines())


def test_has_valid_bridge_outputs_rejects_missing_and_invalid_payload(tmp_path) -> None:
    assert not has_valid_bridge_outputs(tmp_path)

    (tmp_path / "bridge_positions.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "bridge_metrics.json").write_text('{"fgo_iters": -1, "mse_pr": 1.0}', encoding="utf-8")

    assert not has_valid_bridge_outputs(tmp_path)


def test_bridge_position_columns_and_source_validation() -> None:
    columns = {
        "LatitudeDegrees",
        "LongitudeDegrees",
        "BaselineLatitudeDegrees",
        "BaselineLongitudeDegrees",
        "RawWlsLatitudeDegrees",
        "RawWlsLongitudeDegrees",
        "FgoLatitudeDegrees",
        "FgoLongitudeDegrees",
    }

    assert validate_position_source("auto") == "auto"
    assert bridge_position_columns("baseline", columns) == (
        "BaselineLatitudeDegrees",
        "BaselineLongitudeDegrees",
    )
    assert bridge_position_columns("raw_wls", columns) == (
        "RawWlsLatitudeDegrees",
        "RawWlsLongitudeDegrees",
    )
    assert bridge_position_columns("fgo", columns) == ("FgoLatitudeDegrees", "FgoLongitudeDegrees")
    assert bridge_position_columns("fgo", {"LatitudeDegrees", "LongitudeDegrees"}) == (
        "LatitudeDegrees",
        "LongitudeDegrees",
    )
    assert bridge_position_columns("gated", columns) == ("LatitudeDegrees", "LongitudeDegrees")
    with pytest.raises(ValueError, match="unsupported position source"):
        validate_position_source("bad")


def test_metrics_helpers_and_ecef_to_llh() -> None:
    metrics = _metrics()

    llh = ecef_to_llh_deg(np.array([[6378137.0, 0.0, 0.0]], dtype=np.float64))

    np.testing.assert_allclose(llh[0, :2], np.array([0.0, 0.0]), atol=1e-9)
    assert score_from_metrics(metrics) == 1.1
    assert metrics_summary(metrics)["p95_m"] == 1.4
    assert "RMS2D=1.000m" in format_metrics_line("Selected", metrics)
    assert format_metrics_line("Selected", None).endswith("unavailable")
