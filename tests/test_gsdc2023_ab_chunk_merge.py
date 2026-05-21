from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gsdc2023_ab_chunk_merge import (
    ChunkComparison,
    compare_bridge_to_merged_candidate,
    load_bridge_chunk_records,
    load_chunk_records_from_summary,
    merge_adjacent_chunks_pairwise,
    merge_candidate_quality,
    predict_merged_gated_sources,
)
from experiments.gsdc2023_chunk_selection import ChunkCandidateQuality


def _q(
    *,
    mse_pr: float,
    quality_score: float = 0.5,
    step_p95: float = 8.0,
    accel_p95: float = 1.5,
    bgap_max: float = 10.0,
    bgap_p95: float = 8.0,
    step_mean: float = 3.0,
    bridge_jump: float = 0.0,
) -> ChunkCandidateQuality:
    return ChunkCandidateQuality(
        mse_pr=mse_pr,
        step_mean_m=step_mean,
        step_p95_m=step_p95,
        accel_mean_m=1.0,
        accel_p95_m=accel_p95,
        bridge_jump_m=bridge_jump,
        baseline_gap_mean_m=4.0,
        baseline_gap_p95_m=bgap_p95,
        baseline_gap_max_m=bgap_max,
        quality_score=quality_score,
    )


# --- merge_candidate_quality ----------------------------------------------


def test_merge_quality_weighted_mean_for_mse():
    base = _q(mse_pr=10.0)
    a = _q(mse_pr=5.0)
    b = _q(mse_pr=15.0)
    merged = merge_candidate_quality([a, b], [100, 100], merged_baseline=base)
    assert merged.mse_pr == pytest.approx(10.0)


def test_merge_quality_weighted_by_chunk_size():
    base = _q(mse_pr=10.0)
    a = _q(mse_pr=5.0)
    b = _q(mse_pr=15.0)
    merged = merge_candidate_quality([a, b], [200, 100], merged_baseline=base)
    # Weighted: (5*200 + 15*100) / 300 = 8.333...
    assert merged.mse_pr == pytest.approx(8.333333, rel=1e-4)


def test_merge_quality_p95_is_upper_bound_max():
    base = _q(mse_pr=10.0)
    a = _q(mse_pr=5.0, step_p95=4.0, accel_p95=1.0, bgap_p95=6.0)
    b = _q(mse_pr=15.0, step_p95=12.0, accel_p95=3.0, bgap_p95=15.0)
    merged = merge_candidate_quality([a, b], [100, 100], merged_baseline=base)
    assert merged.step_p95_m == 12.0
    assert merged.accel_p95_m == 3.0
    assert merged.baseline_gap_p95_m == 15.0


def test_merge_quality_baseline_gap_max_is_exact_max():
    base = _q(mse_pr=10.0)
    a = _q(mse_pr=5.0, bgap_max=8.0)
    b = _q(mse_pr=15.0, bgap_max=22.0)
    merged = merge_candidate_quality([a, b], [100, 100], merged_baseline=base)
    assert merged.baseline_gap_max_m == 22.0


def test_merge_quality_bridge_jump_uses_first_chunk_value():
    base = _q(mse_pr=10.0)
    a = _q(mse_pr=5.0, bridge_jump=1.2)
    b = _q(mse_pr=15.0, bridge_jump=99.0)
    merged = merge_candidate_quality([a, b], [100, 100], merged_baseline=base)
    assert merged.bridge_jump_m == 1.2


def test_merge_quality_quality_score_recomputed_from_merged_metrics():
    base = _q(mse_pr=10.0, step_p95=5.0, accel_p95=1.0, bgap_p95=4.0, step_mean=2.0, bridge_jump=0.5)
    a = _q(mse_pr=4.0, step_p95=4.0, accel_p95=0.8, bgap_p95=3.0)
    b = _q(mse_pr=4.0, step_p95=4.0, accel_p95=0.8, bgap_p95=3.0)
    merged = merge_candidate_quality([a, b], [100, 100], merged_baseline=base)
    # Recomputed; must not equal the placeholder 0.0.
    assert merged.quality_score > 0


def test_merge_quality_requires_matching_lengths():
    with pytest.raises(ValueError):
        merge_candidate_quality([_q(mse_pr=5.0)], [100, 100], merged_baseline=None)


def test_merge_quality_rejects_empty():
    with pytest.raises(ValueError):
        merge_candidate_quality([], [], merged_baseline=None)


# --- merge_adjacent_chunks_pairwise ---------------------------------------


def _chunk(start: int, end: int, baseline_mse: float, fgo_mse: float, gated: str = "baseline") -> dict:
    return {
        "start_epoch": start,
        "end_epoch": end,
        "auto_source": "fgo" if fgo_mse < baseline_mse else "baseline",
        "gated_source": gated,
        "candidates": {
            "baseline": {
                "mse_pr": baseline_mse,
                "step_p95_m": 10.0,
                "accel_p95_m": 1.5,
                "baseline_gap_p95_m": 0.0,
                "baseline_gap_max_m": 0.0,
                "quality_score": 1.0,
                "step_mean_m": 5.0,
                "accel_mean_m": 1.0,
                "bridge_jump_m": 0.0,
                "baseline_gap_mean_m": 0.0,
            },
            "fgo": {
                "mse_pr": fgo_mse,
                "step_p95_m": 6.0,
                "accel_p95_m": 0.8,
                "baseline_gap_p95_m": 4.0,
                "baseline_gap_max_m": 6.0,
                "quality_score": 0.7,
                "step_mean_m": 3.0,
                "accel_mean_m": 0.5,
                "bridge_jump_m": 0.0,
                "baseline_gap_mean_m": 3.0,
            },
        },
    }


def test_merge_adjacent_pairs_two_chunks_into_one_span():
    chunks = [_chunk(0, 100, 20.0, 18.0), _chunk(100, 200, 30.0, 28.0)]
    merged = merge_adjacent_chunks_pairwise(chunks)
    assert len(merged) == 1
    assert merged[0].start_epoch == 0
    assert merged[0].end_epoch == 200
    assert merged[0].candidates["baseline"].mse_pr == pytest.approx(25.0)
    assert merged[0].candidates["fgo"].mse_pr == pytest.approx(23.0)


def test_merge_adjacent_pairs_odd_count_keeps_final_singleton():
    chunks = [
        _chunk(0, 100, 20.0, 18.0),
        _chunk(100, 200, 30.0, 28.0),
        _chunk(200, 300, 25.0, 24.0),
    ]
    merged = merge_adjacent_chunks_pairwise(chunks)
    assert [(m.start_epoch, m.end_epoch) for m in merged] == [(0, 200), (200, 300)]
    assert merged[1].candidates["baseline"].mse_pr == 25.0


def test_merge_skips_candidate_not_present_in_every_sub_chunk():
    a = _chunk(0, 100, 20.0, 18.0)
    b = _chunk(100, 200, 30.0, 28.0)
    # Remove ``fgo`` from the second sub-chunk
    del b["candidates"]["fgo"]
    merged = merge_adjacent_chunks_pairwise([a, b])
    assert "fgo" not in merged[0].candidates  # cannot honestly merge
    assert "baseline" in merged[0].candidates


# --- predict_merged_gated_sources -----------------------------------------


def test_predict_merged_returns_one_record_per_merged_chunk():
    chunks = [_chunk(0, 100, 20.0, 18.0), _chunk(100, 200, 30.0, 28.0)]
    preds = predict_merged_gated_sources(chunks)
    assert len(preds) == 1
    start, end, gated = preds[0]
    assert (start, end) == (0, 200)
    assert isinstance(gated, str)


def test_predict_merged_defaults_to_baseline_when_no_baseline_candidate():
    # Synthetic minimal record without a ``baseline`` candidate (e.g. partial
    # extract).  The production gate cannot operate without baseline, so the
    # prediction is the safe ``baseline`` fall-through.
    chunks = [
        {
            "start_epoch": 0,
            "end_epoch": 100,
            "candidates": {
                "fgo_no_tdcp": {
                    "mse_pr": 8.0,
                    "step_p95_m": 5.0,
                    "accel_p95_m": 1.0,
                    "baseline_gap_p95_m": 0.0,
                    "baseline_gap_max_m": 0.0,
                    "quality_score": 0.5,
                    "step_mean_m": 2.0,
                    "accel_mean_m": 0.5,
                    "bridge_jump_m": 0.0,
                    "baseline_gap_mean_m": 0.0,
                }
            },
        }
    ]
    preds = predict_merged_gated_sources(chunks)
    assert preds == [(0, 100, "baseline")]


# --- compare_bridge_to_merged_candidate -----------------------------------


def test_compare_marks_matches_correctly():
    bridge_chunks = [
        {"start_epoch": 0, "end_epoch": 200, "gated_source": "fgo_no_tdcp"},
        {"start_epoch": 200, "end_epoch": 400, "gated_source": "baseline"},
    ]
    predictions = [(0, 200, "fgo_no_tdcp"), (200, 400, "baseline")]
    out = compare_bridge_to_merged_candidate(
        trip_id="t",
        n_epochs=400,
        bridge_chunks=bridge_chunks,
        merged_predictions=predictions,
    )
    assert [c.matches for c in out] == [True, True]
    assert [c.n_rows for c in out] == [200, 200]


def test_compare_marks_mismatch():
    bridge_chunks = [{"start_epoch": 0, "end_epoch": 200, "gated_source": "fgo_no_tdcp"}]
    out = compare_bridge_to_merged_candidate(
        trip_id="t",
        n_epochs=200,
        bridge_chunks=bridge_chunks,
        merged_predictions=[(0, 200, "baseline")],
    )
    assert out[0].matches is False
    assert out[0].bridge_gated_source == "fgo_no_tdcp"
    assert out[0].candidate_merged_gated_source == "baseline"


def test_compare_skips_when_bridge_has_no_coverage():
    out = compare_bridge_to_merged_candidate(
        trip_id="t", n_epochs=200, bridge_chunks=[], merged_predictions=[(0, 200, "baseline")]
    )
    assert out == []


def test_compare_uses_majority_vote_on_boundary_mismatch():
    bridge_chunks = [
        {"start_epoch": 0, "end_epoch": 100, "gated_source": "baseline"},
        {"start_epoch": 100, "end_epoch": 200, "gated_source": "fgo"},
    ]
    # Merged prediction asks about [0, 150), 100 baseline + 50 fgo → majority baseline
    out = compare_bridge_to_merged_candidate(
        trip_id="t",
        n_epochs=200,
        bridge_chunks=bridge_chunks,
        merged_predictions=[(0, 150, "fgo")],
    )
    assert out[0].bridge_gated_source == "baseline"
    assert out[0].matches is False


# --- IO helpers -----------------------------------------------------------


def test_load_chunk_records_from_summary_strips_train_test(tmp_path: Path):
    summary = {
        "trip_metrics": [
            {
                "trip": "train/foo/p",
                "n_epochs": 200,
                "chunk_selection_records": [{"start_epoch": 0, "end_epoch": 100}],
            }
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    out = load_chunk_records_from_summary(path)
    assert "foo/p" in out
    assert out["foo/p"]["n_epochs"] == 200


def test_load_bridge_chunk_records_walks_trip_phone_dirs(tmp_path: Path):
    trip_dir = tmp_path / "trip/phone"
    trip_dir.mkdir(parents=True)
    (trip_dir / "bridge_metrics.json").write_text(
        json.dumps({"n_epochs": 100, "chunk_selection_records": [{"start_epoch": 0}]})
    )
    out = load_bridge_chunk_records(tmp_path)
    assert "trip/phone" in out
    assert out["trip/phone"]["n_epochs"] == 100
