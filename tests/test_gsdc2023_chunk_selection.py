import numpy as np

from experiments.gsdc2023_chunk_selection import (
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    add_tdcp_off_fgo_candidates,
    chunk_candidate_quality,
    chunk_selection_payload,
    select_auto_chunk_source,
    select_gated_chunk_source,
)


def _quality(
    mse_pr: float,
    quality_score: float,
    *,
    gap_p95: float = 0.0,
    gap_max: float = 0.0,
    step_p95: float = 10.0,
) -> ChunkCandidateQuality:
    return ChunkCandidateQuality(
        mse_pr=mse_pr,
        step_mean_m=2.0,
        step_p95_m=step_p95,
        accel_mean_m=1.0,
        accel_p95_m=3.0,
        bridge_jump_m=0.5,
        baseline_gap_mean_m=gap_p95 * 0.5,
        baseline_gap_p95_m=gap_p95,
        baseline_gap_max_m=gap_max,
        quality_score=quality_score,
    )


def test_chunk_candidate_quality_uses_baseline_gap_and_prev_tail():
    baseline_state = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    candidate_state = baseline_state + np.array([0.0, 3.0, 0.0], dtype=np.float64)
    baseline_quality = chunk_candidate_quality(
        baseline_state,
        100.0,
        baseline_quality=None,
        prev_tail_xyz=None,
    )

    quality = chunk_candidate_quality(
        candidate_state,
        80.0,
        baseline_quality=baseline_quality,
        prev_tail_xyz=np.array([-1.0, 3.0, 0.0], dtype=np.float64),
        baseline_xyz=baseline_state,
    )

    assert quality.mse_pr == 80.0
    np.testing.assert_allclose(quality.bridge_jump_m, 1.0)
    np.testing.assert_allclose(quality.baseline_gap_mean_m, 3.0)
    np.testing.assert_allclose(quality.baseline_gap_p95_m, 3.0)
    np.testing.assert_allclose(quality.baseline_gap_max_m, 3.0)
    assert quality.quality_score < 1.0


def test_select_auto_chunk_source_prefers_clear_quality_gain():
    candidates = {
        "baseline": _quality(100.0, 1.0),
        "fgo": _quality(106.0, 0.78, gap_p95=2.0, gap_max=3.0),
    }

    assert select_auto_chunk_source(candidates) == "fgo"


def test_select_auto_chunk_source_rejects_high_mse_fgo():
    candidates = {
        "baseline": _quality(1000.0, 1.0),
        "fgo": _quality(450.0, 0.1, gap_p95=2.0, gap_max=3.0),
    }

    assert select_auto_chunk_source(candidates) == "baseline"


def test_select_gated_chunk_source_keeps_safe_tdcp_candidate_when_tdcp_off_tied():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo_no_tdcp",
        candidates={
            "baseline": _quality(20.0, 1.0),
            "fgo": _quality(18.6, 0.864, gap_p95=9.1, gap_max=12.0),
            "fgo_no_tdcp": _quality(18.5, 0.861, gap_p95=9.0, gap_max=12.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_rejects_high_mse_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": _quality(1000.0, 1.0, step_p95=12.0),
            "fgo": _quality(450.0, 0.1, gap_p95=10.0, gap_max=12.0, step_p95=10.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_keeps_bounded_high_baseline_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": _quality(1000.0, 1.0, step_p95=12.0),
            "fgo": _quality(350.0, 0.1, gap_p95=10.0, gap_max=12.0, step_p95=10.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_rescues_train_backed_high_pr_raw_wls():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": _quality(56.713, 1.0, step_p95=22.106),
            "raw_wls": _quality(15.667, 2.0, gap_p95=94.627, gap_max=123.432, step_p95=60.398),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_select_gated_chunk_source_rejects_relaxed_high_pr_raw_wls():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": _quality(119.0495, 1.0),
            "raw_wls": _quality(44.3967, 0.1, gap_p95=94.0, gap_max=123.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_high_baseline_raw_wls_with_implausible_motion():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(3_519_056.153, 1.0, step_p95=21.221),
            "raw_wls": _quality(
                2_358_467.695,
                376.333,
                gap_p95=10_828.703,
                gap_max=62_901.589,
                step_p95=15_863.786,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_extreme_raw_wls_mse():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(2_809_337_861.644, 1.0, step_p95=120.0),
            "raw_wls": _quality(
                2_246_680_695.226,
                0.1,
                gap_p95=41.438,
                gap_max=48.0,
                step_p95=100.0,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_auto_chunk_source_rejects_extreme_raw_wls_mse():
    candidates = {
        "baseline": _quality(2_809_337_861.644, 1.0),
        "raw_wls": _quality(2_246_680_695.226, 0.1, step_p95=8.0),
    }

    assert select_auto_chunk_source(candidates) == "baseline"


def test_select_gated_chunk_source_prefers_raw_wls_when_high_baseline_fgo_has_worse_pr_mse():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": _quality(593.259, 1.0, step_p95=54.717),
            "raw_wls": _quality(340.022, 0.8, gap_p95=141.044, gap_max=180.0, step_p95=81.115),
            "fgo_no_tdcp": _quality(404.781, 0.2, gap_p95=109.776, gap_max=140.0, step_p95=19.943),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_select_gated_chunk_source_rejects_raw_wls_over_optional_gap_guard():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": _quality(593.259, 1.0, step_p95=54.717),
            "raw_wls": _quality(340.022, 0.8, gap_p95=141.044, gap_max=232.924, step_p95=81.115),
            "fgo_no_tdcp": _quality(404.781, 0.2, gap_p95=109.776, gap_max=190.338, step_p95=19.943),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            raw_wls_max_gap_m=200.0,
        )
        == "baseline"
    )


def test_select_gated_chunk_source_keeps_raw_wls_under_optional_gap_guard():
    record = ChunkSelectionRecord(
        start_epoch=200,
        end_epoch=400,
        auto_source="baseline",
        candidates={
            "baseline": _quality(521.037, 1.0, step_p95=16.057),
            "raw_wls": _quality(440.066, 0.5, gap_p95=76.200, gap_max=86.750, step_p95=18.101),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            raw_wls_max_gap_m=200.0,
        )
        == "raw_wls"
    )


def test_select_gated_chunk_source_rejects_high_pr_raw_wls_above_ratio_guard():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": _quality(55.0, 1.0),
            "raw_wls": _quality(19.5, 0.1, gap_p95=94.0, gap_max=123.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_high_pr_raw_wls_above_mse_guard():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": _quality(200.0, 1.0),
            "raw_wls": _quality(20.5, 0.1, gap_p95=94.0, gap_max=123.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_high_pr_raw_wls_above_gap_guard():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": _quality(56.713, 1.0, step_p95=22.106),
            "raw_wls": _quality(15.667, 0.1, gap_p95=95.0, gap_max=151.0, step_p95=60.398),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_keeps_plausible_catastrophic_raw_wls_rescue():
    record = ChunkSelectionRecord(
        start_epoch=1000,
        end_epoch=1190,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(7_138.392, 1.0, step_p95=14.031),
            "raw_wls": _quality(
                5_897.768,
                1.656,
                gap_p95=15.689,
                gap_max=1_787.316,
                step_p95=19.406,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_add_tdcp_off_fgo_candidates_matches_records_by_span():
    baseline_state = np.zeros((4, 3), dtype=np.float64)
    auto_state = np.zeros((4, 3), dtype=np.float64)
    tdcp_off_fgo_state = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    records = [
        ChunkSelectionRecord(
            start_epoch=0,
            end_epoch=2,
            auto_source="baseline",
            candidates={"baseline": _quality(100.0, 1.0), "fgo": _quality(80.0, 0.8)},
        ),
        ChunkSelectionRecord(
            start_epoch=2,
            end_epoch=4,
            auto_source="baseline",
            candidates={"baseline": _quality(100.0, 1.0), "fgo": _quality(80.0, 0.8)},
        ),
    ]
    tdcp_off_records = [
        ChunkSelectionRecord(
            start_epoch=2,
            end_epoch=4,
            auto_source="fgo",
            candidates={"baseline": _quality(100.0, 1.0), "fgo": _quality(60.0, 0.7)},
        ),
    ]

    add_tdcp_off_fgo_candidates(records, tdcp_off_records, tdcp_off_fgo_state, baseline_state, auto_state)

    assert "fgo_no_tdcp" not in records[0].candidates
    assert records[1].candidates["fgo_no_tdcp"].mse_pr == 60.0


def test_chunk_selection_payload_reports_gated_source_and_candidate_metrics():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": _quality(170.0, 1.0),
            "fgo": _quality(178.0, 0.78, gap_p95=1.5, gap_max=2.0),
        },
    )

    payload = chunk_selection_payload([record], 500.0)

    assert payload[0]["gated_source"] == "fgo"
    assert payload[0]["candidates"]["fgo"]["mse_pr"] == 178.0
