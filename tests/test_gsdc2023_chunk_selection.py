import numpy as np

from experiments.gsdc2023_chunk_selection import (
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
    add_fgo_candidate_from_records,
    add_tdcp_off_fgo_candidates,
    chunk_candidate_quality,
    chunk_selection_payload,
    compute_dd_carrier_anchor_coverage_ratio,
    dd_carrier_anchor_coverage_passes,
    is_fgo_candidate_source,
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


def test_fgo_prefixed_ct_rbpf_source_uses_fgo_candidate_policy():
    assert is_fgo_candidate_source("fgo_ct_rbpf") is True


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


def test_select_gated_chunk_source_allows_high_confidence_low_baseline_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=20,
        auto_source="fgo",
        candidates={
            "baseline": _quality(9.322, 1.0, step_p95=15.396),
            "raw_wls": _quality(6.827, 0.851, gap_p95=10.678, gap_max=12.923, step_p95=18.911),
            "fgo": _quality(8.478, 0.723, gap_p95=9.609, gap_max=10.466, step_p95=15.086),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_rejects_weak_low_baseline_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=20,
        auto_source="fgo",
        candidates={
            "baseline": _quality(18.460, 1.0, step_p95=16.0),
            "raw_wls": _quality(14.196, 1.216, gap_p95=16.730, gap_max=17.0, step_p95=18.0),
            "fgo": _quality(16.275, 0.930, gap_p95=16.726, gap_max=17.0, step_p95=17.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_fgo_when_raw_wls_proxy_is_better_by_default():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=80,
        auto_source="fgo",
        candidates={
            "baseline": _quality(20.0, 1.0, step_p95=10.0),
            "raw_wls": _quality(10.0, 0.9, gap_p95=9.0, gap_max=10.0, step_p95=10.0),
            "fgo": _quality(11.0, 0.6, gap_p95=10.0, gap_max=11.0, step_p95=10.0),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_allows_dd_carrier_when_it_matches_safe_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(25.66, 1.0, step_p95=20.6),
            "raw_wls": _quality(20.04, 0.96, gap_p95=19.2, gap_max=36.1, step_p95=22.8),
            "fgo": _quality(22.89, 0.700, gap_p95=12.6, gap_max=16.9, step_p95=20.1),
            "fgo_dd_carrier": _quality(22.88, 0.698, gap_p95=12.5, gap_max=16.8, step_p95=20.2),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo_dd_carrier"


def test_select_gated_chunk_source_rejects_dd_carrier_when_it_is_worse_than_fgo():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(25.66, 1.0, step_p95=20.6),
            "raw_wls": _quality(20.04, 0.96, gap_p95=19.2, gap_max=36.1, step_p95=22.8),
            "fgo": _quality(22.89, 0.700, gap_p95=12.6, gap_max=16.9, step_p95=20.1),
            "fgo_dd_carrier": _quality(23.20, 0.735, gap_p95=12.5, gap_max=16.8, step_p95=20.2),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_dd_carrier_when_baseline_pr_mse_is_low():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(21.50, 1.0, step_p95=14.7),
            "raw_wls": _quality(16.82, 0.94, gap_p95=16.2, gap_max=27.7, step_p95=16.7),
            "fgo": _quality(18.53, 0.669, gap_p95=10.6, gap_max=12.4, step_p95=14.3),
            "fgo_dd_carrier": _quality(18.53, 0.670, gap_p95=10.7, gap_max=12.3, step_p95=14.4),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_dd_carrier_when_pr_gain_is_weak():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(28.85, 1.0, step_p95=18.1),
            "raw_wls": _quality(25.39, 1.20, gap_p95=13.9, gap_max=23.6, step_p95=20.0),
            "fgo": _quality(27.31, 0.687, gap_p95=8.7, gap_max=10.5, step_p95=18.4),
            "fgo_dd_carrier": _quality(27.32, 0.687, gap_p95=8.7, gap_max=10.5, step_p95=18.4),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_dd_carrier_when_baseline_gap_is_large():
    record = ChunkSelectionRecord(
        start_epoch=100,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": _quality(32.94, 1.0, step_p95=17.9),
            "raw_wls": _quality(22.49, 1.19, gap_p95=22.6, gap_max=30.7, step_p95=23.3),
            "fgo": _quality(25.60, 0.800, gap_p95=17.5, gap_max=19.3, step_p95=17.5),
            "fgo_dd_carrier": _quality(25.59, 0.802, gap_p95=17.5, gap_max=19.2, step_p95=17.6),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_keeps_low_baseline_fgo_family_over_dd_carrier():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(8.37, 1.0, step_p95=17.6),
            "raw_wls": _quality(7.62, 0.70, gap_p95=6.4, gap_max=8.8, step_p95=18.4),
            "fgo": _quality(8.27, 0.615, gap_p95=6.0, gap_max=12.8, step_p95=17.4),
            "fgo_dd_carrier": _quality(8.28, 0.614, gap_p95=5.9, gap_max=13.0, step_p95=17.4),
            "fgo_no_tdcp": _quality(8.26, 0.615, gap_p95=5.8, gap_max=12.5, step_p95=17.4),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_allows_opt_in_fgo_over_raw_wls_proxy_rescue():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=80,
        auto_source="fgo",
        candidates={
            "baseline": _quality(20.0, 1.0, step_p95=10.0),
            "raw_wls": _quality(10.0, 0.9, gap_p95=9.0, gap_max=10.0, step_p95=10.0),
            "fgo": _quality(11.0, 0.6, gap_p95=10.0, gap_max=11.0, step_p95=10.0),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            allow_fgo_raw_wls_proxy_rescue=True,
            fgo_raw_wls_proxy_rescue_mse_ratio_max=1.15,
            fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=1.5,
            fgo_raw_wls_proxy_rescue_quality_delta_max=-0.35,
            fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max=0.0,
        )
        == "fgo"
    )


def test_select_gated_chunk_source_rejects_proxy_rescue_above_ratio_limit():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=80,
        auto_source="fgo",
        candidates={
            "baseline": _quality(20.0, 1.0, step_p95=10.0),
            "raw_wls": _quality(10.0, 0.9, gap_p95=9.0, gap_max=10.0, step_p95=10.0),
            "fgo": _quality(12.0, 0.6, gap_p95=10.0, gap_max=11.0, step_p95=10.0),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            allow_fgo_raw_wls_proxy_rescue=True,
            fgo_raw_wls_proxy_rescue_mse_ratio_max=1.15,
        )
        == "baseline"
    )


def test_select_gated_chunk_source_rejects_pixel6pro_2023_raw_proxy_with_low_baseline_pr():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(17.641993, 1.0, step_p95=15.456),
            "fgo": _quality(7.798628, 0.933937, gap_p95=20.0, gap_max=35.090, step_p95=18.436),
            "raw_wls": _quality(7.798628, 0.933937, gap_p95=20.0, gap_max=35.090, step_p95=18.436),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


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


def test_select_gated_chunk_source_rejects_high_baseline_raw_wls_with_weak_quality():
    record = ChunkSelectionRecord(
        start_epoch=120,
        end_epoch=150,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(83_651.338, 1.0, gap_max=0.0, step_p95=11.854),
            "raw_wls": _quality(
                51_807.647,
                2.228,
                gap_p95=10.219,
                gap_max=2_788.731,
                step_p95=13.464,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_marginal_high_baseline_raw_wls_quality():
    record = ChunkSelectionRecord(
        start_epoch=200,
        end_epoch=400,
        auto_source="baseline",
        candidates={
            "baseline": _quality(521.037, 1.0, step_p95=16.057),
            "raw_wls": _quality(
                440.066,
                1.838,
                gap_p95=76.200,
                gap_max=86.750,
                step_p95=18.101,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_high_baseline_raw_wls_with_extreme_gap_p95():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": _quality(3_546_611.397, 1.0, step_p95=8_151.345),
            "raw_wls": _quality(
                20.644,
                0.164,
                gap_p95=6_635.967,
                gap_max=30_767.303,
                step_p95=19.204,
            ),
        },
    )

    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_high_baseline_fgo_with_extreme_gap_p95():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo_no_tdcp",
        candidates={
            "baseline": _quality(3_546_611.397, 1.0, step_p95=8_151.345),
            "fgo_no_tdcp": _quality(
                33.467,
                0.163,
                gap_p95=6_629.535,
                gap_max=30_773.423,
                step_p95=9.125,
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


def test_add_fgo_candidate_from_records_adds_named_ct_candidate():
    records = [
        ChunkSelectionRecord(
            start_epoch=0,
            end_epoch=2,
            auto_source="baseline",
            candidates={"baseline": _quality(100.0, 1.0)},
        ),
    ]
    candidate_records = [
        ChunkSelectionRecord(
            start_epoch=0,
            end_epoch=2,
            auto_source="fgo",
            candidates={
                "baseline": _quality(100.0, 1.0),
                "fgo": _quality(55.0, 0.6),
            },
        ),
    ]
    candidate_state = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    baseline_state = np.zeros((2, 3), dtype=np.float64)
    auto_state = baseline_state.copy()

    add_fgo_candidate_from_records(
        records,
        candidate_records,
        source_name="fgo_ct_rbpf",
        candidate_state=candidate_state,
        baseline_state=baseline_state,
        auto_state=auto_state,
    )

    assert records[0].candidates["fgo_ct_rbpf"].mse_pr == 55.0
    assert records[0].candidates["fgo_ct_rbpf"].baseline_gap_max_m == 2.0


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


# --- Phase 74 DD anchor coverage gate ---------------------------------------


def _dd_carrier_record_passing_safe_fgo() -> ChunkSelectionRecord:
    """Build a record where ``fgo_dd_carrier`` would normally win the gate.

    Mirrors the fixture from
    ``test_select_gated_chunk_source_allows_dd_carrier_when_it_matches_safe_fgo``
    so the only varying input across the anchor-coverage tests is the
    per-trip anchor coverage signal.
    """

    return ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(25.66, 1.0, step_p95=20.6),
            "raw_wls": _quality(20.04, 0.96, gap_p95=19.2, gap_max=36.1, step_p95=22.8),
            "fgo": _quality(22.89, 0.700, gap_p95=12.6, gap_max=16.9, step_p95=20.1),
            "fgo_dd_carrier": _quality(22.88, 0.698, gap_p95=12.5, gap_max=16.8, step_p95=20.2),
        },
    )


def test_compute_dd_carrier_anchor_coverage_ratio_basic():
    assert compute_dd_carrier_anchor_coverage_ratio(20, 100) == 0.2
    assert compute_dd_carrier_anchor_coverage_ratio(80, 100) == 0.8


def test_compute_dd_carrier_anchor_coverage_ratio_returns_none_for_empty_trip():
    assert compute_dd_carrier_anchor_coverage_ratio(20, 0) is None
    assert compute_dd_carrier_anchor_coverage_ratio(20, -3) is None


def test_compute_dd_carrier_anchor_coverage_ratio_clamps_negative_numerator():
    assert compute_dd_carrier_anchor_coverage_ratio(-5, 100) == 0.0


def test_dd_carrier_anchor_coverage_passes_returns_true_when_signal_is_missing():
    # Default behaviour: gate is opt-in, so an unknown coverage must not
    # suppress DD-carrier emission.
    assert dd_carrier_anchor_coverage_passes(None) is True
    assert dd_carrier_anchor_coverage_passes(float("nan")) is True


def test_dd_carrier_anchor_coverage_passes_respects_threshold():
    assert dd_carrier_anchor_coverage_passes(0.6) is True
    assert dd_carrier_anchor_coverage_passes(0.599) is False
    assert dd_carrier_anchor_coverage_passes(0.5, min_coverage=0.4) is True
    assert dd_carrier_anchor_coverage_passes(0.3, min_coverage=0.4) is False


def test_select_gated_chunk_source_default_unchanged_when_no_anchor_coverage_supplied():
    record = _dd_carrier_record_passing_safe_fgo()

    # Default call path mirrors every existing caller — gate is inactive.
    assert select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo_dd_carrier"


def test_select_gated_chunk_source_suppresses_dd_carrier_below_anchor_coverage():
    record = _dd_carrier_record_passing_safe_fgo()

    chosen = select_gated_chunk_source(
        record,
        baseline_threshold=500.0,
        dd_carrier_anchor_coverage=0.3,  # below 0.6 default → suppress
    )

    # The dd-carrier branch is skipped.  In this fixture the plain ``fgo``
    # candidate is itself blocked by the raw-WLS PR-MSE guard, so the gate
    # falls back to ``baseline`` — the intentional safe-default behaviour
    # when DD-carrier cannot be trusted at the trip level.
    assert chosen == "baseline"


def test_select_gated_chunk_source_keeps_dd_carrier_at_or_above_anchor_coverage():
    record = _dd_carrier_record_passing_safe_fgo()

    chosen = select_gated_chunk_source(
        record,
        baseline_threshold=500.0,
        dd_carrier_anchor_coverage=0.6,
    )

    assert chosen == "fgo_dd_carrier"


def test_select_gated_chunk_source_honours_custom_min_anchor_coverage():
    record = _dd_carrier_record_passing_safe_fgo()

    # Tighten the threshold to 0.8 — a 0.7 coverage now fails the gate.
    chosen_tight = select_gated_chunk_source(
        record,
        baseline_threshold=500.0,
        dd_carrier_anchor_coverage=0.7,
        dd_carrier_min_anchor_coverage=0.8,
    )
    # Same fixture as above: with DD-carrier suppressed the gate falls back
    # to ``baseline`` (the plain ``fgo`` candidate is also blocked by the
    # raw-WLS PR-MSE guard).
    assert chosen_tight == "baseline"

    # Loosen the threshold to 0.5 — a 0.55 coverage now passes the gate.
    chosen_loose = select_gated_chunk_source(
        record,
        baseline_threshold=500.0,
        dd_carrier_anchor_coverage=0.55,
        dd_carrier_min_anchor_coverage=0.5,
    )
    assert chosen_loose == "fgo_dd_carrier"


def test_select_gated_chunk_source_low_coverage_falls_back_to_fgo_when_fgo_independently_passes():
    # No raw-WLS candidate present, so the plain ``fgo`` sibling does NOT
    # hit the raw-WLS PR-MSE guard and can win after DD-carrier is gated.
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(25.66, 1.0, step_p95=20.6),
            "fgo": _quality(22.89, 0.700, gap_p95=12.6, gap_max=16.9, step_p95=20.1),
            "fgo_dd_carrier": _quality(22.88, 0.698, gap_p95=12.5, gap_max=16.8, step_p95=20.2),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            dd_carrier_anchor_coverage=0.3,
        )
        == "fgo"
    )

    # Sanity: with high coverage the DD-carrier candidate still wins because
    # it is a strictly safer match against the FGO baseline.
    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            dd_carrier_anchor_coverage=0.8,
        )
        == "fgo_dd_carrier"
    )


def test_select_gated_chunk_source_does_not_affect_non_dd_carrier_paths():
    # A record where the winning candidate is plain ``fgo`` (no DD-carrier
    # candidate present) must be unaffected by the anchor coverage parameter.
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=100,
        auto_source="fgo",
        candidates={
            "baseline": _quality(25.66, 1.0, step_p95=20.6),
            "fgo": _quality(22.89, 0.700, gap_p95=12.6, gap_max=16.9, step_p95=20.1),
        },
    )

    assert (
        select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            dd_carrier_anchor_coverage=0.0,
        )
        == "fgo"
    )


def test_chunk_selection_payload_threads_anchor_coverage_into_gate():
    record = _dd_carrier_record_passing_safe_fgo()

    payload_below = chunk_selection_payload(
        [record], 500.0, dd_carrier_anchor_coverage=0.3
    )
    payload_above = chunk_selection_payload(
        [record], 500.0, dd_carrier_anchor_coverage=0.7
    )
    payload_default = chunk_selection_payload([record], 500.0)

    assert payload_below[0]["gated_source"] == "baseline"
    assert payload_above[0]["gated_source"] == "fgo_dd_carrier"
    # Backward-compatible default keeps the legacy DD-carrier emission.
    assert payload_default[0]["gated_source"] == "fgo_dd_carrier"


def test_default_min_anchor_coverage_constant_matches_audit_framework():
    # The production default must equal the audit framework's
    # ``DDAnchorGate.min_anchor_coverage`` so empirical thresholds stay in
    # lock-step.  Drift would silently break the Phase 74 A/B replay.
    assert DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT == 0.6
