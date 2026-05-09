from __future__ import annotations

import json
import pandas as pd

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    CANDIDATES,
    PIXEL5_EBF_XX_20220427_TRIP,
    PIXEL5_EBF_Y_20220422_TRIP,
    PIXEL5_EBF_Z_20220425_TRIP,
    PIXEL5_EBF_ZZ_20220427_TRIP,
    PIXEL5_LAX_I_20220224_TRIP,
    PIXEL5_LAX_M_20220223_TRIP,
    PIXEL5_LAX_N_20220223_TRIP,
    PIXEL5_LAX_P_20220224_TRIP,
    PIXEL5_MTV_DE1_20230523_TRIP,
    PIXEL5_MTV_PE1_20220322_TRIP,
    PIXEL5_MTV_PE1_20230427_TRIP,
    PIXEL5_PATCH_TRIP,
    PIXEL5_SJC_BE2_20230526_TRIP,
    PIXEL5_SJC_HE2_20230606_TRIP,
    PIXEL5_SJC_Q_20230427_TRIP,
    PIXEL5_SJC_R_TRIP,
    build_candidates,
)


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [
                PIXEL5_PATCH_TRIP,
                PIXEL5_PATCH_TRIP,
                "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
                "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
                "2022-05-12-20-19-us-ca-mtv-pe1/samsunga325g",
                "2022-05-12-20-19-us-ca-mtv-pe1/samsunga325g",
                "2022-10-06-20-46-us-ca-sjc-r/sm-a205u",
                "2022-10-06-20-46-us-ca-sjc-r/sm-a205u",
                "2023-05-25-21-50-us-ca-sjc-ke2/sm-s908b",
                "2023-05-25-21-50-us-ca-sjc-ke2/sm-s908b",
                PIXEL5_SJC_R_TRIP,
                PIXEL5_SJC_R_TRIP,
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000, 1000, 2000, 1000, 2000, 1000, 2000, 1000, 2000],
            "LatitudeDegrees": [
                37.00000,
                37.00001,
                37.10000,
                37.10001,
                37.20000,
                37.20001,
                37.30000,
                37.30001,
                37.40000,
                37.40001,
                37.50000,
                37.50001,
            ],
            "LongitudeDegrees": [
                -122.00000,
                -121.99999,
                -122.10000,
                -122.09999,
                -122.20000,
                -122.19999,
                -122.30000,
                -122.29999,
                -122.40000,
                -122.39999,
                -122.50000,
                -122.49999,
            ],
        },
    )


def test_candidate_presets_capture_kaggle_scale_choices() -> None:
    combo = CANDIDATES["combo_sma205u3p5_a325g0512_3p75_sms908b_3p125"]
    private_best = CANDIDATES["pixel5phone_3p375_private_best"]
    private_best_sjc_r0 = CANDIDATES["pixel5phone_3p375_sjc_r0_private_best"]
    public_best = CANDIDATES["pixel5phone_1p875_public_best"]

    assert combo.phone_scale_overrides == {"sm-a205u": 3.5, "sm-s908b": 3.125}
    assert combo.trip_scale_overrides["2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"] == 3.25
    assert combo.trip_scale_overrides["2023-05-25-17-32-us-ca-pao-j/pixel6pro"] == 3.25
    assert combo.trip_scale_overrides["2022-05-12-20-19-us-ca-mtv-pe1/samsunga325g"] == 3.75
    assert combo.kaggle_public == 3.725
    assert combo.kaggle_private == 4.790
    assert private_best.phone_scale_overrides["pixel5"] == 3.375
    assert private_best.kaggle_public == 3.725
    assert private_best.kaggle_private == 4.720
    assert private_best_sjc_r0.phone_scale_overrides["pixel5"] == 3.375
    assert private_best_sjc_r0.trip_scale_overrides[PIXEL5_SJC_R_TRIP] == 0.0
    assert private_best_sjc_r0.kaggle_public == 3.725
    assert private_best_sjc_r0.kaggle_private == 4.710
    assert public_best.phone_scale_overrides["pixel5"] == 1.875
    assert public_best.kaggle_public == 3.701
    assert public_best.kaggle_private == 4.729


def test_sjc_r_scale_sweep_presets_start_from_pixel5_3p375_private_best() -> None:
    expected = {
        "pixel5phone_3p375_sjc_r0p421875": (0.421875, 4.710),
        "pixel5phone_3p375_sjc_r0p84375": (0.84375, 4.711),
        "pixel5phone_3p375_sjc_r1p6875": (1.6875, 4.713),
        "pixel5phone_3p375_sjc_r2p53125": (2.53125, 4.716),
    }

    for candidate_name, (scale, private_score) in expected.items():
        candidate = CANDIDATES[candidate_name]

        assert candidate.phone_scale_overrides["pixel5"] == 3.375
        assert candidate.trip_scale_overrides[PIXEL5_SJC_R_TRIP] == scale
        assert candidate.kaggle_public == 3.725
        assert candidate.kaggle_private == private_score


def test_pixel6pro_zero_presets_do_not_offset_risky_pixel6pro_trips() -> None:
    candidate = CANDIDATES["pixel5phone_3p375_sjc_r0p84375_p6p0"]

    assert candidate.phone_scale_overrides["pixel6pro"] == 0.0
    assert "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro" not in candidate.trip_scale_overrides
    assert "2023-05-25-17-32-us-ca-pao-j/pixel6pro" not in candidate.trip_scale_overrides
    assert candidate.trip_scale_overrides[PIXEL5_SJC_R_TRIP] == 0.84375


def test_pixel5_trip_ablation_presets_start_from_sjc_r0_private_best() -> None:
    ablations = {
        "pixel5phone_3p375_sjcr0_ablate_mtv_de1_20230523": PIXEL5_MTV_DE1_20230523_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_sjc_he2_20230606": PIXEL5_SJC_HE2_20230606_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_sjc_be2_20230526": PIXEL5_SJC_BE2_20230526_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_sjc_q_20230427": PIXEL5_SJC_Q_20230427_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20230427": PIXEL5_MTV_PE1_20230427_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20220322": PIXEL5_MTV_PE1_20220322_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_lax_p_20220224": PIXEL5_LAX_P_20220224_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_lax_i_20220224": PIXEL5_LAX_I_20220224_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_lax_m_20220223": PIXEL5_LAX_M_20220223_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_lax_n_20220223": PIXEL5_LAX_N_20220223_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_ebf_z_20220425": PIXEL5_EBF_Z_20220425_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_ebf_y_20220422": PIXEL5_EBF_Y_20220422_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_ebf_xx_20220427": PIXEL5_EBF_XX_20220427_TRIP,
        "pixel5phone_3p375_sjcr0_ablate_ebf_zz_20220427": PIXEL5_EBF_ZZ_20220427_TRIP,
    }

    for candidate_name, ablated_trip in ablations.items():
        candidate = CANDIDATES[candidate_name]

        assert candidate.phone_scale_overrides["pixel5"] == 3.375
        assert candidate.trip_scale_overrides[PIXEL5_SJC_R_TRIP] == 0.0
        assert candidate.trip_scale_overrides[ablated_trip] == 0.0
        assert candidate.kaggle_public is None
        assert candidate.kaggle_private is None


def test_public_private_neutral_combo_presets_capture_scores() -> None:
    combo = CANDIDATES["pixel5phone_3p375_sjcr0_combo_sjcq_ebfxx_ebfzz"]
    pair = CANDIDATES["pixel5phone_3p375_sjcr0_combo_ebfxx_ebfzz"]

    assert combo.phone_scale_overrides["pixel5"] == 3.375
    assert combo.trip_scale_overrides[PIXEL5_SJC_R_TRIP] == 0.0
    assert combo.trip_scale_overrides[PIXEL5_SJC_Q_20230427_TRIP] == 0.0
    assert combo.trip_scale_overrides[PIXEL5_EBF_XX_20220427_TRIP] == 0.0
    assert combo.trip_scale_overrides[PIXEL5_EBF_ZZ_20220427_TRIP] == 0.0
    assert combo.kaggle_public == 3.687
    assert combo.kaggle_private == 4.710
    assert pair.kaggle_public == 3.690
    assert pair.kaggle_private == 4.710


def test_build_candidates_applies_offsets_and_pixel5_patch(tmp_path) -> None:
    source = _submission_rows()
    input_path = tmp_path / "basecorr_source.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    output_dir = tmp_path / "out"
    source.to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "tripId": [PIXEL5_PATCH_TRIP, PIXEL5_PATCH_TRIP],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [38.0, 38.00001],
            "LongitudeDegrees": [-123.0, -122.99999],
        },
    ).to_csv(patch_path, index=False)

    summary = build_candidates(
        input_path,
        patch_path,
        output_dir,
        [CANDIDATES["pixel5phone_3p375_sjc_r0_private_best"]],
        tag="test",
    )

    candidate = summary["candidates"][0]
    output = pd.read_csv(candidate["output"])
    pixel5_rows = output[output["tripId"] == PIXEL5_PATCH_TRIP]
    source_sjc_rows = source[source["tripId"] == PIXEL5_SJC_R_TRIP]
    output_sjc_rows = output[output["tripId"] == PIXEL5_SJC_R_TRIP]

    assert summary["candidate_count"] == 1
    assert candidate["rows"] == len(source)
    assert candidate["nan_lat_lon_rows"] == 0
    assert candidate["pixel5_patch_summary"]["rows_replaced"] == 2
    assert candidate["pixel5_step_summary"]["over80_count"] == 0
    assert pixel5_rows["LatitudeDegrees"].tolist() == [38.0, 38.00001]
    assert pixel5_rows["LongitudeDegrees"].tolist() == [-123.0, -122.99999]
    assert output_sjc_rows["LatitudeDegrees"].tolist() == source_sjc_rows["LatitudeDegrees"].tolist()
    assert output_sjc_rows["LongitudeDegrees"].tolist() == source_sjc_rows["LongitudeDegrees"].tolist()
    assert (output_dir / "build_summary.json").is_file()
    assert (output_dir / "pixel5phone_3p375_sjc_r0_private_best" / "build_summary.json").is_file()


def test_build_candidates_writes_pr_proxy_risk_report(tmp_path) -> None:
    source = _submission_rows()
    input_path = tmp_path / "basecorr_source.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    metrics_path = tmp_path / "bridge_metrics.json"
    output_dir = tmp_path / "out"
    source.to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "tripId": [PIXEL5_PATCH_TRIP, PIXEL5_PATCH_TRIP],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [38.0, 38.00001],
            "LongitudeDegrees": [-123.0, -122.99999],
        },
    ).to_csv(patch_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "trip": "test/course/pixel6pro",
                "vd_seed_guard_records": [
                    {
                        "chunk_start_epoch": 0,
                        "chunk_end_epoch": 200,
                        "segment_start_epoch": 0,
                        "segment_end_epoch": 200,
                        "segment_epochs": 200,
                        "doppler_count": 100,
                        "doppler_rms_mps": 123.0,
                        "tdcp_count": 90,
                        "tdcp_rms_m": 4.0,
                        "reject_reason": "doppler",
                    },
                ],
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 200,
                        "auto_source": "raw_wls",
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": {
                                "mse_pr": 20.0,
                                "step_mean_m": 4.0,
                                "step_p95_m": 10.0,
                                "accel_mean_m": 2.0,
                                "accel_p95_m": 6.0,
                                "bridge_jump_m": 0.0,
                                "baseline_gap_mean_m": 0.0,
                                "baseline_gap_p95_m": 0.0,
                                "baseline_gap_max_m": 0.0,
                                "quality_score": 1.0,
                            },
                            "raw_wls": {
                                "mse_pr": 10.0,
                                "step_mean_m": 5.0,
                                "step_p95_m": 12.0,
                                "accel_mean_m": 3.0,
                                "accel_p95_m": 8.0,
                                "bridge_jump_m": 0.0,
                                "baseline_gap_mean_m": 10.0,
                                "baseline_gap_p95_m": 18.0,
                                "baseline_gap_max_m": 30.0,
                                "quality_score": 0.9,
                            },
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    summary = build_candidates(
        input_path,
        patch_path,
        output_dir,
        [CANDIDATES["pixel5phone_3p375_sjc_r0_private_best"]],
        risk_metrics_inputs=[str(metrics_path)],
        tag="test",
    )

    report = summary["pr_proxy_risk_report"]
    assert report["enabled"] is True
    assert report["risky_chunks"] == 1
    assert report["candidate_actionable_risky_chunks"] == 1
    assert report["vd_guard_rows"] == 1
    assert report["vd_guard_reject_reasons"] == {"doppler": 1}
    assert (output_dir / "pr_proxy_risk_report" / "summary.json").is_file()
    risk_rows = pd.read_csv(output_dir / "pr_proxy_risk_report" / "pr_proxy_risk_chunks.csv")
    assert risk_rows.loc[0, "vd_guard_overlap_segments"] == 1
    assert risk_rows.loc[0, "vd_guard_reject_reasons"] == "doppler"


def test_build_candidates_can_record_non_actionable_pixel6pro_risk_for_p6p0_candidate(tmp_path) -> None:
    source = _submission_rows()
    input_path = tmp_path / "basecorr_source.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    metrics_path = tmp_path / "bridge_metrics.json"
    output_dir = tmp_path / "out"
    source.to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "tripId": [PIXEL5_PATCH_TRIP, PIXEL5_PATCH_TRIP],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [38.0, 38.00001],
            "LongitudeDegrees": [-123.0, -122.99999],
        },
    ).to_csv(patch_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "trip": "test/2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 200,
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": {
                                "mse_pr": 20.0,
                                "step_mean_m": 4.0,
                                "step_p95_m": 10.0,
                                "accel_mean_m": 2.0,
                                "accel_p95_m": 6.0,
                                "bridge_jump_m": 0.0,
                                "baseline_gap_mean_m": 0.0,
                                "baseline_gap_p95_m": 0.0,
                                "baseline_gap_max_m": 0.0,
                                "quality_score": 1.0,
                            },
                            "raw_wls": {
                                "mse_pr": 10.0,
                                "step_mean_m": 5.0,
                                "step_p95_m": 12.0,
                                "accel_mean_m": 3.0,
                                "accel_p95_m": 8.0,
                                "bridge_jump_m": 0.0,
                                "baseline_gap_mean_m": 10.0,
                                "baseline_gap_p95_m": 18.0,
                                "baseline_gap_max_m": 30.0,
                                "quality_score": 0.9,
                            },
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    summary = build_candidates(
        input_path,
        patch_path,
        output_dir,
        [CANDIDATES["pixel5phone_3p375_sjc_r0p84375_p6p0"]],
        risk_metrics_inputs=[str(metrics_path)],
        tag="test",
    )

    report = summary["pr_proxy_risk_report"]
    assert report["risky_chunks"] == 1
    assert report["candidate_actionable_risky_chunks"] == 0
    assert report["candidate_actionable_by_candidate"] == {"pixel5phone_3p375_sjc_r0p84375_p6p0": 0}


def test_build_candidates_applies_extra_trip_patch(tmp_path) -> None:
    source = pd.concat(
        [
            _submission_rows(),
            pd.DataFrame(
                {
                    "tripId": [PIXEL5_SJC_Q_20230427_TRIP, PIXEL5_SJC_Q_20230427_TRIP],
                    "UnixTimeMillis": [3000, 4000],
                    "LatitudeDegrees": [37.60000, 37.60001],
                    "LongitudeDegrees": [-122.60000, -122.59999],
                },
            ),
        ],
        ignore_index=True,
    )
    input_path = tmp_path / "basecorr_source.csv"
    pixel5_patch_path = tmp_path / "pixel5_patch.csv"
    sjc_q_patch_path = tmp_path / "sjc_q_patch.csv"
    output_dir = tmp_path / "out"
    source.to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "tripId": [PIXEL5_PATCH_TRIP, PIXEL5_PATCH_TRIP],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [38.0, 38.00001],
            "LongitudeDegrees": [-123.0, -122.99999],
        },
    ).to_csv(pixel5_patch_path, index=False)
    pd.DataFrame(
        {
            "tripId": [PIXEL5_SJC_Q_20230427_TRIP, PIXEL5_SJC_Q_20230427_TRIP],
            "UnixTimeMillis": [3000, 4000],
            "LatitudeDegrees": [39.0, 39.00001],
            "LongitudeDegrees": [-124.0, -123.99999],
        },
    ).to_csv(sjc_q_patch_path, index=False)

    summary = build_candidates(
        input_path,
        pixel5_patch_path,
        output_dir,
        [CANDIDATES["pixel5phone_3p375_sjc_r0_private_best"]],
        extra_trip_patch_paths={PIXEL5_SJC_Q_20230427_TRIP: sjc_q_patch_path},
        tag="test",
    )

    candidate = summary["candidates"][0]
    output = pd.read_csv(candidate["output"])
    sjc_q_rows = output[output["tripId"] == PIXEL5_SJC_Q_20230427_TRIP]

    assert summary["extra_trip_patches"][PIXEL5_SJC_Q_20230427_TRIP]["path"] == str(sjc_q_patch_path)
    assert candidate["pixel5_patch_summary"]["rows_replaced"] == 4
    assert set(candidate["patched_trip_step_summary"]) == {PIXEL5_PATCH_TRIP, PIXEL5_SJC_Q_20230427_TRIP}
    assert sjc_q_rows["LatitudeDegrees"].tolist() == [39.0, 39.00001]
    assert sjc_q_rows["LongitudeDegrees"].tolist() == [-124.0, -123.99999]
