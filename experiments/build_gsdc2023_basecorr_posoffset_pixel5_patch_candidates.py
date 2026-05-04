#!/usr/bin/env python3
"""Build basecorr/position-offset submissions with the Pixel5 LAX-X patch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.apply_gsdc2023_position_offsets import (
    apply_offsets,
    scale_map_for_policy,
    sha256_file,
)
from experiments.audit_gsdc2023_pr_proxy_risk import (
    expand_inputs as expand_pr_proxy_risk_inputs,
    load_guard_rows as load_pr_proxy_guard_rows,
    load_risky_rows as load_pr_proxy_risky_rows,
    summarize as summarize_pr_proxy_risk,
    write_outputs as write_pr_proxy_risk_outputs,
)
from experiments.reproduce_gsdc2023_best_submission import parse_trip_position_overrides, replace_trip_coordinates
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIXEL5_PATCH = (
    REPO_ROOT
    / "experiments/results/source_selection_lowbaseline_submission_probe_20260430/"
    "pixel5_fgo_early_raw_late_final_trip_rows.csv"
)
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments/results/source_selection_lowbaseline_submission_probe_20260430/"
    "basecorr_posoffset_pixel5_patch_source/"
    "basecorr_pixel4xl_base_on_submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/source_selection_lowbaseline_submission_probe_20260430/"
    "basecorr_posoffset_pixel5_patch_scripted"
)

PIXEL5_PATCH_TRIP = "2022-04-04-16-31-us-ca-lax-x/pixel5"
PIXEL5_SJC_R_TRIP = "2022-02-08-22-04-us-ca-sjc-r/pixel5"
PIXEL5_MTV_DE1_20230523_TRIP = "2023-05-23-21-06-us-ca-mtv-de1/pixel5"
PIXEL5_SJC_HE2_20230606_TRIP = "2023-06-06-22-43-us-ca-sjc-he2/pixel5"
PIXEL5_SJC_BE2_20230526_TRIP = "2023-05-26-21-23-us-ca-sjc-be2/pixel5"
PIXEL5_SJC_Q_20230427_TRIP = "2023-04-27-20-55-us-ca-sjc-q/pixel5"
PIXEL5_MTV_PE1_20230427_TRIP = "2023-04-27-19-25-us-ca-mtv-pe1/pixel5"
PIXEL5_MTV_PE1_20220322_TRIP = "2022-03-22-18-44-us-ca-mtv-pe1/pixel5"
PIXEL5_LAX_P_20220224_TRIP = "2022-02-24-15-10-us-ca-lax-p/pixel5"
PIXEL5_LAX_I_20220224_TRIP = "2022-02-24-22-14-us-ca-lax-i/pixel5"
PIXEL5_LAX_M_20220223_TRIP = "2022-02-23-22-35-us-ca-lax-m/pixel5"
PIXEL5_LAX_N_20220223_TRIP = "2022-02-23-17-46-us-ca-lax-n/pixel5"
PIXEL5_EBF_Z_20220425_TRIP = "2022-04-25-22-36-us-ca-ebf-z/pixel5"
PIXEL5_EBF_Y_20220422_TRIP = "2022-04-22-20-11-us-ca-ebf-y/pixel5"
PIXEL5_EBF_XX_20220427_TRIP = "2022-04-27-19-23-us-ca-ebf-xx/pixel5"
PIXEL5_EBF_ZZ_20220427_TRIP = "2022-04-27-18-16-us-ca-ebf-zz/pixel5"
P60523_TRIP = "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"
P60525_TRIP = "2023-05-25-17-32-us-ca-pao-j/pixel6pro"
A325G_0512_TRIP = "2022-05-12-20-19-us-ca-mtv-pe1/samsunga325g"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    phone_scale_overrides: dict[str, float]
    trip_scale_overrides: dict[str, float]
    kaggle_public: float | None = None
    kaggle_private: float | None = None


def _trip_scales(*, a325g0512: float) -> dict[str, float]:
    return {
        P60523_TRIP: 3.25,
        P60525_TRIP: 3.25,
        A325G_0512_TRIP: float(a325g0512),
    }


def _without_pixel6pro_scales(scales: dict[str, float]) -> dict[str, float]:
    return {
        trip: scale
        for trip, scale in scales.items()
        if trip not in {P60523_TRIP, P60525_TRIP}
    }


def _pixel5_3p375_sjc_r0_trip_scales() -> dict[str, float]:
    return {
        **_trip_scales(a325g0512=3.75),
        PIXEL5_SJC_R_TRIP: 0.0,
    }


def _pixel5_3p375_sjc_r_scale_trip_scales(scale: float) -> dict[str, float]:
    return {
        **_trip_scales(a325g0512=3.75),
        PIXEL5_SJC_R_TRIP: float(scale),
    }


def _pixel5_3p375_sjc_r_scale_no_pixel6pro_trip_scales(scale: float) -> dict[str, float]:
    return {
        **_without_pixel6pro_scales(_trip_scales(a325g0512=3.75)),
        PIXEL5_SJC_R_TRIP: float(scale),
    }


def _pixel5_sjc_r0_ablation(name: str, trip: str) -> CandidateConfig:
    return CandidateConfig(
        name=name,
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides={
            **_pixel5_3p375_sjc_r0_trip_scales(),
            trip: 0.0,
        },
    )


def _pixel5_sjc_r0_combo(
    name: str,
    trips: tuple[str, ...],
    *,
    kaggle_public: float,
    kaggle_private: float,
) -> CandidateConfig:
    return CandidateConfig(
        name=name,
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides={
            **_pixel5_3p375_sjc_r0_trip_scales(),
            **{trip: 0.0 for trip in trips},
        },
        kaggle_public=kaggle_public,
        kaggle_private=kaggle_private,
    )


CANDIDATES: dict[str, CandidateConfig] = {
    "sma205u_3p5": CandidateConfig(
        name="sma205u_3p5",
        phone_scale_overrides={"sm-a205u": 3.5, "sm-s908b": 3.25},
        trip_scale_overrides=_trip_scales(a325g0512=3.375),
        kaggle_public=3.725,
        kaggle_private=4.790,
    ),
    "a325g0512_3p75": CandidateConfig(
        name="a325g0512_3p75",
        phone_scale_overrides={"sm-a205u": 3.25, "sm-s908b": 3.25},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.725,
        kaggle_private=4.790,
    ),
    "sms908b_3p125": CandidateConfig(
        name="sms908b_3p125",
        phone_scale_overrides={"sm-a205u": 3.25, "sm-s908b": 3.125},
        trip_scale_overrides=_trip_scales(a325g0512=3.375),
        kaggle_public=3.725,
        kaggle_private=4.791,
    ),
    "combo_sma205u3p5_a325g0512_3p75_sms908b_3p125": CandidateConfig(
        name="combo_sma205u3p5_a325g0512_3p75_sms908b_3p125",
        phone_scale_overrides={"sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.725,
        kaggle_private=4.790,
    ),
    "combo_sma205u3p5_a325g0512_3p75_sms908b_3p25": CandidateConfig(
        name="combo_sma205u3p5_a325g0512_3p75_sms908b_3p25",
        phone_scale_overrides={"sm-a205u": 3.5, "sm-s908b": 3.25},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.725,
        kaggle_private=4.790,
    ),
    "combo_sma205u3p5_a325g0512_3p75_sms908b_3p375": CandidateConfig(
        name="combo_sma205u3p5_a325g0512_3p75_sms908b_3p375",
        phone_scale_overrides={"sm-a205u": 3.5, "sm-s908b": 3.375},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.725,
        kaggle_private=4.790,
    ),
    "pixel5phone_1p875_public_best": CandidateConfig(
        name="pixel5phone_1p875_public_best",
        phone_scale_overrides={"pixel5": 1.875, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.701,
        kaggle_private=4.729,
    ),
    "pixel5phone_3p375_private_best": CandidateConfig(
        name="pixel5phone_3p375_private_best",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_trip_scales(a325g0512=3.75),
        kaggle_public=3.725,
        kaggle_private=4.720,
    ),
    "pixel5phone_3p375_sjc_r0_private_best": CandidateConfig(
        name="pixel5phone_3p375_sjc_r0_private_best",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r0_trip_scales(),
        kaggle_public=3.725,
        kaggle_private=4.710,
    ),
    "pixel5phone_3p375_sjc_r0p421875": CandidateConfig(
        name="pixel5phone_3p375_sjc_r0p421875",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_trip_scales(0.421875),
        kaggle_public=3.725,
        kaggle_private=4.710,
    ),
    "pixel5phone_3p375_sjc_r0p84375": CandidateConfig(
        name="pixel5phone_3p375_sjc_r0p84375",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_trip_scales(0.84375),
        kaggle_public=3.725,
        kaggle_private=4.711,
    ),
    "pixel5phone_3p375_sjc_r1p6875": CandidateConfig(
        name="pixel5phone_3p375_sjc_r1p6875",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_trip_scales(1.6875),
        kaggle_public=3.725,
        kaggle_private=4.713,
    ),
    "pixel5phone_3p375_sjc_r2p53125": CandidateConfig(
        name="pixel5phone_3p375_sjc_r2p53125",
        phone_scale_overrides={"pixel5": 3.375, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_trip_scales(2.53125),
        kaggle_public=3.725,
        kaggle_private=4.716,
    ),
    "pixel5phone_3p375_sjc_r0p84375_p6p0": CandidateConfig(
        name="pixel5phone_3p375_sjc_r0p84375_p6p0",
        phone_scale_overrides={"pixel5": 3.375, "pixel6pro": 0.0, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_no_pixel6pro_trip_scales(0.84375),
    ),
    "pixel5phone_3p375_sjc_r1p6875_p6p0": CandidateConfig(
        name="pixel5phone_3p375_sjc_r1p6875_p6p0",
        phone_scale_overrides={"pixel5": 3.375, "pixel6pro": 0.0, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_no_pixel6pro_trip_scales(1.6875),
    ),
    "pixel5phone_3p375_sjc_r2p53125_p6p0": CandidateConfig(
        name="pixel5phone_3p375_sjc_r2p53125_p6p0",
        phone_scale_overrides={"pixel5": 3.375, "pixel6pro": 0.0, "sm-a205u": 3.5, "sm-s908b": 3.125},
        trip_scale_overrides=_pixel5_3p375_sjc_r_scale_no_pixel6pro_trip_scales(2.53125),
    ),
    "pixel5phone_3p375_sjcr0_ablate_mtv_de1_20230523": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_mtv_de1_20230523",
        PIXEL5_MTV_DE1_20230523_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_sjc_he2_20230606": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_sjc_he2_20230606",
        PIXEL5_SJC_HE2_20230606_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_sjc_be2_20230526": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_sjc_be2_20230526",
        PIXEL5_SJC_BE2_20230526_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_sjc_q_20230427": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_sjc_q_20230427",
        PIXEL5_SJC_Q_20230427_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20230427": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20230427",
        PIXEL5_MTV_PE1_20230427_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20220322": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20220322",
        PIXEL5_MTV_PE1_20220322_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_lax_p_20220224": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_lax_p_20220224",
        PIXEL5_LAX_P_20220224_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_lax_i_20220224": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_lax_i_20220224",
        PIXEL5_LAX_I_20220224_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_lax_m_20220223": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_lax_m_20220223",
        PIXEL5_LAX_M_20220223_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_lax_n_20220223": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_lax_n_20220223",
        PIXEL5_LAX_N_20220223_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_ebf_z_20220425": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_ebf_z_20220425",
        PIXEL5_EBF_Z_20220425_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_ebf_y_20220422": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_ebf_y_20220422",
        PIXEL5_EBF_Y_20220422_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_ebf_xx_20220427": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_ebf_xx_20220427",
        PIXEL5_EBF_XX_20220427_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_ablate_ebf_zz_20220427": _pixel5_sjc_r0_ablation(
        "pixel5phone_3p375_sjcr0_ablate_ebf_zz_20220427",
        PIXEL5_EBF_ZZ_20220427_TRIP,
    ),
    "pixel5phone_3p375_sjcr0_combo_sjcq_ebfzz": _pixel5_sjc_r0_combo(
        "pixel5phone_3p375_sjcr0_combo_sjcq_ebfzz",
        (PIXEL5_SJC_Q_20230427_TRIP, PIXEL5_EBF_ZZ_20220427_TRIP),
        kaggle_public=3.699,
        kaggle_private=4.710,
    ),
    "pixel5phone_3p375_sjcr0_combo_ebfxx_ebfzz": _pixel5_sjc_r0_combo(
        "pixel5phone_3p375_sjcr0_combo_ebfxx_ebfzz",
        (PIXEL5_EBF_XX_20220427_TRIP, PIXEL5_EBF_ZZ_20220427_TRIP),
        kaggle_public=3.690,
        kaggle_private=4.710,
    ),
    "pixel5phone_3p375_sjcr0_combo_sjcq_ebfxx_ebfzz": _pixel5_sjc_r0_combo(
        "pixel5phone_3p375_sjcr0_combo_sjcq_ebfxx_ebfzz",
        (
            PIXEL5_SJC_Q_20230427_TRIP,
            PIXEL5_EBF_XX_20220427_TRIP,
            PIXEL5_EBF_ZZ_20220427_TRIP,
        ),
        kaggle_public=3.687,
        kaggle_private=4.710,
    ),
}


def _require_submission_columns(frame: pd.DataFrame, path: Path) -> None:
    required = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _delta_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    delta_m = haversine_m(
        before["LatitudeDegrees"].to_numpy(),
        before["LongitudeDegrees"].to_numpy(),
        after["LatitudeDegrees"].to_numpy(),
        after["LongitudeDegrees"].to_numpy(),
    )
    out = gsdc_score_m(delta_m)
    out["changed_rows_gt_0p01m"] = int(np.count_nonzero(delta_m > 0.01))
    out["changed_rows_gt_1m"] = int(np.count_nonzero(delta_m > 1.0))
    return out


def _step_summary(frame: pd.DataFrame, trip: str) -> dict[str, Any]:
    trip_frame = frame[frame["tripId"] == trip]
    if len(trip_frame) < 2:
        return {
            "trip": trip,
            "rows": int(len(trip_frame)),
            "max_step_m": None,
            "p95_step_m": None,
            "over80_count": 0,
            "over150_count": 0,
        }
    step_m = haversine_m(
        trip_frame["LatitudeDegrees"].to_numpy()[:-1],
        trip_frame["LongitudeDegrees"].to_numpy()[:-1],
        trip_frame["LatitudeDegrees"].to_numpy()[1:],
        trip_frame["LongitudeDegrees"].to_numpy()[1:],
    )
    return {
        "trip": trip,
        "rows": int(len(trip_frame)),
        "max_step_m": float(np.max(step_m)),
        "p95_step_m": float(np.percentile(step_m, 95)),
        "over80_count": int(np.count_nonzero(step_m > 80.0)),
        "over150_count": int(np.count_nonzero(step_m > 150.0)),
    }


def build_candidate(
    source: pd.DataFrame,
    *,
    input_path: Path,
    pixel5_patch_path: Path,
    extra_trip_patch_paths: dict[str, Path] | None = None,
    output_dir: Path,
    config: CandidateConfig,
    tag: str,
) -> dict[str, Any]:
    candidate_dir = output_dir / config.name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    scales = scale_map_for_policy("phone-tuned", 1.0)
    scales.update(config.phone_scale_overrides)
    offset_frame, trip_summary = apply_offsets(source, scales, config.trip_scale_overrides)

    before_patch_path = candidate_dir / f"{config.name}_before_pixel5_patch.csv"
    trip_summary_path = candidate_dir / f"{config.name}_trip_summary.csv"
    before_patch_summary_path = candidate_dir / f"{config.name}_before_pixel5_patch_summary.json"
    output_path = candidate_dir / f"submission_best_basecorr_posoffset_{config.name}_plus_pixel5_patch_{tag}.csv"
    summary_path = candidate_dir / "build_summary.json"

    offset_frame.to_csv(before_patch_path, index=False)
    trip_summary.to_csv(trip_summary_path, index=False)

    patch_base_frame = pd.read_csv(before_patch_path)
    trip_patch_paths = {PIXEL5_PATCH_TRIP: pixel5_patch_path}
    trip_patch_paths.update(extra_trip_patch_paths or {})
    patch_trips = tuple(trip_patch_paths)
    patched_frame, patch_summary = replace_trip_coordinates(
        patch_base_frame,
        patch_base_frame,
        patch_trips,
        trip_position_overrides=trip_patch_paths,
    )
    patched_frame.to_csv(output_path, index=False)

    before_patch_summary = {
        "input": str(input_path),
        "output": str(before_patch_path),
        "sha256": sha256_file(before_patch_path),
        "policy": "phone-tuned-with-trip-overrides",
        "effective_phone_scales": dict(sorted(scales.items())),
        "phone_scale_overrides": dict(sorted(config.phone_scale_overrides.items())),
        "trip_scale_overrides": dict(sorted(config.trip_scale_overrides.items())),
        "rows": int(len(offset_frame)),
        "nan_lat_lon_rows": int(offset_frame[["LatitudeDegrees", "LongitudeDegrees"]].isna().any(axis=1).sum()),
        "delta_vs_input": _delta_summary(source, offset_frame),
        "trip_summary": str(trip_summary_path),
    }
    before_patch_summary_path.write_text(
        json.dumps(before_patch_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "candidate": config.name,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "pixel5_patch": str(pixel5_patch_path),
        "pixel5_patch_sha256": sha256_file(pixel5_patch_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "before_pixel5_patch": str(before_patch_path),
        "before_pixel5_patch_summary": str(before_patch_summary_path),
        "rows": int(len(patched_frame)),
        "nan_lat_lon_rows": int(patched_frame[["LatitudeDegrees", "LongitudeDegrees"]].isna().any(axis=1).sum()),
        "policy": "phone-tuned-with-trip-overrides",
        "effective_phone_scales": dict(sorted(scales.items())),
        "phone_scale_overrides": dict(sorted(config.phone_scale_overrides.items())),
        "trip_scale_overrides": dict(sorted(config.trip_scale_overrides.items())),
        "pixel5_patch_summary": patch_summary,
        "delta_vs_input": _delta_summary(source, patched_frame),
        "delta_vs_before_pixel5_patch": _delta_summary(patch_base_frame, patched_frame),
        "pixel5_step_summary": _step_summary(patched_frame, PIXEL5_PATCH_TRIP),
        "patched_trip_step_summary": {
            trip: _step_summary(patched_frame, trip)
            for trip in patch_trips
        },
        "kaggle_score_reference": {
            "public": config.kaggle_public,
            "private": config.kaggle_private,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _normalise_trip_id(trip: str) -> str:
    parts = str(trip).strip("/").split("/")
    if parts and parts[0] in {"train", "test"}:
        return "/".join(parts[1:])
    return "/".join(parts)


def _candidate_trip_scale(config: CandidateConfig, trip: str) -> float:
    trip_id = _normalise_trip_id(trip)
    phone = trip_id.rstrip("/").split("/")[-1].lower()
    scales = scale_map_for_policy("phone-tuned", 1.0)
    scales.update(config.phone_scale_overrides)
    return float(config.trip_scale_overrides.get(trip_id, scales.get(phone, 0.0)))


def _candidate_affects_trip(config: CandidateConfig, trip: str) -> bool:
    return _candidate_trip_scale(config, trip) > 0.0


def _candidate_actionable_risk_summary(rows: pd.DataFrame, configs: list[CandidateConfig]) -> dict[str, Any]:
    if rows.empty or not configs:
        return {
            "candidate_actionable_risky_rows": 0,
            "candidate_actionable_risky_chunks": 0,
            "candidate_actionable_by_candidate": {},
        }
    actionable_rows: list[pd.DataFrame] = []
    by_candidate: dict[str, int] = {}
    chunk_keys: set[tuple[str, int, int, str]] = set()
    for config in configs:
        mask = rows["trip"].map(lambda trip: _candidate_affects_trip(config, str(trip)))
        candidate_rows = rows[mask].copy()
        if candidate_rows.empty:
            by_candidate[config.name] = 0
            continue
        candidate_rows["candidate"] = config.name
        actionable_rows.append(candidate_rows)
        unique_chunks = candidate_rows[["trip", "start_epoch", "end_epoch"]].drop_duplicates()
        by_candidate[config.name] = int(len(unique_chunks))
        for chunk in unique_chunks.itertuples(index=False):
            chunk_keys.add((str(chunk.trip), int(chunk.start_epoch), int(chunk.end_epoch), config.name))
    actionable_count = int(sum(len(frame) for frame in actionable_rows))
    return {
        "candidate_actionable_risky_rows": actionable_count,
        "candidate_actionable_risky_chunks": int(len(chunk_keys)),
        "candidate_actionable_by_candidate": by_candidate,
    }


def build_pr_proxy_risk_report(
    metrics_inputs: list[str],
    output_dir: Path,
    configs: list[CandidateConfig] | None = None,
) -> dict[str, Any]:
    paths = expand_pr_proxy_risk_inputs(metrics_inputs)
    rows = load_pr_proxy_risky_rows(paths)
    guard_rows = load_pr_proxy_guard_rows(paths)
    summary = summarize_pr_proxy_risk(rows, len(paths), guard_rows)
    candidate_summary = _candidate_actionable_risk_summary(rows, configs or [])
    summary.update(candidate_summary)
    write_pr_proxy_risk_outputs(output_dir, rows, summary, guard_rows)
    return {
        "enabled": True,
        "input_patterns": list(metrics_inputs),
        "input_files": [str(path) for path in paths],
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        **summary,
    }


def build_candidates(
    input_path: Path,
    pixel5_patch_path: Path,
    output_dir: Path,
    configs: list[CandidateConfig],
    *,
    extra_trip_patch_paths: dict[str, Path] | None = None,
    risk_metrics_inputs: list[str] | None = None,
    risk_report_dir: Path | None = None,
    tag: str,
) -> dict[str, Any]:
    source = pd.read_csv(input_path)
    _require_submission_columns(source, input_path)
    _require_submission_columns(pd.read_csv(pixel5_patch_path), pixel5_patch_path)
    extra_trip_patch_paths = extra_trip_patch_paths or {}
    for patch_path in extra_trip_patch_paths.values():
        _require_submission_columns(pd.read_csv(patch_path), patch_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summaries = [
        build_candidate(
            source,
            input_path=input_path,
            pixel5_patch_path=pixel5_patch_path,
            extra_trip_patch_paths=extra_trip_patch_paths,
            output_dir=output_dir,
            config=config,
            tag=tag,
        )
        for config in configs
    ]
    aggregate = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "pixel5_patch": str(pixel5_patch_path),
        "pixel5_patch_sha256": sha256_file(pixel5_patch_path),
        "extra_trip_patches": {
            trip: {"path": str(path), "sha256": sha256_file(path)}
            for trip, path in sorted(extra_trip_patch_paths.items())
        },
        "output_dir": str(output_dir),
        "tag": tag,
        "candidate_count": len(candidate_summaries),
        "candidates": candidate_summaries,
    }
    if risk_metrics_inputs:
        aggregate["pr_proxy_risk_report"] = build_pr_proxy_risk_report(
            risk_metrics_inputs,
            risk_report_dir or output_dir / "pr_proxy_risk_report",
            configs,
        )
    else:
        aggregate["pr_proxy_risk_report"] = {"enabled": False}
    (output_dir / "build_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _selected_configs(names: list[str] | None) -> list[CandidateConfig]:
    selected = names or sorted(CANDIDATES)
    return [CANDIDATES[name] for name in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pixel5-patch", type=Path, default=DEFAULT_PIXEL5_PATCH)
    parser.add_argument(
        "--trip-patch",
        action="append",
        default=[],
        metavar="TRIP=CSV",
        help="additional trip coordinate override CSV; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="20260501")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=sorted(CANDIDATES),
        help="candidate preset to build; repeatable, defaults to all presets",
    )
    parser.add_argument(
        "--risk-metrics",
        action="append",
        default=[],
        help="bridge_metrics.json path or glob for the PR proxy risk report; repeatable",
    )
    parser.add_argument(
        "--risk-report-dir",
        type=Path,
        default=None,
        help="output directory for PR proxy risk report; defaults to OUTPUT_DIR/pr_proxy_risk_report",
    )
    parser.add_argument(
        "--fail-on-risk",
        action="store_true",
        help="return exit code 2 when --risk-metrics finds risky chunks",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"missing input submission: {args.input}")
    if not args.pixel5_patch.is_file():
        raise SystemExit(f"missing Pixel5 patch CSV: {args.pixel5_patch}")
    extra_trip_patch_paths = parse_trip_position_overrides(args.trip_patch)
    for trip, patch_path in extra_trip_patch_paths.items():
        if not patch_path.is_file():
            raise SystemExit(f"missing patch CSV for {trip}: {patch_path}")

    summary = build_candidates(
        args.input,
        args.pixel5_patch,
        args.output_dir,
        _selected_configs(args.candidate),
        extra_trip_patch_paths=extra_trip_patch_paths,
        risk_metrics_inputs=args.risk_metrics,
        risk_report_dir=args.risk_report_dir,
        tag=args.tag,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    risk_report = summary.get("pr_proxy_risk_report")
    if args.fail_on_risk and isinstance(risk_report, dict):
        fail_chunks = int(
            risk_report.get("candidate_actionable_risky_chunks", risk_report.get("risky_chunks", 0)),
        )
        if fail_chunks > 0:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
