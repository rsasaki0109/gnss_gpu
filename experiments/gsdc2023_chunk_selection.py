"""Chunk-level source selection for GSDC2023 raw bridge outputs.

This module contains the pure policy used to choose among baseline, raw WLS,
FGO, and TDCP-off FGO candidates.  It has no CSV, solver, or raw-bridge
orchestration dependencies, which keeps selection rules independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GATED_BASELINE_THRESHOLD_DEFAULT = 500.0
GATED_FGO_BASELINE_MSE_PR_MIN = 20.0
GATED_FGO_BASELINE_GAP_P95_FLOOR_M = 12.0
GATED_CANDIDATE_QUALITY_MARGIN = 0.08
GATED_TDCP_OFF_CANDIDATE_MARGIN = 0.03
GATED_TDCP_BASELINE_GAP_INCREASE_MARGIN_M = 0.15
GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN = 50.0
GATED_RAW_WLS_RESCUE_MSE_PR_MAX = 20.0
GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX = 0.35
GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M = 150.0
GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_RATIO_MAX = 2.0
GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_FLOOR_M = 100.0
GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_RATIO_MAX = 3.0
GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_FLOOR_M = 150.0
GATED_MI8_BASELINE_JUMP_STEP_P95_M = 100.0
GATED_MI8_RAW_WLS_BASELINE_GAP_MAX_M = 200.0
# Pixel5 raw_wls high-baseline chunks improved the local PR proxy but worsened
# Kaggle Private in direct A/B, so keep them disabled until train-backed masked
# evidence or a direct follow-up A/B proves a narrower exception.
GATED_PIXEL5_RAW_WLS_BASELINE_GAP_MAX_M = 0.0
WINDOW_SELECTION_STEP_P95_MAX_M = 30.0
CATASTROPHIC_BASELINE_GAP_MAX_M = 1000.0


@dataclass(frozen=True)
class ChunkCandidateQuality:
    mse_pr: float
    step_mean_m: float
    step_p95_m: float
    accel_mean_m: float
    accel_p95_m: float
    bridge_jump_m: float
    baseline_gap_mean_m: float
    baseline_gap_p95_m: float
    baseline_gap_max_m: float
    quality_score: float


@dataclass(frozen=True)
class ChunkSelectionRecord:
    start_epoch: int
    end_epoch: int
    auto_source: str
    candidates: dict[str, ChunkCandidateQuality]


def trajectory_motion_stats(xyz: np.ndarray) -> tuple[float, float, float, float]:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] <= 1:
        return 0.0, 0.0, 0.0, 0.0
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if step.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    accel = np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1) if xyz.shape[0] > 2 else np.zeros(0, dtype=np.float64)
    return (
        float(np.mean(step)),
        float(np.percentile(step, 95)),
        float(np.mean(accel)) if accel.size else 0.0,
        float(np.percentile(accel, 95)) if accel.size else 0.0,
    )


def quality_ratio(value: float, reference: float, floor: float) -> float:
    return float(value) / max(float(reference), float(floor))


def chunk_candidate_quality(
    state: np.ndarray,
    mse_pr: float,
    baseline_quality: ChunkCandidateQuality | None,
    prev_tail_xyz: np.ndarray | None,
    baseline_xyz: np.ndarray | None = None,
) -> ChunkCandidateQuality:
    xyz = np.asarray(state[:, :3], dtype=np.float64)
    step_mean_m, step_p95_m, accel_mean_m, accel_p95_m = trajectory_motion_stats(xyz)
    bridge_jump_m = float(np.linalg.norm(xyz[0] - prev_tail_xyz)) if prev_tail_xyz is not None and len(xyz) > 0 else 0.0
    baseline_gap_mean_m = 0.0
    baseline_gap_p95_m = 0.0
    baseline_gap_max_m = 0.0
    if baseline_xyz is not None and len(xyz) > 0:
        gap = np.linalg.norm(xyz - np.asarray(baseline_xyz, dtype=np.float64).reshape(-1, 3), axis=1)
        baseline_gap_mean_m = float(np.mean(gap))
        baseline_gap_p95_m = float(np.percentile(gap, 95))
        baseline_gap_max_m = float(np.max(gap))

    if baseline_quality is None:
        quality_score = 1.0
    else:
        quality_score = (
            0.40 * quality_ratio(mse_pr, baseline_quality.mse_pr, 10.0)
            + 0.10 * quality_ratio(step_mean_m, baseline_quality.step_mean_m, 0.25)
            + 0.10 * quality_ratio(step_p95_m, baseline_quality.step_p95_m, 0.5)
            + 0.10 * quality_ratio(accel_p95_m, baseline_quality.accel_p95_m, 0.5)
            + 0.10 * quality_ratio(bridge_jump_m, baseline_quality.bridge_jump_m, 0.5)
            + 0.20 * (baseline_gap_p95_m / max(baseline_quality.step_p95_m, 5.0))
        )

    return ChunkCandidateQuality(
        mse_pr=float(mse_pr),
        step_mean_m=step_mean_m,
        step_p95_m=step_p95_m,
        accel_mean_m=accel_mean_m,
        accel_p95_m=accel_p95_m,
        bridge_jump_m=bridge_jump_m,
        baseline_gap_mean_m=baseline_gap_mean_m,
        baseline_gap_p95_m=baseline_gap_p95_m,
        baseline_gap_max_m=baseline_gap_max_m,
        quality_score=float(quality_score),
    )


def candidate_passes_high_baseline_mse_motion_guard(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality,
) -> bool:
    return (
        quality.step_p95_m
        <= max(
            baseline.step_p95_m * GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_RATIO_MAX,
            GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_FLOOR_M,
        )
        and quality.baseline_gap_p95_m
        <= max(
            baseline.step_p95_m * GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_RATIO_MAX,
            GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_FLOOR_M,
        )
    )


def catastrophic_baseline_alternative(
    candidates: dict[str, ChunkCandidateQuality],
    *,
    baseline_mse_threshold: float | None = None,
    raw_wls_max_gap_m: float | None = None,
) -> str | None:
    baseline = candidates["baseline"]
    if baseline_mse_threshold is not None and baseline.mse_pr <= float(baseline_mse_threshold):
        return None
    eligible: list[tuple[float, float, str]] = []
    for name, quality in candidates.items():
        if name == "baseline":
            continue
        if name == "raw_wls" and not raw_wls_candidate_passes_max_gap_guard(quality, raw_wls_max_gap_m):
            continue
        if quality.baseline_gap_max_m < CATASTROPHIC_BASELINE_GAP_MAX_M:
            continue
        if quality.mse_pr >= baseline.mse_pr:
            continue
        if not candidate_passes_high_baseline_mse_motion_guard(quality, baseline):
            continue
        eligible.append((quality.mse_pr, quality.baseline_gap_max_m, name))
    if not eligible:
        return None
    eligible.sort()
    return eligible[0][2]


def select_auto_chunk_source(candidates: dict[str, ChunkCandidateQuality]) -> str:
    baseline = candidates["baseline"]
    catastrophic_source = catastrophic_baseline_alternative(candidates)
    if catastrophic_source is not None:
        return catastrophic_source
    if baseline.step_p95_m > WINDOW_SELECTION_STEP_P95_MAX_M:
        return "baseline"
    best_source = "baseline"
    best_score = baseline.quality_score
    for name, quality in candidates.items():
        if name == "baseline":
            continue
        if quality.mse_pr > baseline.mse_pr * 1.5:
            continue
        if quality.quality_score + 0.03 < best_score:
            best_source = name
            best_score = quality.quality_score
    return best_source


def is_fgo_candidate_source(name: str) -> bool:
    return name == "fgo" or name.startswith("fgo_")


def fgo_candidate_passes_baseline_gap_guard(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality,
) -> bool:
    if baseline.mse_pr < GATED_FGO_BASELINE_MSE_PR_MIN:
        return False
    return quality.baseline_gap_p95_m <= max(
        baseline.step_p95_m,
        GATED_FGO_BASELINE_GAP_P95_FLOOR_M,
    )


def fgo_candidate_passes_raw_wls_mse_guard(
    quality: ChunkCandidateQuality,
    raw_wls: ChunkCandidateQuality | None,
) -> bool:
    return raw_wls is None or quality.mse_pr <= raw_wls.mse_pr


def raw_wls_candidate_passes_mi8_baseline_jump_guard(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality,
) -> bool:
    return (
        baseline.step_p95_m >= GATED_MI8_BASELINE_JUMP_STEP_P95_M
        and quality.baseline_gap_max_m <= GATED_MI8_RAW_WLS_BASELINE_GAP_MAX_M
        and quality.mse_pr <= baseline.mse_pr * 0.90
        and quality.quality_score + GATED_CANDIDATE_QUALITY_MARGIN < baseline.quality_score
    )


def raw_wls_candidate_passes_max_gap_guard(
    quality: ChunkCandidateQuality,
    max_gap_m: float | None,
) -> bool:
    return max_gap_m is None or quality.baseline_gap_max_m <= float(max_gap_m)


def raw_wls_candidate_passes_high_pr_mse_rescue(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality,
) -> bool:
    return (
        np.isfinite(baseline.mse_pr)
        and np.isfinite(quality.mse_pr)
        and np.isfinite(quality.baseline_gap_max_m)
        and baseline.mse_pr >= GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN
        and quality.mse_pr <= GATED_RAW_WLS_RESCUE_MSE_PR_MAX
        and quality.mse_pr <= baseline.mse_pr * GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX
        and quality.baseline_gap_max_m <= GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M
    )


def candidate_passes_gated_quality(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality,
) -> bool:
    return (
        quality.mse_pr <= baseline.mse_pr * 1.12
        and quality.quality_score + GATED_CANDIDATE_QUALITY_MARGIN < baseline.quality_score
    )


def select_gated_chunk_source(
    record: ChunkSelectionRecord,
    baseline_threshold: float,
    *,
    allow_raw_wls_on_mi8_baseline_jump: bool = False,
    raw_wls_max_gap_m: float | None = None,
) -> str:
    baseline = record.candidates["baseline"]
    catastrophic_source = catastrophic_baseline_alternative(
        record.candidates,
        baseline_mse_threshold=baseline_threshold,
        raw_wls_max_gap_m=raw_wls_max_gap_m,
    )
    if catastrophic_source is not None:
        return catastrophic_source
    baseline_jump_override = (
        allow_raw_wls_on_mi8_baseline_jump
        and baseline.step_p95_m >= GATED_MI8_BASELINE_JUMP_STEP_P95_M
    )
    if (
        baseline.step_p95_m > WINDOW_SELECTION_STEP_P95_MAX_M
        and baseline.mse_pr <= float(baseline_threshold)
        and not baseline_jump_override
    ):
        return "baseline"
    candidate_order = sorted(
        (quality.quality_score, name, quality)
        for name, quality in record.candidates.items()
        if name != "baseline"
    )
    if not candidate_order:
        return "baseline"

    if baseline.mse_pr > float(baseline_threshold):
        raw_wls = record.candidates.get("raw_wls")
        for _score, name, quality in candidate_order:
            if name == "raw_wls" and not raw_wls_candidate_passes_max_gap_guard(quality, raw_wls_max_gap_m):
                continue
            if is_fgo_candidate_source(name) and not fgo_candidate_passes_raw_wls_mse_guard(quality, raw_wls):
                continue
            if quality.mse_pr < baseline.mse_pr and candidate_passes_high_baseline_mse_motion_guard(quality, baseline):
                return name
        return "baseline"
    for _score, name, quality in candidate_order:
        if name == "raw_wls":
            if not raw_wls_candidate_passes_max_gap_guard(quality, raw_wls_max_gap_m):
                continue
            if raw_wls_candidate_passes_high_pr_mse_rescue(quality, baseline):
                return name
            if not (
                allow_raw_wls_on_mi8_baseline_jump
                and raw_wls_candidate_passes_mi8_baseline_jump_guard(quality, baseline)
            ):
                continue
        if is_fgo_candidate_source(name):
            if not fgo_candidate_passes_baseline_gap_guard(quality, baseline):
                continue
            if not fgo_candidate_passes_raw_wls_mse_guard(quality, record.candidates.get("raw_wls")):
                continue
        if name == "fgo":
            tdcp_off_fgo = record.candidates.get("fgo_no_tdcp")
            if (
                tdcp_off_fgo is not None
                and fgo_candidate_passes_baseline_gap_guard(tdcp_off_fgo, baseline)
                and candidate_passes_gated_quality(tdcp_off_fgo, baseline)
                and quality.baseline_gap_p95_m
                > tdcp_off_fgo.baseline_gap_p95_m + GATED_TDCP_BASELINE_GAP_INCREASE_MARGIN_M
            ):
                continue
        if name == "fgo_no_tdcp":
            tdcp_fgo = record.candidates.get("fgo")
            if tdcp_fgo is not None and fgo_candidate_passes_baseline_gap_guard(tdcp_fgo, baseline):
                tdcp_gap_increased = (
                    tdcp_fgo.baseline_gap_p95_m
                    > quality.baseline_gap_p95_m + GATED_TDCP_BASELINE_GAP_INCREASE_MARGIN_M
                )
                if (
                    not tdcp_gap_increased
                    and quality.quality_score + GATED_TDCP_OFF_CANDIDATE_MARGIN >= tdcp_fgo.quality_score
                ):
                    continue
        if candidate_passes_gated_quality(quality, baseline):
            return name
    return "baseline"


def chunk_quality_payload(quality: ChunkCandidateQuality) -> dict[str, float]:
    return {
        "mse_pr": float(quality.mse_pr),
        "step_mean_m": float(quality.step_mean_m),
        "step_p95_m": float(quality.step_p95_m),
        "accel_mean_m": float(quality.accel_mean_m),
        "accel_p95_m": float(quality.accel_p95_m),
        "bridge_jump_m": float(quality.bridge_jump_m),
        "baseline_gap_mean_m": float(quality.baseline_gap_mean_m),
        "baseline_gap_p95_m": float(quality.baseline_gap_p95_m),
        "baseline_gap_max_m": float(quality.baseline_gap_max_m),
        "quality_score": float(quality.quality_score),
    }


def chunk_selection_payload(
    records: list[ChunkSelectionRecord],
    gated_threshold: float,
    *,
    allow_raw_wls_on_mi8_baseline_jump: bool = False,
    raw_wls_max_gap_m: float | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "start_epoch": int(record.start_epoch),
            "end_epoch": int(record.end_epoch),
            "auto_source": str(record.auto_source),
            "gated_source": select_gated_chunk_source(
                record,
                gated_threshold,
                allow_raw_wls_on_mi8_baseline_jump=allow_raw_wls_on_mi8_baseline_jump,
                raw_wls_max_gap_m=raw_wls_max_gap_m,
            ),
            "candidates": {
                name: chunk_quality_payload(quality)
                for name, quality in sorted(record.candidates.items())
            },
        }
        for record in records
    ]


def add_tdcp_off_fgo_candidates(
    records: list[ChunkSelectionRecord],
    tdcp_off_records: list[ChunkSelectionRecord],
    tdcp_off_fgo_state: np.ndarray,
    baseline_state: np.ndarray,
    auto_state: np.ndarray,
) -> None:
    tdcp_off_by_span = {
        (record.start_epoch, record.end_epoch): record
        for record in tdcp_off_records
    }
    for record in records:
        tdcp_off_record = tdcp_off_by_span.get((record.start_epoch, record.end_epoch))
        if tdcp_off_record is None or "fgo" not in tdcp_off_record.candidates:
            continue
        start = record.start_epoch
        end = record.end_epoch
        prev_tail_xyz = auto_state[start - 1, :3] if start > 0 else None
        record.candidates["fgo_no_tdcp"] = chunk_candidate_quality(
            tdcp_off_fgo_state[start:end],
            tdcp_off_record.candidates["fgo"].mse_pr,
            baseline_quality=record.candidates["baseline"],
            prev_tail_xyz=prev_tail_xyz,
            baseline_xyz=baseline_state[start:end, :3],
        )
