"""Solver source-catalog and gated-selection helpers for GSDC2023."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_chunk_selection import (
    ChunkSelectionRecord,
    DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_GAP_STEP_P95_RATIO_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_DELTA_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_RATIO_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_QUALITY_DELTA_MAX,
    GATED_PIXEL5_RAW_WLS_BASELINE_GAP_MAX_M,
    select_gated_chunk_source,
)
from experiments.gsdc2023_clock_state import MULTI_GNSS_BLOCKLIST_PHONES
from experiments.gsdc2023_observation_matrix import TripArrays


@dataclass(frozen=True)
class SourceSolutionCatalog:
    states: dict[str, np.ndarray]
    arrays: dict[str, np.ndarray]
    counts: dict[str, dict[str, int]]
    mse: dict[str, float]

    def selected(self, source: str) -> tuple[np.ndarray, np.ndarray, dict[str, int], float]:
        return (
            self.states[source],
            self.arrays[source],
            self.counts[source],
            self.mse[source],
        )


def tdcp_off_candidate_enabled(config: BridgeConfig, batch: TripArrays) -> bool:
    if config.position_source != "gated" or not config.use_vd or not config.tdcp_enabled:
        return False
    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        return False
    return bool(np.any(batch.tdcp_weights > 0.0))


def tdcp_scale_candidate_enabled(config: BridgeConfig, batch: TripArrays, phone_name: str) -> bool:
    if not config.tdcp_scale_candidate_enabled:
        return False
    if config.position_source != "gated" or not config.use_vd or not config.tdcp_enabled:
        return False
    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        return False
    phones = {str(phone).lower() for phone in config.tdcp_scale_candidate_phones}
    if phone_name.lower() not in phones:
        return False
    if float(config.tdcp_scale_candidate_weight_scale) == float(config.tdcp_weight_scale):
        return False
    return bool(np.any(batch.tdcp_weights > 0.0))


def fgo_raw_wls_proxy_rescue_enabled(config: BridgeConfig, phone_name: str) -> bool:
    if not config.fgo_raw_wls_proxy_rescue_enabled:
        return False
    if config.position_source != "gated":
        return False
    phones = {str(phone).lower() for phone in config.fgo_raw_wls_proxy_rescue_phones}
    return phone_name.lower() in phones


def batch_without_tdcp(batch: TripArrays) -> TripArrays:
    return replace(
        batch,
        tdcp_meas=None,
        tdcp_weights=None,
        tdcp_consistency_mask_count=0,
        tdcp_geometry_correction_count=0,
    )


def mi8_gated_baseline_jump_guard_enabled(phone_name: str, position_source: str) -> bool:
    return phone_name.lower() in MULTI_GNSS_BLOCKLIST_PHONES and position_source == "gated"


def raw_wls_max_gap_guard_m(phone_name: str, position_source: str) -> float | None:
    if phone_name.lower() == "pixel5" and position_source == "gated":
        return GATED_PIXEL5_RAW_WLS_BASELINE_GAP_MAX_M
    return None


def fixed_source_array(n_epoch: int, source: str) -> np.ndarray:
    return np.full(int(n_epoch), str(source), dtype=object)


def fixed_source_counts(
    n_epoch: int,
    source: str,
    *,
    base_sources: tuple[str, ...] = ("baseline", "raw_wls", "fgo"),
) -> dict[str, int]:
    counts = {name: 0 for name in base_sources}
    counts.setdefault(source, 0)
    counts[source] = int(n_epoch)
    return counts


def normalized_source_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(name): int(count) for name, count in counts.items()}


def build_source_solution_catalog(
    *,
    n_epoch: int,
    baseline_state: np.ndarray,
    raw_state: np.ndarray,
    fgo_state: np.ndarray,
    auto_state: np.ndarray,
    auto_sources: np.ndarray,
    auto_source_counts: Mapping[str, int],
    baseline_mse_pr: float,
    raw_wls_mse_pr: float,
    fgo_mse_pr: float,
    auto_mse_pr: float,
) -> SourceSolutionCatalog:
    n = int(n_epoch)
    return SourceSolutionCatalog(
        states={
            "baseline": baseline_state,
            "raw_wls": raw_state,
            "fgo": fgo_state,
            "auto": auto_state,
        },
        arrays={
            "baseline": fixed_source_array(n, "baseline"),
            "raw_wls": fixed_source_array(n, "raw_wls"),
            "fgo": fixed_source_array(n, "fgo"),
            "auto": np.asarray(auto_sources, dtype=object),
        },
        counts={
            "baseline": fixed_source_counts(n, "baseline"),
            "raw_wls": fixed_source_counts(n, "raw_wls"),
            "fgo": fixed_source_counts(n, "fgo"),
            "auto": normalized_source_counts(auto_source_counts),
        },
        mse={
            "baseline": float(baseline_mse_pr),
            "raw_wls": float(raw_wls_mse_pr),
            "fgo": float(fgo_mse_pr),
            "auto": float(auto_mse_pr),
        },
    )


def with_fixed_source_solution(
    catalog: SourceSolutionCatalog,
    *,
    source: str,
    state: np.ndarray,
    mse_pr: float,
) -> SourceSolutionCatalog:
    n_epoch = int(state.shape[0])
    return with_source_solution(
        catalog,
        source=source,
        state=state,
        source_array=fixed_source_array(n_epoch, source),
        source_counts=fixed_source_counts(n_epoch, source),
        mse_pr=mse_pr,
    )


def with_source_solution(
    catalog: SourceSolutionCatalog,
    *,
    source: str,
    state: np.ndarray,
    source_array: np.ndarray,
    source_counts: Mapping[str, int],
    mse_pr: float,
) -> SourceSolutionCatalog:
    states = dict(catalog.states)
    arrays = dict(catalog.arrays)
    counts = {name: dict(value) for name, value in catalog.counts.items()}
    mse = dict(catalog.mse)

    states[source] = state
    arrays[source] = np.asarray(source_array, dtype=object)
    counts[source] = normalized_source_counts(source_counts)
    mse[source] = float(mse_pr)
    return SourceSolutionCatalog(states=states, arrays=arrays, counts=counts, mse=mse)


def select_gated_solution(
    catalog: SourceSolutionCatalog,
    records: list[ChunkSelectionRecord],
    *,
    n_epoch: int,
    baseline_threshold: float,
    allow_raw_wls_on_mi8_baseline_jump: bool = False,
    raw_wls_max_gap_m: float | None = None,
    allow_fgo_raw_wls_proxy_rescue: bool = False,
    fgo_raw_wls_proxy_rescue_mse_ratio_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_RATIO_MAX,
    fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_GAP_STEP_P95_RATIO_MAX,
    fgo_raw_wls_proxy_rescue_quality_delta_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_QUALITY_DELTA_MAX,
    fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_DELTA_MAX,
    dd_carrier_anchor_coverage: float | None = None,
    dd_carrier_min_anchor_coverage: float = DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    gated_state = catalog.states["baseline"].copy()
    gated_sources = fixed_source_array(n_epoch, "baseline")
    gated_counts = fixed_source_counts(n_epoch, "baseline")
    for source in catalog.states:
        if source.startswith("fgo_"):
            gated_counts.setdefault(source, 0)

    for record in records:
        chosen_source = select_gated_chunk_source(
            record,
            baseline_threshold,
            allow_raw_wls_on_mi8_baseline_jump=allow_raw_wls_on_mi8_baseline_jump,
            raw_wls_max_gap_m=raw_wls_max_gap_m,
            allow_fgo_raw_wls_proxy_rescue=allow_fgo_raw_wls_proxy_rescue,
            fgo_raw_wls_proxy_rescue_mse_ratio_max=fgo_raw_wls_proxy_rescue_mse_ratio_max,
            fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max,
            fgo_raw_wls_proxy_rescue_quality_delta_max=fgo_raw_wls_proxy_rescue_quality_delta_max,
            fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max=fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max,
            dd_carrier_anchor_coverage=dd_carrier_anchor_coverage,
            dd_carrier_min_anchor_coverage=dd_carrier_min_anchor_coverage,
        )
        if chosen_source == "baseline":
            continue
        start = int(record.start_epoch)
        end = int(record.end_epoch)
        gated_state[start:end] = catalog.states[chosen_source][start:end]
        gated_sources[start:end] = chosen_source
        gated_counts["baseline"] -= end - start
        gated_counts.setdefault(chosen_source, 0)
        gated_counts[chosen_source] += end - start
    return gated_state, gated_sources, gated_counts


__all__ = [
    "SourceSolutionCatalog",
    "batch_without_tdcp",
    "build_source_solution_catalog",
    "fixed_source_array",
    "fixed_source_counts",
    "fgo_raw_wls_proxy_rescue_enabled",
    "mi8_gated_baseline_jump_guard_enabled",
    "normalized_source_counts",
    "raw_wls_max_gap_guard_m",
    "select_gated_solution",
    "tdcp_off_candidate_enabled",
    "tdcp_scale_candidate_enabled",
    "with_fixed_source_solution",
    "with_source_solution",
]
