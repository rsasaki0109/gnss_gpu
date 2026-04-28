"""Post-processing and metric assembly for GSDC2023 solver outputs."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from experiments.evaluate import compute_metrics
from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_chunk_selection import ChunkSelectionRecord, chunk_selection_payload
from experiments.gsdc2023_height_constraints import (
    apply_phone_position_offset_state,
    apply_relative_height_constraint,
)
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_output import BridgeResult
from experiments.gsdc2023_result_metadata import bridge_result_metadata_kwargs
from experiments.gsdc2023_solver_selection import SourceSolutionCatalog


@dataclass(frozen=True)
class AssembledSourceOutputs:
    output_states: dict[str, np.ndarray]
    selected_state: np.ndarray
    selected_sources: np.ndarray
    selected_source_counts: dict[str, int]
    selected_mse_pr: float
    truth: np.ndarray | None
    metrics_selected: dict | None
    metrics_kaggle: dict | None
    metrics_raw_wls: dict | None
    metrics_fgo: dict | None


def postprocess_output_state(
    state: np.ndarray,
    *,
    phone_name: str,
    apply_relative_height: bool,
    apply_position_offset: bool,
    reference_wls: np.ndarray,
    stop_epochs: np.ndarray | None,
) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    if apply_relative_height:
        out[:, :3] = apply_relative_height_constraint(out[:, :3], reference_wls, stop_epochs)
    if apply_position_offset:
        out = apply_phone_position_offset_state(out, phone_name)
    return out


def assemble_source_outputs(
    catalog: SourceSolutionCatalog,
    batch: TripArrays,
    config: BridgeConfig,
    *,
    phone_name: str,
) -> AssembledSourceOutputs:
    output_states = {
        name: postprocess_output_state(
            state,
            phone_name=phone_name,
            apply_relative_height=config.apply_relative_height,
            apply_position_offset=config.apply_position_offset,
            reference_wls=batch.kaggle_wls,
            stop_epochs=batch.stop_epochs,
        )
        for name, state in catalog.states.items()
    }

    selected_state = output_states[config.position_source]
    selected_sources = catalog.arrays[config.position_source]
    selected_source_counts = catalog.counts[config.position_source]
    selected_mse_pr = catalog.mse[config.position_source]

    if batch.has_truth:
        truth = batch.truth
        metrics_selected = compute_metrics(selected_state[:, :3], truth)
        metrics_kaggle = compute_metrics(output_states["baseline"][:, :3], truth)
        metrics_raw_wls = compute_metrics(output_states["raw_wls"][:, :3], truth)
        metrics_fgo = compute_metrics(output_states["fgo"][:, :3], truth)
    else:
        truth = None
        metrics_selected = None
        metrics_kaggle = None
        metrics_raw_wls = None
        metrics_fgo = None

    return AssembledSourceOutputs(
        output_states=output_states,
        selected_state=selected_state,
        selected_sources=selected_sources,
        selected_source_counts=selected_source_counts,
        selected_mse_pr=float(selected_mse_pr),
        truth=truth,
        metrics_selected=metrics_selected,
        metrics_kaggle=metrics_kaggle,
        metrics_raw_wls=metrics_raw_wls,
        metrics_fgo=metrics_fgo,
    )


def build_bridge_result(
    *,
    trip: str,
    batch: TripArrays,
    config: BridgeConfig,
    assembled_outputs: AssembledSourceOutputs,
    fgo_iters: int,
    failed_chunks: int,
    baseline_mse_pr: float,
    raw_wls_mse_pr: float,
    fgo_mse_pr: float,
    chunk_records: Sequence[ChunkSelectionRecord],
    allow_raw_wls_on_mi8_baseline_jump: bool,
    raw_wls_max_gap_m: float | None = None,
) -> BridgeResult:
    return BridgeResult(
        trip=trip,
        signal_type=config.signal_type,
        weight_mode=config.weight_mode,
        selected_source_mode=config.position_source,
        times_ms=batch.times_ms,
        kaggle_wls=assembled_outputs.output_states["baseline"][:, :3],
        raw_wls=assembled_outputs.output_states["raw_wls"],
        fgo_state=assembled_outputs.output_states["fgo"],
        selected_state=assembled_outputs.selected_state,
        selected_sources=assembled_outputs.selected_sources,
        truth=assembled_outputs.truth,
        max_sats=batch.max_sats,
        fgo_iters=fgo_iters,
        failed_chunks=failed_chunks,
        selected_mse_pr=assembled_outputs.selected_mse_pr,
        baseline_mse_pr=baseline_mse_pr,
        raw_wls_mse_pr=raw_wls_mse_pr,
        fgo_mse_pr=fgo_mse_pr,
        selected_source_counts=assembled_outputs.selected_source_counts,
        metrics_selected=assembled_outputs.metrics_selected,
        metrics_kaggle=assembled_outputs.metrics_kaggle,
        metrics_raw_wls=assembled_outputs.metrics_raw_wls,
        metrics_fgo=assembled_outputs.metrics_fgo,
        chunk_selection_records=chunk_selection_payload(
            chunk_records,
            config.gated_baseline_threshold,
            allow_raw_wls_on_mi8_baseline_jump=allow_raw_wls_on_mi8_baseline_jump,
            raw_wls_max_gap_m=raw_wls_max_gap_m,
        ),
        **bridge_result_metadata_kwargs(config, batch),
    )


__all__ = [
    "AssembledSourceOutputs",
    "assemble_source_outputs",
    "build_bridge_result",
    "postprocess_output_state",
]
