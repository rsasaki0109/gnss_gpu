"""Validation preflight helpers for GSDC2023 raw bridge entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.gsdc2023_bridge_config import (
    BridgeConfig,
    OUTLIER_REFINEMENT_CHUNK_EPOCHS,
    should_refine_outlier_result,
)
from experiments.gsdc2023_clock_state import (
    effective_multi_gnss_enabled,
    effective_position_source,
)


ParityAuditFn = Callable[[Path, str], dict[str, Any]]


@dataclass(frozen=True)
class RawTripValidationContext:
    config: BridgeConfig
    parity_audit: dict[str, Any]
    trip_dir: Path


def effective_validation_config(trip: str, config: BridgeConfig) -> BridgeConfig:
    effective_multi_gnss = effective_multi_gnss_enabled(trip, config.multi_gnss)
    effective_source = effective_position_source(trip, config.position_source)
    if effective_multi_gnss == config.multi_gnss and effective_source == config.position_source:
        return config
    return replace(
        config,
        multi_gnss=effective_multi_gnss,
        position_source=effective_source,
    )


def require_base_correction_ready(config: BridgeConfig, parity_audit: dict[str, Any]) -> None:
    if config.apply_base_correction and not parity_audit.get("base_correction_ready", False):
        raise RuntimeError(
            "base correction requested but MATLAB parity inputs are not ready: "
            f"{parity_audit.get('base_correction_status', 'unknown')}",
        )


def build_raw_trip_validation_context(
    data_root: Path,
    trip: str,
    config: BridgeConfig | None,
    *,
    parity_audit_fn: ParityAuditFn,
) -> RawTripValidationContext:
    cfg = config or BridgeConfig()
    parity_audit = parity_audit_fn(data_root, trip)
    require_base_correction_ready(cfg, parity_audit)
    return RawTripValidationContext(
        config=effective_validation_config(trip, cfg),
        parity_audit=parity_audit,
        trip_dir=data_root / trip,
    )


def max_epochs_for_build(max_epochs: int) -> int:
    return max_epochs if max_epochs > 0 else 1_000_000_000


def outlier_refinement_config(config: BridgeConfig, selected_mse_pr: float) -> BridgeConfig | None:
    if not should_refine_outlier_result(config.position_source, config.chunk_epochs, selected_mse_pr):
        return None
    return replace(config, chunk_epochs=OUTLIER_REFINEMENT_CHUNK_EPOCHS)


__all__ = [
    "ParityAuditFn",
    "RawTripValidationContext",
    "build_raw_trip_validation_context",
    "effective_validation_config",
    "max_epochs_for_build",
    "outlier_refinement_config",
    "require_base_correction_ready",
]
