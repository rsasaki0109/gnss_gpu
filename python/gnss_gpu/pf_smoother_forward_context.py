"""Forward-pass context for PF smoother evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gnss_gpu.pf_smoother_config import PfSmootherConfig, PfSmootherConfigParts
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pf_smoother_run_context import (
    PfSmootherRunDependencies,
    PfSmootherRunOptions,
)
from gnss_gpu.pf_smoother_runtime import (
    ForwardRunBuffers,
    ObservationComputers,
    RunDataset,
)


@dataclass
class PfSmootherForwardPassContext:
    run_name: str
    run_config: PfSmootherConfig
    config_parts: PfSmootherConfigParts
    run_options: PfSmootherRunOptions
    dependencies: PfSmootherRunDependencies
    dataset: RunDataset
    imu_filter: Any
    pf: Any
    buffers: ForwardRunBuffers
    stats: ForwardRunStats
    history: ForwardEpochHistory
    observation_setup: ObservationComputers
    pr_history: dict[int, list[float]]
