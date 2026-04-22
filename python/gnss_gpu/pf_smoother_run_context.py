"""Shared context objects for PF smoother run orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnss_gpu.local_fgo_bridge import _local_fgo_enabled
from gnss_gpu.pf_smoother_config import PfSmootherConfig
from gnss_gpu.predict_motion import EpochPredictMotionOptions


@dataclass(frozen=True)
class PfSmootherRunDependencies:
    load_dataset_func: Callable[[Path, str], Mapping[str, Any]]
    ecef_to_lla_func: Callable[[float, float, float], tuple[float, float, float]]
    compute_metrics_func: Callable[..., dict[str, Any]]
    ecef_errors_func: Callable[..., tuple[np.ndarray, np.ndarray]]
    sigma_cb: float
    seed: int = 42


@dataclass(frozen=True)
class PfSmootherRunOptions:
    fgo_motion_source: str
    need_fgo_tdcp_motion: bool
    predict_motion_options: EpochPredictMotionOptions


def build_pf_smoother_run_options(run_config: PfSmootherConfig) -> PfSmootherRunOptions:
    if run_config.dd_pseudorange and run_config.use_gmm:
        raise ValueError("dd_pseudorange cannot be combined with --gmm")
    if run_config.rbpf_velocity_kf and run_config.doppler_per_particle:
        raise ValueError("--rbpf-velocity-kf and --doppler-per-particle are mutually exclusive")

    fgo_motion_source = str(run_config.fgo_local_motion_source).strip().lower()
    if fgo_motion_source not in {"predict", "tdcp", "prefer_tdcp"}:
        raise ValueError(
            "fgo_local_motion_source must be one of: predict, tdcp, prefer_tdcp"
        )

    tdcp_pu_gate_logic_norm = str(run_config.tdcp_pu_gate_logic).strip().lower()
    if tdcp_pu_gate_logic_norm not in {"any", "all"}:
        raise ValueError("tdcp_pu_gate_logic must be one of: any, all")
    tdcp_pu_gate_stop_mode_norm = str(run_config.tdcp_pu_gate_stop_mode).strip().lower()
    if tdcp_pu_gate_stop_mode_norm not in {"any", "stopped", "moving"}:
        raise ValueError("tdcp_pu_gate_stop_mode must be one of: any, stopped, moving")

    need_fgo_tdcp_motion = (
        _local_fgo_enabled(run_config.fgo_local_window)
        and fgo_motion_source in {"tdcp", "prefer_tdcp"}
    )
    return PfSmootherRunOptions(
        fgo_motion_source=fgo_motion_source,
        need_fgo_tdcp_motion=need_fgo_tdcp_motion,
        predict_motion_options=EpochPredictMotionOptions(
            predict_guide=run_config.predict_guide,
            tdcp_position_update=run_config.tdcp_position_update,
            tdcp_elevation_weight=run_config.tdcp_elevation_weight,
            tdcp_el_sin_floor=run_config.tdcp_el_sin_floor,
            tdcp_rms_threshold=run_config.tdcp_rms_threshold,
            tdcp_pu_rms_max=run_config.tdcp_pu_rms_max,
            tdcp_pu_spp_max_diff_mps=run_config.tdcp_pu_spp_max_diff_mps,
            need_fgo_tdcp_motion=need_fgo_tdcp_motion,
            fgo_local_tdcp_rms_max_m=run_config.fgo_local_tdcp_rms_max_m,
            fgo_local_tdcp_spp_max_diff_mps=run_config.fgo_local_tdcp_spp_max_diff_mps,
        ),
    )
