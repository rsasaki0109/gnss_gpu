"""Prediction step for one PF smoother forward epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.pf_smoother_epoch_state import EpochForwardState, create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.predict_motion import apply_epoch_predict_motion, select_predict_sigma


@dataclass(frozen=True)
class ForwardEpochPredictResult:
    epoch_state: EpochForwardState
    tow: float
    tow_key: float
    dt: float
    sigma_predict: float


def apply_forward_epoch_prediction(
    context: PfSmootherForwardPassContext,
    sol_epoch: Any,
    measurements: list[Any],
) -> ForwardEpochPredictResult:
    run_config = context.run_config
    history = context.history
    pf = context.pf

    tow = sol_epoch.time.tow
    tow_key = round(tow, 1)
    dt = history.dt_for(tow)
    epoch_state = create_epoch_forward_state(run_config.dd_pseudorange_sigma)

    apply_epoch_predict_motion(
        epoch_state,
        context.stats,
        history,
        imu_filter=context.imu_filter,
        options=context.run_options.predict_motion_options,
        tow=tow,
        tow_key=tow_key,
        dt=dt,
        receiver_position_ecef=np.asarray(sol_epoch.position_ecef_m[:3], dtype=np.float64),
        current_pf_position_ecef=np.asarray(pf.estimate()[:3], dtype=np.float64),
        measurements=measurements,
        spp_lookup=context.dataset.spp_lookup,
        ecef_to_lla_func=context.dependencies.ecef_to_lla_func,
    )

    sigma_predict = select_predict_sigma(
        run_config.sigma_pos,
        imu_stop_detected=epoch_state.imu_stop_detected,
        imu_stop_sigma_pos=run_config.imu_stop_sigma_pos,
        used_tdcp=epoch_state.used_tdcp,
        sigma_pos_tdcp=run_config.sigma_pos_tdcp,
        sigma_pos_tdcp_tight=run_config.sigma_pos_tdcp_tight,
        tdcp_rms=epoch_state.tdcp_rms,
        tdcp_tight_rms_max_m=run_config.tdcp_tight_rms_max_m,
    )

    pf.predict(
        velocity=epoch_state.velocity,
        dt=dt,
        sigma_pos=sigma_predict,
        rbpf_velocity_kf=run_config.rbpf_velocity_kf,
        velocity_process_noise=run_config.rbpf_velocity_process_noise,
    )

    return ForwardEpochPredictResult(
        epoch_state=epoch_state,
        tow=tow,
        tow_key=tow_key,
        dt=dt,
        sigma_predict=sigma_predict,
    )
