"""Carrier bias tracker epoch-update helper."""

from __future__ import annotations

import numpy as np

from gnss_gpu.carrier_rescue import (
    CarrierAnchorAttempt,
    CarrierBiasState,
    _propagate_carrier_bias_tracker_tdcp,
    _update_carrier_bias_tracker,
)
from gnss_gpu.pf_smoother_config import CarrierRescueConfig


def update_carrier_bias_tracker_after_epoch(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    anchor_attempt: CarrierAnchorAttempt,
    receiver_state: np.ndarray,
    tow: float,
    dd_carrier_result,
    config: CarrierRescueConfig,
) -> int:
    if not config.anchor_enabled:
        return 0

    receiver_state = np.asarray(receiver_state, dtype=np.float64)
    if (
        carrier_rows
        and dd_carrier_result is not None
        and int(getattr(dd_carrier_result, "n_dd", 0)) >= int(config.anchor_seed_dd_min_pairs)
    ):
        _update_carrier_bias_tracker(
            tracker,
            carrier_rows,
            receiver_state,
            tow,
            blend_alpha=config.anchor_blend_alpha,
            reanchor_jump_cycles=config.anchor_reanchor_jump_cycles,
            max_age_s=config.anchor_max_age_s,
            trusted=True,
            max_continuity_residual_m=config.anchor_max_continuity_residual_m,
        )
        return 0

    if anchor_attempt.used and anchor_attempt.rows_used:
        _update_carrier_bias_tracker(
            tracker,
            anchor_attempt.rows_used,
            receiver_state,
            tow,
            blend_alpha=config.anchor_blend_alpha,
            reanchor_jump_cycles=config.anchor_reanchor_jump_cycles,
            max_age_s=config.anchor_max_age_s,
            trusted=False,
            max_continuity_residual_m=config.anchor_max_continuity_residual_m,
        )
        return 0

    if carrier_rows and anchor_attempt.state is not None and not anchor_attempt.used:
        anchor_attempt.propagated_rows = _propagate_carrier_bias_tracker_tdcp(
            tracker,
            carrier_rows,
            anchor_attempt.state,
            tow,
            blend_alpha=config.anchor_blend_alpha,
            reanchor_jump_cycles=config.anchor_reanchor_jump_cycles,
            max_age_s=config.anchor_max_age_s,
            max_continuity_residual_m=config.anchor_max_continuity_residual_m,
        )
        return int(anchor_attempt.propagated_rows)

    return 0
