"""Epoch observation input preparation for PF smoother forward passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.carrier_anchor_rows import select_carrier_anchor_rows
from gnss_gpu.pf_smoother_config import ObservationConfig
from gnss_gpu.pseudorange_weighting import apply_pseudorange_weighting


@dataclass(frozen=True)
class EpochObservationInputs:
    sat_ecef: np.ndarray
    pseudoranges: np.ndarray
    weights: np.ndarray
    carrier_anchor_rows: dict[tuple[int, int], dict[str, object]]


def build_epoch_observation_inputs(
    measurements,
    spp_position_ecef: np.ndarray,
    pr_history: dict[int, list[float]],
    observations: ObservationConfig,
) -> EpochObservationInputs:
    rows = list(measurements)
    sat_ecef = np.array([m.satellite_ecef for m in rows])
    pseudoranges = np.array([m.corrected_pseudorange for m in rows])
    base_weights = np.array([m.weight for m in rows])
    spp_position = np.asarray(spp_position_ecef, dtype=np.float64)

    carrier_anchor_rows = select_carrier_anchor_rows(
        rows,
        pseudoranges,
        spp_position,
        observations.mupf,
        observations.carrier_rescue,
    )

    weights = apply_pseudorange_weighting(
        rows,
        sat_ecef,
        pseudoranges,
        base_weights,
        spp_position,
        pr_history,
        residual_downweight=observations.robust.residual_downweight,
        residual_threshold=observations.robust.residual_threshold,
        pr_accel_downweight=observations.robust.pr_accel_downweight,
        pr_accel_threshold=observations.robust.pr_accel_threshold,
    )

    return EpochObservationInputs(
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        carrier_anchor_rows=carrier_anchor_rows,
    )
