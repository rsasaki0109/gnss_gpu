"""Epoch observation input preparation for PF smoother forward passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.carrier_anchor_rows import select_carrier_anchor_rows
from gnss_gpu.nlos_mask import NlosMaskTables, apply_mask_to_weights
from gnss_gpu.pf_smoother_config import ObservationConfig
from gnss_gpu.pseudorange_weighting import apply_pseudorange_weighting


def _measurement_prn(measurement) -> str:
    prn = getattr(measurement, "prn", None)
    if prn is not None:
        return str(prn).strip()
    sat_id = getattr(measurement, "satellite_id", None)
    if sat_id is not None:
        return str(sat_id).strip()
    return ""


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
    *,
    epoch_idx: int | None = None,
    nlos_tables: NlosMaskTables | None = None,
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

    if (
        nlos_tables is not None
        and epoch_idx is not None
        and (nlos_tables.weak or nlos_tables.strong)
    ):
        prns = [_measurement_prn(m) for m in rows]
        weights = np.asarray(
            apply_mask_to_weights(
                int(epoch_idx),
                prns,
                weights,
                nlos_tables,
                k_weak=observations.robust.nlos_k_weak,
                k_strong=observations.robust.nlos_k_strong,
            ),
            dtype=np.float64,
        )

    return EpochObservationInputs(
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        carrier_anchor_rows=carrier_anchor_rows,
    )
