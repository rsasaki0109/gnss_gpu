"""Quality gating helpers for multi-GNSS single-epoch solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.multi_gnss import SYSTEM_GPS


@dataclass(frozen=True)
class MultiGNSSQualityVetoConfig:
    """Thresholds for accepting a multi-GNSS WLS epoch."""

    residual_p95_max_m: float = 100.0
    residual_max_abs_m: float = 250.0
    bias_delta_max_m: float = 100.0
    extra_satellite_min: int = 2
    reference_system: int = SYSTEM_GPS


@dataclass(frozen=True)
class MultiGNSSQualityMetrics:
    """Per-epoch diagnostics used by the quality veto."""

    reference_satellite_count: int
    multi_satellite_count: int
    extra_satellite_count: int
    reference_residual_p95_abs_m: float
    reference_residual_max_abs_m: float
    multi_residual_p95_abs_m: float
    multi_residual_max_abs_m: float
    multi_bias_range_m: float


@dataclass(frozen=True)
class MultiGNSSQualityDecision:
    """Chosen epoch solution after applying the quality veto."""

    position: np.ndarray
    clock_bias_m: float
    use_multi: bool
    metrics: MultiGNSSQualityMetrics


def _residual_stats(
    position: np.ndarray,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    system_ids: np.ndarray,
    bias_by_system: dict[int, float],
) -> tuple[float, float]:
    if len(pseudoranges) == 0:
        return float("inf"), float("inf")

    ranges = np.linalg.norm(sat_ecef - position.reshape(1, 3), axis=1)
    pred = np.array(
        [
            ranges[i] + float(bias_by_system.get(int(system_ids[i]), 0.0))
            for i in range(len(ranges))
        ],
        dtype=np.float64,
    )
    abs_residual = np.abs(np.asarray(pseudoranges, dtype=np.float64) - pred)
    return float(np.percentile(abs_residual, 95)), float(np.max(abs_residual))


def compute_multi_gnss_quality_metrics(
    reference_solution: np.ndarray,
    multi_position: np.ndarray,
    multi_biases: dict[int, float],
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    system_ids: np.ndarray,
    reference_system: int = SYSTEM_GPS,
) -> MultiGNSSQualityMetrics:
    """Compute simple epoch-level diagnostics for a multi-GNSS candidate."""

    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    pseudoranges = np.asarray(pseudoranges, dtype=np.float64)
    system_ids = np.asarray(system_ids, dtype=np.int32)
    reference_solution = np.asarray(reference_solution, dtype=np.float64)
    multi_position = np.asarray(multi_position, dtype=np.float64)

    reference_mask = system_ids == int(reference_system)
    reference_sat_count = int(np.count_nonzero(reference_mask))
    multi_sat_count = int(len(system_ids))
    extra_sat_count = multi_sat_count - reference_sat_count

    reference_bias_m = float(reference_solution[3]) if reference_solution.shape[0] > 3 else 0.0
    reference_residual_p95, reference_residual_max = _residual_stats(
        np.asarray(reference_solution[:3], dtype=np.float64),
        sat_ecef[reference_mask],
        pseudoranges[reference_mask],
        np.full(reference_sat_count, int(reference_system), dtype=np.int32),
        {int(reference_system): reference_bias_m},
    )
    multi_residual_p95, multi_residual_max = _residual_stats(
        multi_position,
        sat_ecef,
        pseudoranges,
        system_ids,
        multi_biases,
    )

    ref_bias_for_range = float(multi_biases.get(int(reference_system), reference_bias_m))
    used_systems = {int(system_id) for system_id in system_ids if int(system_id) != int(reference_system)}
    if used_systems:
        multi_bias_range_m = float(
            max(
                abs(float(multi_biases.get(system_id, ref_bias_for_range)) - ref_bias_for_range)
                for system_id in used_systems
            )
        )
    else:
        multi_bias_range_m = 0.0

    return MultiGNSSQualityMetrics(
        reference_satellite_count=reference_sat_count,
        multi_satellite_count=multi_sat_count,
        extra_satellite_count=extra_sat_count,
        reference_residual_p95_abs_m=reference_residual_p95,
        reference_residual_max_abs_m=reference_residual_max,
        multi_residual_p95_abs_m=multi_residual_p95,
        multi_residual_max_abs_m=multi_residual_max,
        multi_bias_range_m=multi_bias_range_m,
    )


def accept_multi_gnss_solution(
    metrics: MultiGNSSQualityMetrics,
    config: MultiGNSSQualityVetoConfig,
) -> bool:
    """Return whether the multi-GNSS solution clears the fixed veto."""

    return (
        metrics.extra_satellite_count >= int(config.extra_satellite_min)
        and metrics.multi_residual_p95_abs_m <= float(config.residual_p95_max_m)
        and metrics.multi_residual_max_abs_m <= float(config.residual_max_abs_m)
        and metrics.multi_bias_range_m <= float(config.bias_delta_max_m)
    )


def select_multi_gnss_solution(
    reference_solution: np.ndarray,
    multi_position: np.ndarray,
    multi_biases: dict[int, float],
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    system_ids: np.ndarray,
    config: MultiGNSSQualityVetoConfig | None = None,
) -> MultiGNSSQualityDecision:
    """Choose between a reference-system-only and multi-GNSS epoch solution."""

    config = config or MultiGNSSQualityVetoConfig()
    metrics = compute_multi_gnss_quality_metrics(
        reference_solution=reference_solution,
        multi_position=multi_position,
        multi_biases=multi_biases,
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        system_ids=system_ids,
        reference_system=config.reference_system,
    )
    use_multi = accept_multi_gnss_solution(metrics, config)
    reference_solution = np.asarray(reference_solution, dtype=np.float64)
    if use_multi:
        clock_bias_m = float(
            multi_biases.get(int(config.reference_system), reference_solution[3])
        )
        position = np.asarray(multi_position, dtype=np.float64)
    else:
        clock_bias_m = float(reference_solution[3])
        position = np.asarray(reference_solution[:3], dtype=np.float64)
    return MultiGNSSQualityDecision(
        position=position,
        clock_bias_m=clock_bias_m,
        use_multi=use_multi,
        metrics=metrics,
    )
