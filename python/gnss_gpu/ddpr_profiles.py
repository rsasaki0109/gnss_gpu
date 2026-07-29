"""Truth-free DDPR offset profiles, arc screening, and evidence weighting."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

from gnss_gpu.evidence import BasinScore


class OffsetMode(str, Enum):
    CONSTANT = "constant"
    AFFINE = "affine"
    PIECEWISE_LINEAR = "piecewise_linear"


@dataclass(frozen=True)
class OffsetProfile:
    mode: OffsetMode
    knot_epochs: tuple[float, ...]
    knot_offsets_ecef_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        knots = np.asarray(self.knot_epochs, dtype=np.float64)
        offsets = np.asarray(self.knot_offsets_ecef_m, dtype=np.float64)
        if (
            knots.ndim != 1
            or not knots.size
            or offsets.shape != (knots.size, 3)
            or not np.all(np.isfinite(knots))
            or not np.all(np.isfinite(offsets))
            or np.any(np.diff(knots) <= 0)
        ):
            raise ValueError("offset profile knots and offsets are invalid")
        expected = 1 if self.mode == OffsetMode.CONSTANT else 2
        if self.mode != OffsetMode.PIECEWISE_LINEAR and knots.size != expected:
            raise ValueError(f"{self.mode.value} profile has the wrong knot count")
        if self.mode == OffsetMode.PIECEWISE_LINEAR and knots.size < 3:
            raise ValueError("piecewise-linear profile requires at least three knots")

    def evaluate(self, epochs: Sequence[float] | np.ndarray) -> np.ndarray:
        query = np.asarray(epochs, dtype=np.float64)
        knots = np.asarray(self.knot_epochs, dtype=np.float64)
        offsets = np.asarray(self.knot_offsets_ecef_m, dtype=np.float64)
        if self.mode == OffsetMode.CONSTANT:
            return np.repeat(offsets[:1], query.size, axis=0)
        return np.column_stack(
            [
                np.interp(query, knots, offsets[:, axis], left=offsets[0, axis], right=offsets[-1, axis])
                for axis in range(3)
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "knot_epochs": list(self.knot_epochs),
            "knot_offsets_ecef_m": [list(offset) for offset in self.knot_offsets_ecef_m],
        }


@dataclass(frozen=True)
class OffsetFit:
    accepted: bool
    reason: str
    profile: OffsetProfile | None
    weighted_rms_m: float | None
    max_residual_m: float | None
    effective_observations: int
    condition_number: float | None
    parameter_count: int
    bic: float | None


@dataclass(frozen=True)
class OffsetModelSelection:
    accepted: bool
    reason: str
    selected: OffsetFit | None
    fits: Mapping[str, OffsetFit]
    improvement_fraction: float | None


def _validate_fit_inputs(
    epochs: Sequence[float] | np.ndarray,
    offsets_ecef_m: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(offsets_ecef_m, dtype=np.float64)
    w = np.ones(x.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or y.shape != (x.size, 3) or w.shape != (x.size,):
        raise ValueError("offset fit inputs have incompatible shapes")
    if x.size == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("offset fit inputs must be non-empty and finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("offset fit weights must be finite and non-negative")
    if np.any(np.diff(x) <= 0):
        raise ValueError("offset fit epochs must be strictly increasing")
    return x, y, w


def _piecewise_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    design = np.zeros((x.size, knots.size), dtype=np.float64)
    for index, value in enumerate(x):
        if value <= knots[0]:
            design[index, 0] = 1.0
        elif value >= knots[-1]:
            design[index, -1] = 1.0
        else:
            right = int(np.searchsorted(knots, value, side="right"))
            left = right - 1
            fraction = (value - knots[left]) / (knots[right] - knots[left])
            design[index, left] = 1.0 - fraction
            design[index, right] = fraction
    return design


def fit_offset_profile(
    epochs: Sequence[float] | np.ndarray,
    offsets_ecef_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    mode: OffsetMode | str = OffsetMode.CONSTANT,
    knot_epochs: Sequence[float] | None = None,
    maximum_condition_number: float = 1.0e8,
) -> OffsetFit:
    """Fit a constant, affine, or continuous piecewise-linear ECEF profile."""

    x, y, w = _validate_fit_inputs(epochs, offsets_ecef_m, weights)
    mode = OffsetMode(mode)
    if mode == OffsetMode.CONSTANT:
        knots = np.asarray([x[0]], dtype=np.float64)
        design = np.ones((x.size, 1), dtype=np.float64)
    elif mode == OffsetMode.AFFINE:
        knots = np.asarray([x[0], x[-1]], dtype=np.float64)
        design = _piecewise_design(x, knots)
    else:
        if knot_epochs is None:
            raise ValueError("piecewise-linear fit requires knot_epochs")
        knots = np.asarray(knot_epochs, dtype=np.float64)
        if (
            knots.ndim != 1
            or knots.size < 3
            or not np.all(np.isfinite(knots))
            or np.any(np.diff(knots) <= 0)
            or knots[0] != x[0]
            or knots[-1] != x[-1]
        ):
            raise ValueError("piecewise knots must be finite, increasing, and span all epochs")
        design = _piecewise_design(x, knots)

    effective = int(np.count_nonzero(w > 0))
    parameter_count = int(design.shape[1] * 3)
    if effective <= design.shape[1]:
        return OffsetFit(
            accepted=False,
            reason="insufficient_evidence",
            profile=None,
            weighted_rms_m=None,
            max_residual_m=None,
            effective_observations=effective,
            condition_number=None,
            parameter_count=parameter_count,
            bic=None,
        )

    sqrt_w = np.sqrt(w)[:, None]
    weighted_design = design * sqrt_w
    condition = float(np.linalg.cond(weighted_design))
    if not math.isfinite(condition) or condition > maximum_condition_number:
        return OffsetFit(
            accepted=False,
            reason="ill_conditioned_profile",
            profile=None,
            weighted_rms_m=None,
            max_residual_m=None,
            effective_observations=effective,
            condition_number=condition,
            parameter_count=parameter_count,
            bic=None,
        )

    coefficients, *_ = np.linalg.lstsq(weighted_design, y * sqrt_w, rcond=None)
    predicted = design @ coefficients
    residual_norm = np.linalg.norm(y - predicted, axis=1)
    positive = w > 0
    weighted_rss = float(np.sum(w[positive] * np.square(residual_norm[positive])))
    total_weight = float(np.sum(w[positive]))
    rms = math.sqrt(weighted_rss / total_weight)
    bic_n = effective * 3
    variance = max(weighted_rss / max(bic_n, 1), np.finfo(np.float64).tiny)
    bic = float(bic_n * math.log(variance) + parameter_count * math.log(bic_n))
    profile = OffsetProfile(
        mode=mode,
        knot_epochs=tuple(float(value) for value in knots),
        knot_offsets_ecef_m=tuple(
            tuple(float(value) for value in row)
            for row in coefficients
        ),
    )
    return OffsetFit(
        accepted=True,
        reason="fit_available",
        profile=profile,
        weighted_rms_m=rms,
        max_residual_m=float(np.max(residual_norm[positive])),
        effective_observations=effective,
        condition_number=condition,
        parameter_count=parameter_count,
        bic=bic,
    )


def select_offset_profile(
    epochs: Sequence[float] | np.ndarray,
    offsets_ecef_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    piecewise_knots: Sequence[float] | None = None,
    minimum_improvement_fraction: float = 0.15,
) -> OffsetModelSelection:
    """Choose complexity only when BIC and residual improvement both support it."""

    fits = {
        OffsetMode.CONSTANT.value: fit_offset_profile(
            epochs, offsets_ecef_m, weights=weights, mode=OffsetMode.CONSTANT
        ),
        OffsetMode.AFFINE.value: fit_offset_profile(
            epochs, offsets_ecef_m, weights=weights, mode=OffsetMode.AFFINE
        ),
    }
    if piecewise_knots is not None:
        fits[OffsetMode.PIECEWISE_LINEAR.value] = fit_offset_profile(
            epochs,
            offsets_ecef_m,
            weights=weights,
            mode=OffsetMode.PIECEWISE_LINEAR,
            knot_epochs=piecewise_knots,
        )
    available = [fit for fit in fits.values() if fit.accepted]
    if not available:
        return OffsetModelSelection(False, "no_offset_model_fit", None, fits, None)

    constant = fits[OffsetMode.CONSTANT.value]
    if not constant.accepted or constant.weighted_rms_m is None:
        return OffsetModelSelection(False, "constant_reference_unavailable", None, fits, None)
    selected = min(
        available,
        key=lambda fit: (
            float("inf") if fit.bic is None else fit.bic,
            fit.parameter_count,
        ),
    )
    improvement = (
        (constant.weighted_rms_m - selected.weighted_rms_m) / constant.weighted_rms_m
        if selected.weighted_rms_m is not None and constant.weighted_rms_m > 0
        else 0.0
    )
    if selected.profile is not None and selected.profile.mode != OffsetMode.CONSTANT:
        if improvement < minimum_improvement_fraction:
            selected = constant
            improvement = 0.0
    return OffsetModelSelection(
        accepted=True,
        reason=f"selected_{selected.profile.mode.value}" if selected.profile else "no_profile",
        selected=selected,
        fits=fits,
        improvement_fraction=improvement,
    )


@dataclass(frozen=True)
class ArcScreenPolicy:
    edge_m: float = 5.0
    maximum_epoch_gap: int = 2
    minimum_hard_exclusion_epochs: int = 4
    hard_outlier_fraction: float = 0.5
    residual_soft_scale_m: float = 5.0
    neutral_sparse_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.edge_m <= 0 or self.residual_soft_scale_m <= 0:
            raise ValueError("arc screen scales must be positive")
        if self.maximum_epoch_gap < 0 or self.minimum_hard_exclusion_epochs < 1:
            raise ValueError("arc screen epoch limits are invalid")
        if not 0 <= self.hard_outlier_fraction <= 1:
            raise ValueError("hard_outlier_fraction must be in [0, 1]")
        if not 0 <= self.neutral_sparse_weight <= 1:
            raise ValueError("neutral_sparse_weight must be in [0, 1]")


@dataclass(frozen=True)
class ArcObservation:
    epoch: int
    sat_id: str
    residual_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, int):
            raise TypeError("epoch must be an integer")
        if not self.sat_id or not math.isfinite(self.residual_m):
            raise ValueError("arc observation must have a satellite id and finite residual")


@dataclass(frozen=True)
class ArcQuality:
    arc_id: str
    sat_id: str
    start_epoch: int
    end_epoch: int
    epochs_present: int
    outlier_fraction: float
    ambiguous_fraction: float
    median_abs_residual_m: float
    quality_weight: float
    hard_excluded: bool


def score_arc_quality(
    *,
    sat_id: str,
    start_epoch: int,
    end_epoch: int,
    epochs_present: int,
    outlier_fraction: float,
    median_abs_residual_m: float,
    ambiguous_fraction: float = 0.0,
    policy: ArcScreenPolicy | None = None,
    arc_index: int = 0,
) -> ArcQuality:
    policy = policy or ArcScreenPolicy()
    if epochs_present < 1:
        raise ValueError("arc must contain at least one epoch")
    if not 0 <= outlier_fraction <= 1 or not 0 <= ambiguous_fraction <= 1:
        raise ValueError("arc fractions must be in [0, 1]")
    confidence = min(1.0, epochs_present / policy.minimum_hard_exclusion_epochs)
    clean_support = math.exp(-3.0 * outlier_fraction) * math.exp(
        -median_abs_residual_m / policy.residual_soft_scale_m
    )
    quality = (
        (1.0 - confidence) * policy.neutral_sparse_weight
        + confidence * clean_support
    ) * (1.0 - 0.5 * ambiguous_fraction)
    hard_excluded = (
        epochs_present >= policy.minimum_hard_exclusion_epochs
        and outlier_fraction >= policy.hard_outlier_fraction
        and median_abs_residual_m >= policy.edge_m
    )
    return ArcQuality(
        arc_id=f"{sat_id}:{arc_index}",
        sat_id=sat_id,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        epochs_present=epochs_present,
        outlier_fraction=outlier_fraction,
        ambiguous_fraction=ambiguous_fraction,
        median_abs_residual_m=median_abs_residual_m,
        quality_weight=0.0 if hard_excluded else min(1.0, max(0.0, quality)),
        hard_excluded=hard_excluded,
    )


def _components(residuals: Mapping[str, float], edge_m: float) -> list[set[str]]:
    adjacency = {sat: set() for sat in residuals}
    satellites = sorted(residuals)
    for index, left in enumerate(satellites):
        for right in satellites[index + 1 :]:
            if abs(residuals[left] - residuals[right]) < edge_m:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[set[str]] = []
    unseen = set(satellites)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = {root}
        unseen.remove(root)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def screen_satellite_arcs(
    observations: Iterable[ArcObservation],
    policy: ArcScreenPolicy | None = None,
) -> tuple[ArcQuality, ...]:
    """Screen satellite arcs without arbitrary largest-cluster tie breaking."""

    policy = policy or ArcScreenPolicy()
    rows = sorted(observations, key=lambda row: (row.epoch, row.sat_id))
    by_group: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (row.epoch, row.sat_id)
        if key in seen:
            raise ValueError(f"duplicate arc observation: epoch={row.epoch}, sat={row.sat_id}")
        seen.add(key)
        by_group[(row.epoch, row.sat_id[0])][row.sat_id] = row.residual_m

    status: dict[tuple[int, str], tuple[float, float]] = {}
    for (epoch, _system), residuals in by_group.items():
        components = _components(residuals, policy.edge_m)
        largest_size = max((len(component) for component in components), default=0)
        largest = [component for component in components if len(component) == largest_size]
        ambiguous = len(largest) != 1
        inliers = largest[0] if not ambiguous else set()
        for sat_id in residuals:
            status[(epoch, sat_id)] = (
                0.0 if ambiguous or sat_id in inliers else 1.0,
                1.0 if ambiguous else 0.0,
            )

    by_sat: dict[str, list[ArcObservation]] = defaultdict(list)
    for row in rows:
        by_sat[row.sat_id].append(row)
    qualities: list[ArcQuality] = []
    for sat_id, sat_rows in sorted(by_sat.items()):
        arcs: list[list[ArcObservation]] = []
        for row in sat_rows:
            if not arcs or row.epoch - arcs[-1][-1].epoch > policy.maximum_epoch_gap:
                arcs.append([row])
            else:
                arcs[-1].append(row)
        for arc_index, arc in enumerate(arcs):
            outlier_fraction = statistics.fmean(status[(row.epoch, sat_id)][0] for row in arc)
            ambiguous_fraction = statistics.fmean(status[(row.epoch, sat_id)][1] for row in arc)
            qualities.append(
                score_arc_quality(
                    sat_id=sat_id,
                    start_epoch=arc[0].epoch,
                    end_epoch=arc[-1].epoch,
                    epochs_present=len(arc),
                    outlier_fraction=outlier_fraction,
                    ambiguous_fraction=ambiguous_fraction,
                    median_abs_residual_m=statistics.median(
                        abs(row.residual_m) for row in arc
                    ),
                    policy=policy,
                    arc_index=arc_index,
                )
            )
    return tuple(qualities)


def evidence_aware_weight(
    base_weight: float,
    *,
    arc_quality_weight: float,
    basin_score: BasinScore,
    minimum_family_support: float = 1.0e-3,
) -> float:
    """Combine independent-family support geometrically with arc quality."""

    if not math.isfinite(base_weight) or base_weight < 0:
        raise ValueError("base weight must be finite and non-negative")
    if not 0 <= arc_quality_weight <= 1:
        raise ValueError("arc quality weight must be in [0, 1]")
    if not 0 < minimum_family_support <= 1:
        raise ValueError("minimum_family_support must be in (0, 1]")
    supports = [
        max(minimum_family_support, min(1.0, float(value)))
        for value in basin_score.family_support.values()
    ]
    if not supports:
        return 0.0
    geometric_support = math.exp(statistics.fmean(math.log(value) for value in supports))
    coverage = min(1.0, basin_score.family_count / 3.0)
    temporal = max(0.0, min(1.0, basin_score.temporal_stability))
    return base_weight * arc_quality_weight * geometric_support * coverage * temporal
