"""Truth-free, estimator-independent evidence scoring for candidate basins."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class EvidenceFamily(str, Enum):
    TDCP = "tdcp"
    DOPPLER = "doppler"
    IMU = "imu"
    CARRIER_CONTINUITY = "carrier_continuity"
    SATELLITE_ARC = "satellite_arc"
    ROAD_HEIGHT = "road_height"
    LOS_NLOS = "los_nlos"


MOTION_FAMILIES = frozenset(
    {EvidenceFamily.TDCP, EvidenceFamily.DOPPLER, EvidenceFamily.IMU}
)
CARRIER_FAMILIES = frozenset(
    {EvidenceFamily.CARRIER_CONTINUITY, EvidenceFamily.SATELLITE_ARC}
)
CONTEXT_FAMILIES = frozenset({EvidenceFamily.ROAD_HEIGHT, EvidenceFamily.LOS_NLOS})

_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "truth",
        "ground_truth",
        "reference_trajectory",
        "audit_error",
        "error_m",
        "sub50cm",
        "gained_epochs",
        "lost_epochs",
    }
)


def _forbidden_metadata_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name.lower() in _FORBIDDEN_METADATA_KEYS:
                paths.append(path)
            paths.extend(_forbidden_metadata_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(_forbidden_metadata_paths(nested, f"{prefix}[{index}]"))
    return paths


@dataclass(frozen=True)
class EvidenceSample:
    """One truth-free residual supporting or contradicting a candidate basin."""

    family: EvidenceFamily
    epoch: int
    residual: float
    scale: float
    reliability: float = 1.0
    sample_count: int = 1
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, int):
            raise TypeError("epoch must be an integer")
        if not math.isfinite(self.residual):
            raise ValueError("residual must be finite")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(self.reliability) or not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be in [0, 1]")
        if not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        forbidden = _forbidden_metadata_paths(self.metadata)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"truth/audit metadata is forbidden in production evidence: {names}")

    @property
    def support(self) -> float:
        """Gaussian-kernel support in [0, 1], attenuated by reliability."""

        normalized = abs(self.residual) / self.scale
        return self.reliability * math.exp(-0.5 * normalized * normalized)


@dataclass(frozen=True)
class BasinEvidence:
    basin_id: str
    samples: tuple[EvidenceSample, ...]
    generation: str = ""

    def __post_init__(self) -> None:
        if not self.basin_id:
            raise ValueError("basin_id must not be empty")
        if not isinstance(self.samples, tuple):
            object.__setattr__(self, "samples", tuple(self.samples))


class EvidenceBuilder:
    """Typed adapters from physical residuals to one candidate's evidence."""

    def __init__(self, basin_id: str, *, generation: str = "") -> None:
        if not basin_id:
            raise ValueError("basin_id must not be empty")
        self._basin_id = basin_id
        self._generation = generation
        self._samples: list[EvidenceSample] = []

    def _add(
        self,
        family: EvidenceFamily,
        epoch: int,
        residual: float,
        scale: float,
        *,
        reliability: float,
        sample_count: int,
        source: str,
        metadata: Mapping[str, Any] | None,
    ) -> "EvidenceBuilder":
        self._samples.append(
            EvidenceSample(
                family=family,
                epoch=epoch,
                residual=residual,
                scale=scale,
                reliability=reliability,
                sample_count=sample_count,
                source=source,
                metadata=metadata or {},
            )
        )
        return self

    def tdcp(
        self,
        epoch: int,
        residual_m: float,
        sigma_m: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(EvidenceFamily.TDCP, epoch, residual_m, sigma_m, **_adapter_kwargs(kwargs))

    def doppler(
        self,
        epoch: int,
        residual_mps: float,
        sigma_mps: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(
            EvidenceFamily.DOPPLER,
            epoch,
            residual_mps,
            sigma_mps,
            **_adapter_kwargs(kwargs),
        )

    def imu(
        self,
        epoch: int,
        motion_residual: float,
        sigma: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(EvidenceFamily.IMU, epoch, motion_residual, sigma, **_adapter_kwargs(kwargs))

    def carrier_continuity(
        self,
        epoch: int,
        phase_jump_cycles: float,
        tolerance_cycles: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(
            EvidenceFamily.CARRIER_CONTINUITY,
            epoch,
            phase_jump_cycles,
            tolerance_cycles,
            **_adapter_kwargs(kwargs),
        )

    def satellite_arc(
        self,
        epoch: int,
        gap_epochs: float,
        tolerated_gap_epochs: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(
            EvidenceFamily.SATELLITE_ARC,
            epoch,
            gap_epochs,
            tolerated_gap_epochs,
            **_adapter_kwargs(kwargs),
        )

    def road_height(
        self,
        epoch: int,
        residual_m: float,
        tolerance_m: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        return self._add(
            EvidenceFamily.ROAD_HEIGHT,
            epoch,
            residual_m,
            tolerance_m,
            **_adapter_kwargs(kwargs),
        )

    def los_nlos(
        self,
        epoch: int,
        mismatch_fraction: float,
        tolerance_fraction: float,
        **kwargs: Any,
    ) -> "EvidenceBuilder":
        if not 0 <= mismatch_fraction <= 1:
            raise ValueError("mismatch_fraction must be in [0, 1]")
        return self._add(
            EvidenceFamily.LOS_NLOS,
            epoch,
            mismatch_fraction,
            tolerance_fraction,
            **_adapter_kwargs(kwargs),
        )

    def build(self) -> BasinEvidence:
        return BasinEvidence(
            basin_id=self._basin_id,
            samples=tuple(self._samples),
            generation=self._generation,
        )


def _adapter_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"reliability", "sample_count", "source", "metadata"}
    unexpected = set(kwargs).difference(allowed)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"unexpected evidence adapter arguments: {names}")
    return {
        "reliability": kwargs.get("reliability", 1.0),
        "sample_count": kwargs.get("sample_count", 1),
        "source": kwargs.get("source", ""),
        "metadata": kwargs.get("metadata"),
    }


@dataclass(frozen=True)
class EvidencePolicy:
    minimum_families: int = 3
    minimum_groups: int = 2
    minimum_score: float = 0.55
    minimum_runner_margin: float = 0.08
    minimum_temporal_stability: float = 0.60
    minimum_epoch_span: int = 2
    unopposed_minimum_families: int = 5
    unopposed_minimum_score: float = 0.80
    unopposed_minimum_stability: float = 0.80


@dataclass(frozen=True)
class BasinScore:
    basin_id: str
    score: float
    family_count: int
    group_count: int
    epoch_span: int
    temporal_stability: float
    family_support: Mapping[str, float]
    family_sample_counts: Mapping[str, int]


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    selected_basin_id: str | None
    reason: str
    unsafe_reasons: tuple[str, ...]
    winner: BasinScore | None
    runner: BasinScore | None
    runner_margin: float | None


def _family_groups(families: set[EvidenceFamily]) -> int:
    return sum(
        bool(families.intersection(group))
        for group in (MOTION_FAMILIES, CARRIER_FAMILIES, CONTEXT_FAMILIES)
    )


def _temporal_stability(samples: Iterable[EvidenceSample]) -> float:
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        by_epoch[sample.epoch].append(sample.support)
    epoch_support = [
        statistics.fmean(by_epoch[epoch])
        for epoch in sorted(by_epoch)
    ]
    if len(epoch_support) < 2:
        return 0.0
    changes = [
        abs(current - previous)
        for previous, current in zip(epoch_support, epoch_support[1:])
    ]
    return max(0.0, 1.0 - statistics.fmean(changes))


def score_basin(evidence: BasinEvidence, policy: EvidencePolicy | None = None) -> BasinScore:
    """Score each family once so sample-rich channels cannot dominate."""

    policy = policy or EvidencePolicy()
    by_family: dict[EvidenceFamily, list[EvidenceSample]] = defaultdict(list)
    for sample in evidence.samples:
        by_family[sample.family].append(sample)

    family_support = {
        family: statistics.fmean(sample.support for sample in samples)
        for family, samples in by_family.items()
    }
    family_count = len(family_support)
    coverage = min(1.0, family_count / max(1, policy.minimum_families))
    balanced_support = statistics.fmean(family_support.values()) if family_support else 0.0
    samples = tuple(evidence.samples)
    epochs = [sample.epoch for sample in samples]
    epoch_span = max(epochs) - min(epochs) + 1 if epochs else 0
    stability = _temporal_stability(samples)
    score = balanced_support * coverage
    return BasinScore(
        basin_id=evidence.basin_id,
        score=score,
        family_count=family_count,
        group_count=_family_groups(set(by_family)),
        epoch_span=epoch_span,
        temporal_stability=stability,
        family_support={
            family.value: support
            for family, support in sorted(family_support.items(), key=lambda item: item[0].value)
        },
        family_sample_counts={
            family.value: sum(sample.sample_count for sample in family_samples)
            for family, family_samples in sorted(by_family.items(), key=lambda item: item[0].value)
        },
    )


class UnsafeAcceptanceDetector:
    """Apply fail-closed independent-evidence and competition gates."""

    def __init__(self, policy: EvidencePolicy | None = None) -> None:
        self.policy = policy or EvidencePolicy()

    def decide(self, basins: Iterable[BasinEvidence]) -> AcceptanceDecision:
        scores = sorted(
            (score_basin(basin, self.policy) for basin in basins),
            key=lambda item: (-item.score, item.basin_id),
        )
        if not scores:
            return AcceptanceDecision(
                accepted=False,
                selected_basin_id=None,
                reason="evidence_unavailable",
                unsafe_reasons=("evidence_unavailable",),
                winner=None,
                runner=None,
                runner_margin=None,
            )

        winner = scores[0]
        runner = scores[1] if len(scores) > 1 else None
        margin = winner.score - runner.score if runner is not None else None
        unsafe: list[str] = []
        if winner.family_count < self.policy.minimum_families:
            unsafe.append("insufficient_independent_families")
        if winner.group_count < self.policy.minimum_groups:
            unsafe.append("insufficient_independent_groups")
        if winner.epoch_span < self.policy.minimum_epoch_span:
            unsafe.append("insufficient_temporal_span")
        if winner.temporal_stability < self.policy.minimum_temporal_stability:
            unsafe.append("temporal_instability")
        if winner.score < self.policy.minimum_score:
            unsafe.append("weak_basin_support")
        if runner is not None and margin is not None and margin < self.policy.minimum_runner_margin:
            unsafe.append("ambiguous_basin_identity")
        if runner is None and not (
            winner.family_count >= self.policy.unopposed_minimum_families
            and winner.score >= self.policy.unopposed_minimum_score
            and winner.temporal_stability >= self.policy.unopposed_minimum_stability
        ):
            unsafe.append("unopposed_basin")

        accepted = not unsafe
        return AcceptanceDecision(
            accepted=accepted,
            selected_basin_id=winner.basin_id if accepted else None,
            reason="accepted" if accepted else unsafe[0],
            unsafe_reasons=tuple(unsafe),
            winner=winner,
            runner=runner,
            runner_margin=margin,
        )


@dataclass(frozen=True)
class TemporalState:
    observation_count: int
    winner_switches: int
    winner_continuity: float
    median_margin: float | None
    latest_winner: str | None


class TemporalEvidenceTracker:
    """Track selector continuity without retaining truth or audit outcomes."""

    def __init__(self, window_size: int = 20) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._history: deque[tuple[int, str | None, float | None]] = deque(maxlen=window_size)

    def observe(self, epoch: int, decision: AcceptanceDecision) -> TemporalState:
        if self._history and epoch <= self._history[-1][0]:
            raise ValueError("epochs must be strictly increasing")
        winner = decision.winner.basin_id if decision.winner is not None else None
        self._history.append((epoch, winner, decision.runner_margin))
        return self.state()

    def state(self) -> TemporalState:
        winners = [winner for _, winner, _ in self._history]
        switches = sum(
            previous != current
            for previous, current in zip(winners, winners[1:])
            if previous is not None and current is not None
        )
        comparable = sum(
            previous is not None and current is not None
            for previous, current in zip(winners, winners[1:])
        )
        continuity = 1.0 - switches / comparable if comparable else 0.0
        margins = [margin for _, _, margin in self._history if margin is not None]
        return TemporalState(
            observation_count=len(self._history),
            winner_switches=switches,
            winner_continuity=continuity,
            median_margin=statistics.median(margins) if margins else None,
            latest_winner=winners[-1] if winners else None,
        )
