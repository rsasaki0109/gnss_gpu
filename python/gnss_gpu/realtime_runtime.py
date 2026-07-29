"""Realtime budgets, adaptive particle scheduling, and persistent runtime pooling."""

from __future__ import annotations

import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import numpy as np


class RuntimeMode(str, Enum):
    NORMAL = "normal"
    SEARCH = "search"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class RealtimeBudget:
    normal_epoch_ms: float = 100.0
    search_epoch_ms: float = 1000.0
    degraded_epoch_ms: float = 100.0
    peak_gpu_memory_mb: float = 4096.0

    def __post_init__(self) -> None:
        if min(
            self.normal_epoch_ms,
            self.search_epoch_ms,
            self.degraded_epoch_ms,
            self.peak_gpu_memory_mb,
        ) <= 0:
            raise ValueError("realtime budgets must be positive")

    def deadline_ms(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.NORMAL: self.normal_epoch_ms,
            RuntimeMode.SEARCH: self.search_epoch_ms,
            RuntimeMode.DEGRADED: self.degraded_epoch_ms,
        }[mode]


@dataclass(frozen=True)
class EpochRuntimeSample:
    epoch: int
    mode: RuntimeMode
    elapsed_ms: float
    stage_ms: dict[str, float]
    particle_count: int
    gpu_memory_mb: float
    deadline_met: bool
    fallback_active: bool


@dataclass(frozen=True)
class RuntimeAssessment:
    passed: bool
    sample_count: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    normal_latency_max_ms: float | None
    search_latency_max_ms: float | None
    peak_gpu_memory_mb: float | None
    deadline_misses: int
    reasons: tuple[str, ...]


class _EpochTimer(AbstractContextManager["_EpochTimer"]):
    def __init__(
        self,
        monitor: "RealtimeMonitor",
        *,
        epoch: int,
        mode: RuntimeMode,
        particle_count: int,
        gpu_memory_mb: float,
        fallback_active: bool,
        synchronize: Callable[[], None] | None,
    ) -> None:
        self.monitor = monitor
        self.epoch = int(epoch)
        self.mode = RuntimeMode(mode)
        self.particle_count = int(particle_count)
        self.gpu_memory_mb = float(gpu_memory_mb)
        self.fallback_active = bool(fallback_active)
        self.synchronize = synchronize
        self.stage_ms: dict[str, float] = {}
        self._start = 0.0

    def __enter__(self) -> "_EpochTimer":
        self._start = self.monitor.clock()
        return self

    def measure(self, stage: str) -> AbstractContextManager[None]:
        if not stage:
            raise ValueError("runtime stage must not be empty")
        timer = self

        class _Stage(AbstractContextManager[None]):
            def __enter__(self) -> None:
                self.start = timer.monitor.clock()
                return None

            def __exit__(self, exc_type, exc, traceback) -> bool:
                elapsed = (timer.monitor.clock() - self.start) * 1000.0
                timer.stage_ms[stage] = timer.stage_ms.get(stage, 0.0) + elapsed
                return False

        return _Stage()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.synchronize is not None:
            self.synchronize()
        elapsed_ms = (self.monitor.clock() - self._start) * 1000.0
        if exc_type is None:
            deadline = self.monitor.budget.deadline_ms(self.mode)
            self.monitor.samples.append(
                EpochRuntimeSample(
                    epoch=self.epoch,
                    mode=self.mode,
                    elapsed_ms=elapsed_ms,
                    stage_ms=dict(self.stage_ms),
                    particle_count=self.particle_count,
                    gpu_memory_mb=self.gpu_memory_mb,
                    deadline_met=elapsed_ms <= deadline,
                    fallback_active=self.fallback_active,
                )
            )
        return False


class RealtimeMonitor:
    def __init__(
        self,
        budget: RealtimeBudget | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.budget = budget or RealtimeBudget()
        self.clock = clock
        self.samples: list[EpochRuntimeSample] = []

    def epoch(
        self,
        *,
        epoch: int,
        mode: RuntimeMode,
        particle_count: int,
        gpu_memory_mb: float,
        fallback_active: bool = False,
        synchronize: Callable[[], None] | None = None,
    ) -> _EpochTimer:
        if particle_count < 1 or not math.isfinite(gpu_memory_mb) or gpu_memory_mb < 0:
            raise ValueError("runtime particle count and memory must be valid")
        return _EpochTimer(
            self,
            epoch=epoch,
            mode=mode,
            particle_count=particle_count,
            gpu_memory_mb=gpu_memory_mb,
            fallback_active=fallback_active,
            synchronize=synchronize,
        )

    def record(self, sample: EpochRuntimeSample) -> None:
        self.samples.append(sample)

    def assess(self) -> RuntimeAssessment:
        if not self.samples:
            return RuntimeAssessment(
                False, 0, None, None, None, None, None, 0, ("runtime_evidence_missing",)
            )
        latencies = np.asarray([sample.elapsed_ms for sample in self.samples])
        normal = [
            sample.elapsed_ms
            for sample in self.samples
            if sample.mode in {RuntimeMode.NORMAL, RuntimeMode.DEGRADED}
        ]
        search = [
            sample.elapsed_ms for sample in self.samples if sample.mode == RuntimeMode.SEARCH
        ]
        peak_memory = max(sample.gpu_memory_mb for sample in self.samples)
        misses = sum(not sample.deadline_met for sample in self.samples)
        reasons: list[str] = []
        normal_max = max(normal) if normal else None
        search_max = max(search) if search else None
        if normal_max is None:
            reasons.append("normal_runtime_evidence_missing")
        elif normal_max > self.budget.normal_epoch_ms:
            reasons.append("normal_deadline_exceeded")
        if search_max is not None and search_max > self.budget.search_epoch_ms:
            reasons.append("search_deadline_exceeded")
        if peak_memory > self.budget.peak_gpu_memory_mb:
            reasons.append("gpu_memory_budget_exceeded")
        if misses:
            reasons.append("epoch_deadline_miss")
        return RuntimeAssessment(
            passed=not reasons,
            sample_count=len(self.samples),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            normal_latency_max_ms=normal_max,
            search_latency_max_ms=search_max,
            peak_gpu_memory_mb=peak_memory,
            deadline_misses=misses,
            reasons=tuple(dict.fromkeys(reasons)),
        )


@dataclass(frozen=True)
class AdaptiveParticlePolicy:
    levels: tuple[int, ...] = (50_000, 100_000, 500_000)
    latency_pressure_ratio: float = 0.85
    memory_pressure_ratio: float = 0.90
    search_ess_ratio: float = 0.25
    stable_ess_ratio: float = 0.70
    stable_epochs_before_downshift: int = 5

    def __post_init__(self) -> None:
        if (
            not self.levels
            or any(level < 1 for level in self.levels)
            or tuple(sorted(set(self.levels))) != self.levels
        ):
            raise ValueError("particle levels must be unique, increasing, and positive")
        if not all(
            0 < value <= 1
            for value in (
                self.latency_pressure_ratio,
                self.memory_pressure_ratio,
                self.search_ess_ratio,
                self.stable_ess_ratio,
            )
        ):
            raise ValueError("adaptive particle ratios must be in (0, 1]")
        if self.stable_epochs_before_downshift < 1:
            raise ValueError("stable epoch threshold must be positive")


@dataclass(frozen=True)
class ParticleScheduleDecision:
    target_particles: int
    mode: RuntimeMode
    reason: str
    fallback_active: bool


class AdaptiveParticleController:
    def __init__(
        self,
        policy: AdaptiveParticlePolicy | None = None,
        budget: RealtimeBudget | None = None,
    ) -> None:
        self.policy = policy or AdaptiveParticlePolicy()
        self.budget = budget or RealtimeBudget()
        self.current_particles = self.policy.levels[1 if len(self.policy.levels) > 1 else 0]
        self._stable_epochs = 0

    def decide(
        self,
        *,
        elapsed_ms: float,
        gpu_memory_mb: float,
        ess_ratio: float,
        hypothesis_count: int,
        outage_active: bool,
    ) -> ParticleScheduleDecision:
        if (
            not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
            or not math.isfinite(gpu_memory_mb)
            or gpu_memory_mb < 0
            or not 0 <= ess_ratio <= 1
            or hypothesis_count < 1
        ):
            raise ValueError("adaptive particle inputs are invalid")
        levels = self.policy.levels
        index = levels.index(self.current_particles)
        latency_pressure = elapsed_ms >= self.budget.normal_epoch_ms * self.policy.latency_pressure_ratio
        memory_pressure = (
            gpu_memory_mb >= self.budget.peak_gpu_memory_mb * self.policy.memory_pressure_ratio
        )
        if latency_pressure or memory_pressure:
            self._stable_epochs = 0
            if index > 0:
                self.current_particles = levels[index - 1]
                return ParticleScheduleDecision(
                    self.current_particles,
                    RuntimeMode.DEGRADED,
                    "resource_pressure_downshift",
                    True,
                )
            return ParticleScheduleDecision(
                self.current_particles,
                RuntimeMode.DEGRADED,
                "minimum_particles_safe_fallback",
                True,
            )

        needs_search = outage_active or hypothesis_count > 1 or ess_ratio <= self.policy.search_ess_ratio
        if needs_search:
            self._stable_epochs = 0
            if index < len(levels) - 1:
                self.current_particles = levels[index + 1]
            return ParticleScheduleDecision(
                self.current_particles,
                RuntimeMode.SEARCH,
                "uncertainty_search",
                False,
            )

        if ess_ratio >= self.policy.stable_ess_ratio and hypothesis_count == 1:
            self._stable_epochs += 1
        else:
            self._stable_epochs = 0
        if self._stable_epochs >= self.policy.stable_epochs_before_downshift and index > 0:
            self.current_particles = levels[index - 1]
            self._stable_epochs = 0
            return ParticleScheduleDecision(
                self.current_particles,
                RuntimeMode.NORMAL,
                "stable_posterior_downshift",
                False,
            )
        return ParticleScheduleDecision(
            self.current_particles,
            RuntimeMode.NORMAL,
            "hold_particle_level",
            False,
        )


def estimate_pf_device_memory_mb(n_particles: int, *, bytes_per_particle: int = 320) -> float:
    """Conservative capacity estimate covering state, temp, weights, and reductions."""

    if n_particles < 1 or bytes_per_particle < 1:
        raise ValueError("memory estimate inputs must be positive")
    return n_particles * bytes_per_particle / (1024.0 * 1024.0)


class _ParticleRuntime(Protocol):
    n_particles: int
    _initialized: bool

    def initialize(self, position_ecef, **kwargs) -> None: ...
    def estimate(self): ...
    def get_particle_states(self): ...
    def get_log_weights(self): ...
    def set_particle_states(self, states) -> None: ...
    def set_log_weights(self, log_weights) -> None: ...


class PersistentParticleRuntimePool:
    """Lazily allocate capacity levels once and migrate by systematic resampling."""

    def __init__(
        self,
        factory: Callable[[int], _ParticleRuntime],
        capacities: tuple[int, ...],
        *,
        maximum_resident_memory_mb: float = 4096.0,
    ) -> None:
        if not capacities or tuple(sorted(set(capacities))) != capacities:
            raise ValueError("pool capacities must be unique and increasing")
        estimated = sum(estimate_pf_device_memory_mb(value) for value in capacities)
        if estimated > maximum_resident_memory_mb:
            raise MemoryError(
                f"particle pool estimate {estimated:.1f} MB exceeds "
                f"{maximum_resident_memory_mb:.1f} MB"
            )
        self.factory = factory
        self.capacities = capacities
        self.runtimes: dict[int, _ParticleRuntime] = {}
        self.active: _ParticleRuntime | None = None

    def activate(self, capacity: int) -> _ParticleRuntime:
        if capacity not in self.capacities:
            raise ValueError(f"unsupported particle capacity: {capacity}")
        target = self.runtimes.get(capacity)
        if target is None:
            target = self.factory(capacity)
            self.runtimes[capacity] = target
        if self.active is None:
            self.active = target
            return target
        if target is self.active:
            return target
        self._migrate(self.active, target)
        self.active = target
        return target

    @staticmethod
    def _migrate(source: _ParticleRuntime, target: _ParticleRuntime) -> None:
        if not source._initialized:
            raise RuntimeError("cannot migrate an uninitialized particle runtime")
        states = np.asarray(source.get_particle_states(), dtype=np.float64)
        log_weights = np.asarray(source.get_log_weights(), dtype=np.float64)
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            weights = np.full(log_weights.size, 1.0 / log_weights.size)
        else:
            floor = float(np.min(log_weights[finite])) - 100.0
            shifted = np.where(finite, log_weights, floor)
            weights = np.exp(shifted - float(np.max(shifted)))
            weights /= np.sum(weights)
        cdf = np.cumsum(weights)
        count = int(target.n_particles)
        indices = np.searchsorted(
            cdf,
            (np.arange(count, dtype=np.float64) + 0.5) / count,
            side="left",
        )
        selected = states[np.minimum(indices, states.shape[0] - 1)]
        if not target._initialized:
            target.initialize(
                position_ecef=np.mean(selected[:, :3], axis=0),
                clock_bias=float(np.mean(selected[:, 3])),
                spread_pos=1.0,
                spread_cb=1.0,
            )
        target.set_particle_states(selected)
        target.set_log_weights(np.full(count, -math.log(count), dtype=np.float64))
