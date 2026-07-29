from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.realtime_runtime import (
    AdaptiveParticleController,
    AdaptiveParticlePolicy,
    EpochRuntimeSample,
    PersistentParticleRuntimePool,
    RealtimeBudget,
    RealtimeMonitor,
    RuntimeMode,
    estimate_pf_device_memory_mb,
)


def _sample(epoch: int, mode: RuntimeMode, elapsed: float, memory: float = 500.0):
    budget = RealtimeBudget()
    return EpochRuntimeSample(
        epoch=epoch,
        mode=mode,
        elapsed_ms=elapsed,
        stage_ms={"predict": elapsed * 0.4, "update": elapsed * 0.6},
        particle_count=100_000,
        gpu_memory_mb=memory,
        deadline_met=elapsed <= budget.deadline_ms(mode),
        fallback_active=False,
    )


def test_runtime_monitor_enforces_normal_search_and_memory_budgets() -> None:
    monitor = RealtimeMonitor()
    monitor.record(_sample(1, RuntimeMode.NORMAL, 80.0))
    monitor.record(_sample(2, RuntimeMode.SEARCH, 900.0))
    assessment = monitor.assess()
    assert assessment.passed is True
    assert assessment.normal_latency_max_ms == 80.0
    assert assessment.search_latency_max_ms == 900.0

    monitor.record(_sample(3, RuntimeMode.NORMAL, 101.0))
    failed = monitor.assess()
    assert failed.passed is False
    assert "normal_deadline_exceeded" in failed.reasons
    assert failed.deadline_misses == 1


def test_epoch_timer_includes_explicit_gpu_synchronization() -> None:
    ticks = iter([0.0, 0.01, 0.03, 0.05])
    synchronized = []
    monitor = RealtimeMonitor(clock=lambda: next(ticks))
    with monitor.epoch(
        epoch=1,
        mode=RuntimeMode.NORMAL,
        particle_count=50_000,
        gpu_memory_mb=100.0,
        synchronize=lambda: synchronized.append(True),
    ) as timer:
        with timer.measure("predict"):
            pass
    assert synchronized == [True]
    assert monitor.samples[0].stage_ms["predict"] == pytest.approx(20.0)
    assert monitor.samples[0].elapsed_ms == pytest.approx(50.0)


def test_adaptive_particle_policy_searches_then_degrades_under_pressure() -> None:
    controller = AdaptiveParticleController(
        AdaptiveParticlePolicy(levels=(10, 20, 40), stable_epochs_before_downshift=2)
    )
    search = controller.decide(
        elapsed_ms=20,
        gpu_memory_mb=100,
        ess_ratio=0.1,
        hypothesis_count=2,
        outage_active=True,
    )
    assert search.target_particles == 40
    assert search.mode == RuntimeMode.SEARCH

    degraded = controller.decide(
        elapsed_ms=95,
        gpu_memory_mb=100,
        ess_ratio=0.5,
        hypothesis_count=1,
        outage_active=False,
    )
    assert degraded.target_particles == 20
    assert degraded.mode == RuntimeMode.DEGRADED
    assert degraded.fallback_active is True


class _FakeRuntime:
    def __init__(self, n_particles: int) -> None:
        self.n_particles = n_particles
        self._initialized = False
        self.states = np.empty((n_particles, 16))
        self.log_weights = np.empty(n_particles)

    def initialize(self, position_ecef, **kwargs) -> None:
        del kwargs
        self.states[:] = 0.0
        self.states[:, :3] = np.asarray(position_ecef)
        self.log_weights[:] = -np.log(self.n_particles)
        self._initialized = True

    def estimate(self):
        return np.mean(self.states[:, :4], axis=0)

    def get_particle_states(self):
        return self.states.copy()

    def get_log_weights(self):
        return self.log_weights.copy()

    def set_particle_states(self, states) -> None:
        self.states = np.asarray(states).copy()

    def set_log_weights(self, log_weights) -> None:
        self.log_weights = np.asarray(log_weights).copy()


def test_persistent_pool_reuses_capacity_and_migrates_posterior() -> None:
    pool = PersistentParticleRuntimePool(_FakeRuntime, (4, 8))
    small = pool.activate(4)
    small.initialize([0.0, 0.0, 0.0])
    small.states[:, 0] = [0.0, 1.0, 10.0, 11.0]
    small.log_weights[:] = [-10.0, -10.0, 0.0, 0.0]
    large = pool.activate(8)
    assert large._initialized is True
    assert np.mean(large.states[:, 0]) > 9.0
    assert pool.activate(4) is small
    assert len(pool.runtimes) == 2


def test_pool_refuses_capacity_plan_above_memory_budget() -> None:
    with pytest.raises(MemoryError, match="exceeds"):
        PersistentParticleRuntimePool(
            _FakeRuntime,
            (1_000_000, 2_000_000),
            maximum_resident_memory_mb=100.0,
        )
    assert estimate_pf_device_memory_mb(1_000_000) > 300.0


def test_locked_phase4_benchmark_meets_runtime_contract() -> None:
    result_path = (
        Path(__file__).parents[1]
        / "internal_docs"
        / "phase4_realtime_benchmark_2026_07_29.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assessment = result["assessment"]
    assert result["passed"] is True
    assert assessment["passed"] is True
    assert assessment["normal_latency_max_ms"] <= 100.0
    assert assessment["search_latency_max_ms"] <= 1_000.0
    assert assessment["peak_gpu_memory_mb"] <= 4_096.0
    assert assessment["deadline_misses"] == 0
    assert all(
        entry["actual_sha256"] == entry["expected_sha256"]
        for entry in result["m4"].values()
    )
