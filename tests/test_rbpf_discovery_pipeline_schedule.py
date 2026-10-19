"""Acceptance must isolate scheduling gains and reject noisy controls."""

from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_discovery_pipeline import MODES, SEEDS, call_order, summaries  # noqa: E402


def rows(change=None, split=False):
    times = dict(original_cpu=1000, original_gpu=1000, blocked_gpu=800,
        candidate_gpu=600, cpu_AA=1000, gpu_AA=1000)
    result = []
    for r in range(5):
        for g in range(3):
            for mode in MODES:
                ns = change(r, g, mode, times[mode]) if change else times[mode]
                for v in ((1, ns - 1) if split else (ns,)):
                    result.append(dict(round=r, group=g, mode=mode, ns=v))
    return result


def test_required_gates_and_rotated_schedule():
    value = summaries(rows())
    assert value["GPU_same_backend_pass"] and value["GPU_joint_pass"]
    assert len(MODES) == 6 and len(SEEDS) == len(set(SEEDS)) == 16
    positions = Counter()
    for ci in range(2):
        for rd in range(5):
            for group in range(3):
                for seed in SEEDS:
                    for epoch in range(4):
                        order = call_order(ci, rd, group, seed, epoch)
                        assert len(order) == len(set(order)) == 6 and set(order) == set(MODES)
                        positions.update((mode, i) for i, mode in enumerate(order))
    assert set(positions) == {(mode, i) for mode in MODES for i in range(6)}


def test_old_gpu_gain_does_not_override_small_scheduling_gain():
    value = summaries(rows(lambda r, g, m, t: 610 if m == "blocked_gpu" else t))
    assert value["contrasts"]["gpu"]["numeric_screen"]
    assert not value["contrasts"]["overlap"]["numeric_screen"]
    assert not value["GPU_same_backend_pass"]


def test_signed_gpu_aa_cancellation_cannot_pass():
    value = summaries(rows(lambda r, g, m, t: (1200, 1000, 800)[g] if m == "gpu_AA" else t))
    assert value["contrasts"]["gpu_AA"]["median_paired_percent"] == 0
    assert value["contrasts"]["gpu_AA"]["absolute_median_percent"] == pytest.approx(20)
    assert not value["gpu_AA_valid"] and not value["GPU_same_backend_pass"]


def test_only_three_stable_control_rounds_rejected():
    value = summaries(rows(lambda r, g, m, t: 1200 if m == "cpu_AA" and r >= 3 else t))
    assert value["contrasts"]["cpu_AA"]["absolute_median_percent"] == 0
    assert value["contrasts"]["cpu_AA"]["absolute_rounds_under5"] == 3
    assert value["GPU_same_backend_pass"] and not value["GPU_joint_pass"]


def test_ratios_use_integer_group_sums():
    left, right = summaries(rows()), summaries(rows(split=True))
    assert left == right
    assert right["contrasts"]["gpu"]["median_paired_percent"] == pytest.approx(40)


def test_large_median_needs_four_improving_rounds():
    value = summaries(rows(lambda r, g, m, t: 1100 if m == "candidate_gpu" and r >= 3 else t))
    assert value["contrasts"]["gpu"]["median_paired_percent"] == pytest.approx(40)
    assert value["contrasts"]["gpu"]["improving_rounds"] == 3
    assert not value["GPU_same_backend_pass"]
