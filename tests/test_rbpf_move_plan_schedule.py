"""Reject noisy controls, insufficient off-control gains and ratio averaging."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_move_plan import MODES, SEEDS, summaries  # noqa: E402


def rows(change=None, split=False):
    times = dict(original_cpu=1000, off_cpu=1000, candidate_cpu=700,
        original_gpu=900, off_gpu=900, candidate_gpu=500, cpu_AA=1000, gpu_AA=900)
    result = []
    for r in range(5):
        for g in range(3):
            for mode in MODES:
                ns = times[mode]
                if change:
                    ns = change(r, g, mode, ns)
                values = (1, ns - 1) if split else (ns,)
                for v in values:
                    result.append(dict(round=r, group=g, mode=mode, ns=v))
    return result


def test_all_required_original_off_and_joint_contrasts_pass():
    value = summaries(rows())
    assert len(MODES) == 8 and len(SEEDS) == 16 and len(set(SEEDS)) == 16
    assert value["CPU_pass"] and value["GPU_same_backend_pass"] and value["GPU_joint_pass"]


def test_original_gain_does_not_override_small_off_control_gain():
    value = summaries(rows(lambda r, g, mode, ns: 720 if mode == "off_cpu" else ns))
    assert value["contrasts"]["cpu"]["numeric_screen"]
    assert not value["contrasts"]["cpu_off"]["numeric_screen"] and not value["CPU_pass"]


def test_signed_aa_cancellation_is_rejected():
    value = summaries(rows(lambda r, g, mode, ns: (1200, 1000, 800)[g] if mode == "cpu_AA" else ns))
    assert value["contrasts"]["cpu_AA"]["median_paired_percent"] == 0
    assert value["contrasts"]["cpu_AA"]["absolute_median_percent"] == pytest.approx(20)
    assert not value["cpu_AA_valid"] and not value["CPU_pass"] and not value["GPU_joint_pass"]


def test_median_aa_requires_four_stable_rounds():
    value = summaries(rows(lambda r, g, mode, ns: 1200 if mode == "cpu_AA" and r >= 3 else ns))
    assert value["contrasts"]["cpu_AA"]["absolute_median_percent"] == 0
    assert value["contrasts"]["cpu_AA"]["absolute_rounds_under5"] == 3
    assert not value["cpu_AA_valid"]


def test_integer_call_times_are_summed_before_ratios():
    def alter(r, g, mode, ns):
        return 100 if mode == "candidate_cpu" else ns
    value = summaries(rows(alter, split=True))
    assert value["contrasts"]["cpu"]["median_paired_percent"] == pytest.approx(90)
    assert value["contrasts"]["cpu_AA"]["absolute_median_percent"] == 0
