"""Stability gates must reject signed cancellation and too few stable rounds."""

import pytest

from bench_rbpf_direct_view import MODES, summaries


def rows(cpu_aa):
    result = []
    for rd in range(5):
        for group in range(3):
            times = dict(original_cpu=1000, candidate_cpu=800, original_gpu=750,
                candidate_gpu=600, cpu_AA=cpu_aa[rd][group], gpu_AA=750)
            for mode in MODES:
                result.append(dict(round=rd, group=group, mode=mode, ns=times[mode]))
    return result


def test_signed_aa_cancellation_is_rejected():
    result = summaries(rows([[900, 1100, 1000]] * 5))
    assert result["contrasts"]["cpu_AA"]["median_paired_percent"] == 0
    assert result["contrasts"]["cpu_AA"]["absolute_median_percent"] > 9
    assert not result["cpu_AA_valid"] and not result["CPU_pass"] and not result["GPU_joint_pass"]


@pytest.mark.parametrize("bad_rounds,valid", ((1, True), (2, False)))
def test_four_stable_rounds_required_even_when_median_is_small(bad_rounds, valid):
    result = summaries(rows([[920] * 3] * bad_rounds + [[980] * 3] * (5 - bad_rounds)))
    assert result["contrasts"]["cpu_AA"]["absolute_median_percent"] < 5
    assert result["cpu_AA_valid"] is valid and result["CPU_pass"] is valid


def test_integer_ticks_are_summed_before_pair_ratios():
    source = rows([[1000] * 3] * 5)
    doubled = [{**r, "ns": r["ns"] * 3} for r in source]
    assert summaries(source + doubled) == summaries(source)
