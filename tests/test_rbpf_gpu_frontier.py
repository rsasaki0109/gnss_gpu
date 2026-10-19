"""A readiness ceiling must not be mistaken for actual readiness or speed."""

from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_gpu_frontier import ceiling, source_bound, summarize  # noqa: E402


def test_frozen_source_has_four_held_one_empty_and_late_score():
    value = source_bound(ROOT / "benchmarks/rbpf_move_batch.py")
    assert value["held_prefix_maximum"] == value["prefix_maximum"] == 4
    assert value["empty_prefix_maximum"] == 1 and value["common_after_first_prefix"]


def test_changed_source_is_rejected_before_its_AST_is_trusted(tmp_path):
    changed = tmp_path / "changed.py"
    changed.write_text("pass\n", encoding="utf8")
    with pytest.raises(ValueError, match="source changed"):
        source_bound(changed)


def test_more_pending_than_first_records_requires_later_work():
    row = ceiling(particles=64, unique_moves=22, existing_cache_hits=1, common_records=13)
    assert row["pending"] == 21 and row["ready_ceiling"] == 17
    assert row["definitely_later_minimum"] == 4 and not row["single_batch_still_possible"]


def test_cache_hits_are_not_subtracted_twice_from_the_ceiling():
    row = ceiling(particles=64, unique_moves=18, existing_cache_hits=1, common_records=13)
    assert row["ready_ceiling"] == row["pending"] == 17 and row["single_batch_still_possible"]


def test_no_pending_is_described_but_cannot_pass_nonempty_batch_triage():
    row = ceiling(particles=1, unique_moves=1, existing_cache_hits=1, common_records=0)
    assert row["pending"] == 0 and row["ready_fraction_ceiling"] is None
    with pytest.raises(ValueError):
        summarize([row])


@pytest.mark.parametrize("field,value", (("particles", 0), ("particles", True),
    ("unique_moves", -1), ("existing_cache_hits", 100), ("common_records", 1.5), ("prefix_maximum", 0)))
def test_invalid_work_counts_are_rejected(field, value):
    args = dict(particles=64, unique_moves=21, existing_cache_hits=1, common_records=13)
    args[field] = value
    with pytest.raises(ValueError):
        ceiling(**args)


def test_equality_at_the_predeclared_ninety_percent_threshold():
    possible = ceiling(particles=64, unique_moves=17, existing_cache_hits=0, common_records=13)
    impossible = ceiling(particles=64, unique_moves=18, existing_cache_hits=0, common_records=13)
    assert summarize([possible] * 9 + [impossible])["proceed_to_early_single_batch_prototype"]
    assert not summarize([possible] * 8 + [impossible] * 2)["proceed_to_early_single_batch_prototype"]


def test_exhaustive_prefix_record_counts_and_random_duplicate_cache_sets_obey_upper_bound():
    rng = random.Random(20261018)
    for prefix in range(5):
        for common in range(15):
            for _ in range(8):
                first = [rng.randrange(24) for _ in range(prefix + common)]
                later = [rng.randrange(32) for _ in range(20)]
                cached = set(rng.sample(range(32), rng.randrange(10)))
                all_keys = set(first + later)
                actual_first_pending = len(set(first) - cached)
                row = ceiling(particles=64, unique_moves=len(all_keys),
                    existing_cache_hits=len(all_keys & cached), common_records=common)
                assert actual_first_pending <= row["ready_ceiling"]
                if not row["single_batch_still_possible"]:
                    assert len(all_keys - cached - set(first)) >= row["definitely_later_minimum"] > 0


def test_weighted_readiness_is_sum_of_counts_not_mean_of_fractions():
    rows = [ceiling(particles=64, unique_moves=100, existing_cache_hits=0, common_records=0),
        ceiling(particles=64, unique_moves=1, existing_cache_hits=0, common_records=0)]
    value = summarize(rows)
    assert value["optimistic_ready_fraction"] == pytest.approx(5 / 101)
    assert value["actual_early_launches_observed"] == value["new_timings"] == 0
    assert not value["proceed_to_early_single_batch_prototype"]
