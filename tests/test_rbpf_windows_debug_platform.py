"""Platform collection policy without native calls or global platform patches."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rbpf_gc_clocks_platform import mark_windows_cycle_tests, mark_windows_debug_tests


def item(path, name="test_example"):
    marks = []
    return SimpleNamespace(path=Path(path), name=name, marks=marks, add_marker=marks.append)


def test_non_windows_native_cases_are_skipped_and_unrelated_tests_remain():
    values = [item("test_rbpf_windows_debug.py"), item("test_rbpf_windows_debug_checked.py"),
        item("test_rbpf_windows_debug_timeout.py"), item("test_rbpf_transition_wake.py")]
    mark_windows_debug_tests(values, windows_x64=False)
    assert [len(v.marks) for v in values] == [1,1,1,0]
    assert all(v.marks[0].name == "skip" for v in values[:3])


def test_windows_x64_keeps_native_cases_runnable():
    value = item("test_rbpf_windows_debug_checked.py")
    mark_windows_debug_tests([value], windows_x64=True)
    assert value.marks == []


def test_existing_real_cycle_counter_guard_is_preserved():
    values = [item("test_rbpf_gc_clocks.py", "test_windows_cycle_clock_monotonic_integer_samples"),
        item("test_rbpf_gc_clocks.py", "test_real_independent_children_keep_extended_samples"),
        item("test_rbpf_gc_clocks.py", "test_other_clock_logic")]
    mark_windows_cycle_tests(values, windows=False)
    mark_windows_debug_tests(values, windows_x64=False)
    assert [len(v.marks) for v in values] == [1,1,0]


def test_platform_guard_does_not_create_blanket_prefix_skips():
    values = [item("test_rbpf_windows_debug_platform.py"), item("test_rbpf_native_call_trace.py")]
    mark_windows_debug_tests(values, windows_x64=False)
    assert all(v.marks == [] for v in values)
