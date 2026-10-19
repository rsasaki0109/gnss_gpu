"""Platform collection changes must leave all other tests untouched."""

from pathlib import Path

import pytest

from rbpf_gc_clocks_platform import mark_windows_cycle_tests


@pytest.mark.parametrize("windows", (True, False))
def test_only_real_windows_counter_cases_are_marked(windows):
    class Item:
        def __init__(self, path, name):
            self.path, self.name, self.marks = Path(path), name, []

        def add_marker(self, mark):
            self.marks.append(mark)

    names = (
        "test_windows_cycle_clock_monotonic_integer_samples",
        "test_real_independent_children_keep_extended_samples",
    )
    chosen = [Item("test_rbpf_gc_clocks.py", name) for name in names]
    other = [Item("test_rbpf_gc_clocks.py", "test_exclusive_phases_restore_and_keep_gc_policy"),
        Item("test_other.py", names[0])]
    mark_windows_cycle_tests([*chosen, *other], windows=windows)
    assert all(not item.marks for item in other)
    if windows:
        assert all(not item.marks for item in chosen)
    else:
        assert all(len(item.marks) == 1 and item.marks[0].name == "skip" for item in chosen)
