"""Collect the two real Windows cycle-counter probes only on Windows."""

import struct
import sys

import pytest


def mark_windows_cycle_tests(items, *, windows):
    if windows:
        return
    names = {
        "test_windows_cycle_clock_monotonic_integer_samples",
        "test_real_independent_children_keep_extended_samples",
    }
    for item in items:
        if item.path.name == "test_rbpf_gc_clocks.py" and item.name in names:
            item.add_marker(pytest.mark.skip(reason="Real QueryThreadCycleTime probe requires Windows"))


def mark_windows_debug_tests(items, *, windows_x64):
    if windows_x64:
        return
    names = {"test_rbpf_windows_debug.py", "test_rbpf_windows_debug_checked.py",
        "test_rbpf_windows_debug_timeout.py"}
    for item in items:
        if item.path.name in names:
            item.add_marker(pytest.mark.skip(reason="Native exception-capture diagnostic requires Windows x64"))


def pytest_collection_modifyitems(items):
    mark_windows_cycle_tests(items, windows=sys.platform == "win32")
    mark_windows_debug_tests(items, windows_x64=sys.platform == "win32" and struct.calcsize("P") == 8)
