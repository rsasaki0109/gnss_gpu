"""Collect portable checks and require the recorded native experiment runtime."""

import sys
import numpy as np
import pytest
import rbpf_campaign_source_policy_checked as checked

pytest_configure = checked.pytest_configure
pytest_pycollect_makemodule = checked.pytest_pycollect_makemodule


def pytest_collection_modifyitems(items):
    checked.pytest_collection_modifyitems(items)
    if sys.platform != "win32" or np.__version__ != "2.3.4":
        for item in items:
            if item.path.parent == checked.ROOT / "tests" and item.path.name in {
                "test_rbpf_extend_factor.py", "test_rbpf_extend_factor_pivots.py"
            }:
                item.add_marker(pytest.mark.skip(reason="Frozen LAPACK experiment requires its certified Windows/NumPy 2.3.4 runtime"))
