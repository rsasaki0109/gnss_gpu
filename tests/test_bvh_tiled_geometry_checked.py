"""Same frozen GPU tests; old CPU reference alone uses a sufficient stack."""

import hashlib
import importlib.util
import os
from pathlib import Path
import struct
import sys
import threading

import numpy as np
import pytest

if (
    os.name != "nt"
    or sys.version_info[:2] != (3, 12)
    or struct.calcsize("P") != 8
    or not os.environ.get("GNSS_BVH_TILED_EXTENSION")
):
    pytest.skip("Explicit checked BVH probe required", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "tests/test_bvh_tiled_geometry.py"
SPEC = importlib.util.spec_from_file_location("bvh_tiled_initial_frozen_tests", INITIAL)
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
modules = OLD.modules
pytestmark = OLD.pytestmark
for name, value in vars(OLD).items():
    if (
        name.startswith("test_")
        and name != "test_rebuilt_cpu_builder_preserves_nodes_and_permutation"
    ):
        globals()[name] = value


def large_stack_reference(module, mesh):
    result, error = [], []

    def run():
        try:
            result.append(module.bvh_build(mesh))
        except BaseException as exc:
            error.append(exc)

    previous = threading.stack_size(64 * 1024 * 1024)
    try:
        thread = threading.Thread(target=run)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if error:
        raise error[0]
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("count", [1, 64, 4096, 4097, 8192])
def test_rebuilt_cpu_builder_preserves_nodes_and_permutation(modules, count):
    old, new = modules
    assert new._bvh_iterative_build is True
    for mesh in (OLD._random_mesh(count), np.zeros((count, 3, 3))):
        # Only the frozen original recursive CPU reference needs another stack.
        # Candidate bvh_build executes on this ordinary pytest main thread.
        expected = large_stack_reference(old, mesh)
        OLD.bits(expected, new.bvh_build(mesh))
