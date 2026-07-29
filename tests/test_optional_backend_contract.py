"""Public optional-native backend contract tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import gnss_gpu
from gnss_gpu.backends import (
    NativeBackendUnavailableError,
    backend_unavailable,
    is_missing_optional_module,
    unavailable_function,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_exported_name_is_defined():
    missing = [name for name in gnss_gpu.__all__ if not hasattr(gnss_gpu, name)]
    assert missing == []


def test_unavailable_function_has_actionable_uniform_error():
    function = unavailable_function("demo_feature", "gnss_gpu._missing_demo")
    with pytest.raises(NativeBackendUnavailableError, match="demo_feature") as caught:
        function(1, ignored=True)
    assert "gnss_gpu._missing_demo" in str(caught.value)
    assert "README.md" in str(caught.value)


def test_backend_error_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        backend_unavailable("feature", "module")


def test_only_exact_missing_extension_is_optional():
    exact = ModuleNotFoundError("missing", name="gnss_gpu._native")
    dependency = ModuleNotFoundError("missing", name="cudart64")
    generic = ImportError("broken binary")
    assert is_missing_optional_module(exact, "gnss_gpu._native")
    assert not is_missing_optional_module(dependency, "gnss_gpu._native")
    assert not is_missing_optional_module(generic, "gnss_gpu._native")


def test_cpu_only_import_defines_core_api_and_fails_on_call():
    script = r"""
import builtins

real_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "gnss_gpu._gnss_gpu":
        raise ModuleNotFoundError(
            "simulated CPU-only installation", name="gnss_gpu._gnss_gpu"
        )
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import gnss_gpu
assert gnss_gpu.HAS_NATIVE_CORE is False
for name in ("ecef_to_lla", "lla_to_ecef", "satellite_azel", "wls_position", "wls_batch"):
    assert hasattr(gnss_gpu, name)
try:
    gnss_gpu.wls_position([], [])
except gnss_gpu.NativeBackendUnavailableError:
    pass
else:
    raise AssertionError("CPU-only placeholder did not raise the backend error")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
