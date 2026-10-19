"""Regression for termination after process exit,and preserved native capture."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_windows_debug_checked import checked_api, debug_child  # noqa: E402


def test_already_exited_owned_process_keeps_its_original_exit_code():
    child = subprocess.Popen([sys.executable, "-B", "-c", "raise SystemExit(37)"])
    assert child.wait(timeout=10) == 37
    kernel = checked_api()
    assert kernel.TerminateProcess(child._handle, 124)
    assert child.wait() == 37


def test_live_owned_process_timeout_still_terminates(tmp_path):
    with (tmp_path / "child.log").open("wb") as log:
        result = debug_child([sys.executable, "-B", "-c", "import time;time.sleep(10)"],
            cwd=ROOT, env=dict(os.environ), log=log, timeout_seconds=0.5)
    assert result["timed_out"] and result["exit_code"] == 124
    assert set(result["process_exit_codes"].values()) == {124}


def test_context_capture_still_preserves_handled_AV(tmp_path):
    code = "import ctypes;ctypes.windll.kernel32.SetErrorMode(2)\ntry:\n ctypes.memmove(0,b'x',1)\nexcept OSError:\n pass"
    with (tmp_path / "child.log").open("wb") as log:
        result = debug_child([sys.executable, "-B", "-c", code], cwd=ROOT,
            env=dict(os.environ), log=log, timeout_seconds=10)
    assert result["exit_code"] == 0
    faults = [v for v in result["events"] if v.get("code") == 0xC0000005]
    assert faults and all(v["first_chance"] and v["continue_status"] == 0x80010001 for v in faults)
    assert all(v["registers"]["rip"] == v["address"] and v["module"] for v in faults)
