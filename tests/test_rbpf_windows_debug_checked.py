"""Owned diagnostic child control and native exception capture;no GPU needed."""

import ctypes
import os
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_windows_debug import DebugEvent, ExceptionRecord, ThreadEntry, debug_child, own_threads  # noqa: E402


def run_child(tmp_path, code, timeout=10):
    prefix = "import ctypes;ctypes.windll.kernel32.SetErrorMode(2);"
    with (tmp_path / "child.log").open("wb") as log:
        return debug_child([sys.executable, "-B", "-c", prefix+code], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), log=log, timeout_seconds=timeout)


def test_windows_x64_struct_sizes_and_union_offsets():
    assert ctypes.sizeof(DebugEvent) == 176 and DebugEvent.data.offset == 16
    assert ctypes.sizeof(ExceptionRecord) == 152 and ExceptionRecord.information.offset == 32
    assert ctypes.sizeof(ThreadEntry) == 28


def test_thread_snapshot_contains_current_native_id():
    tids = own_threads()
    assert tids == sorted(set(tids)) and threading.get_native_id() in tids


def test_normal_exit_and_module_image_bounds(tmp_path):
    result = run_child(tmp_path, "print('finished')")
    assert result["exit_code"] == 0 and not result["timed_out"] and result["debugger_error"] is None
    creates = [r for r in result["events"] if r["event"] == 3]
    assert len(creates) == 1 and creates[0]["pid"] == result["launcher_pid"]
    assert creates[0]["image_size"] > 0 and creates[0]["path"].lower().endswith("python.exe")
    assert any(r["event"] == 6 and (r["path"] or "").lower().endswith("python312.dll") for r in result["events"])
    assert b"finished" in (tmp_path / "child.log").read_bytes()


def test_unhandled_access_violation_native_context_and_normal_disposition(tmp_path):
    # CreateThread accepts an invalid start address; its isolated native
    # thread then raises an unhandled execute AV,without ctypes' SEH wrapper.
    code = "k=ctypes.WinDLL('kernel32');k.CreateThread.restype=ctypes.c_void_p;k.CreateThread.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p];h=k.CreateThread(None,0,None,None,0,None);assert h;import time;time.sleep(5)"
    result = run_child(tmp_path, code)
    assert result["exit_code"] == 0xC0000005
    faults = [v for v in result["events"] if v.get("code") == 0xC0000005]
    assert [v["first_chance"] for v in faults] == [True, False]
    for fault in faults:
        assert fault["continue_status"] == 0x80010001 and fault["registers"]["rip"] == fault["address"]
        assert fault["address"] == 0 and fault["module"] is None and fault["module_offset"] is None
        assert fault["information"] == [8, 0]
    assert result["exception_disposition_unchanged"] and not result["speed_evidence"]


def test_handled_access_violation_is_not_swallowed_or_forced_fatal(tmp_path):
    code = "\ntry:\n ctypes.memmove(0,b'x',1)\nexcept OSError:\n print('caught')\n"
    result = run_child(tmp_path, code)
    assert result["exit_code"] == 0 and b"caught" in (tmp_path / "child.log").read_bytes()
    faults = [v for v in result["events"] if v.get("code") == 0xC0000005]
    assert faults and all(v["first_chance"] and v["continue_status"] == 0x80010001 for v in faults)
    assert all(v["registers"]["rip"] == v["address"] and v["module"] and v["module_offset"] >= 0 and v["instruction_bytes"] for v in faults)


def test_debug_process_follows_nested_child_and_keeps_both_exits(tmp_path):
    code = "import subprocess,sys;p=subprocess.run([sys.executable,'-B','-c','raise SystemExit(37)']);raise SystemExit(p.returncode)"
    result = run_child(tmp_path, code)
    assert result["exit_code"] == 37 and len(result["process_exit_codes"]) == 2
    assert set(result["process_exit_codes"].values()) == {37}


def test_timeout_stops_only_owned_sleeping_process(tmp_path):
    result = run_child(tmp_path, "import time;time.sleep(10)", timeout=0.5)
    assert result["timed_out"] and result["exit_code"] == 124
    assert set(result["process_exit_codes"].values()) == {124}


@pytest.mark.parametrize("value", [0, -1, None, True])
def test_bad_timeout_rejected_before_child_creation(tmp_path, value):
    with pytest.raises(ValueError):
        run_child(tmp_path, "raise SystemExit(99)", timeout=value)
