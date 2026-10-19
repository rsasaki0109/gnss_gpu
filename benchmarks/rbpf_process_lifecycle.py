"""Parent-only Windows process observation for isolated numerical diagnostics.

The worker waits for a file acknowledgement before importing numerical code.
Keep process handles open until signalled; a result file is not an exit proof.
No debugger, child watchdog, stack sampling, or numerical callback is installed.
"""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time


class ProcessHandle:
    def __init__(self, process):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel, self.handle = kernel, None
        created = process.create_time()
        assert process.is_running(), "Process identity changed before opening"
        self.handle = kernel.OpenProcess(0x100000 | 0x1000 | 1, False, process.pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(self.handle, *[ctypes.byref(t) for t in times]):
                raise ctypes.WinError(ctypes.get_last_error())
            ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            assert abs(ticks / 10**7 - 11644473600 - created) < 0.001, "Process identity changed while opening"
            self.record = dict(pid=process.pid, parent_pid=process.ppid(),
                create_time=created, creation_filetime=ticks, command=process.cmdline())
            assert process.is_running(), "Process identity changed after opening"
        except BaseException:
            self.close()
            raise

    def exited(self):
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return status == 0

    def exit_code(self):
        if not self.exited():
            return None
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def stop(self):
        if self.exited():
            return False
        if not self.kernel.TerminateProcess(self.handle, 124):
            error = ctypes.get_last_error()
            if not self.exited():
                raise ctypes.WinError(error)
        return True

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def monitor_child(command, *, cwd, env, log, metadata, worker_argv,
                  timeout_seconds, ready_seconds=30, poll_seconds=0.1):
    """Observe owned handles, including a venv launcher and its numeric child.

    Descendants are captured while their observed parent is still alive. This
    polling protocol is for the fixed worker, which creates no subprocesses;
    it is not a sandbox for arbitrary detached or short-lived process trees.
    Timeout cleanup touches only handles whose ancestry and creation time were
    checked. Neither a completed marker nor a saved numeric result ends waiting.
    """
    import psutil

    if os.name != "nt":
        raise RuntimeError("Windows process handles required")
    if min(timeout_seconds, ready_seconds, poll_seconds) <= 0:
        raise ValueError("Positive timeouts and polling interval required")
    metadata = Path(metadata)
    metadata.mkdir(parents=True, exist_ok=False)
    owned, processes, errors, stopped = {}, {}, [], []
    worker_pid, ready, acked = None, None, False
    timed_out, ready_timed_out = False, False
    start = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)

    def capture(process):
        if process.pid not in owned:
            handle = ProcessHandle(process)
            owned[process.pid], processes[process.pid] = handle, process

    def discover():
        for pid, process in list(processes.items()):
            if owned[pid].exited():
                continue
            try:
                descendants = process.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            for descendant in descendants:
                try:
                    capture(descendant)
                except psutil.NoSuchProcess:
                    pass

    try:
        capture(psutil.Process(child.pid))
        while True:
            discover()
            if not acked and (metadata / "ready.json").exists():
                ready = json.loads((metadata / "ready.json").read_text(encoding="utf8"))
                worker_pid = ready["pid"]
                assert worker_pid in owned, "Ready PID is not in the owned process tree"
                handle = owned[worker_pid]
                assert not handle.exited()
                assert ready["parent_pid"] == handle.record["parent_pid"]
                assert worker_pid == child.pid or ready["parent_pid"] == child.pid
                assert ready["argv"] == worker_argv
                assert handle.record["command"][-len(worker_argv):] == worker_argv
                temporary = metadata / "go.tmp"
                with temporary.open("x", encoding="utf8") as stream:
                    json.dump(dict(pid=worker_pid, parent_pid=os.getpid()), stream)
                temporary.rename(metadata / "go.json")
                acked = True
            launcher_done = child.poll() is not None
            if launcher_done and all(h.exited() for h in owned.values()):
                break
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds or (not acked and elapsed >= ready_seconds):
                timed_out, ready_timed_out = True, not acked
                break
            time.sleep(poll_seconds)
    except Exception as error:
        errors.append(dict(type=type(error).__name__, message=str(error)))
    finally:
        # Recheck handle state, including a process which exited at the timeout.
        try:
            discover()
        except Exception as error:
            errors.append(dict(stage="cleanup_discovery", type=type(error).__name__, message=str(error)))
        for pid, handle in reversed(list(owned.items())):
            try:
                if handle.stop():
                    stopped.append(pid)
            except Exception as error:
                errors.append(dict(stage="cleanup", pid=pid, type=type(error).__name__, message=str(error)))
        if child.poll() is None:
            # Popen retains its original OS handle even if initial capture failed.
            child.kill()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            errors.append(dict(stage="launcher_wait", type="TimeoutExpired"))
        deadline = time.monotonic() + 10
        while any(not h.exited() for h in owned.values()) and time.monotonic() < deadline:
            time.sleep(poll_seconds)
        records = [dict(h.record, exited=h.exited(), exit_code=h.exit_code()) for h in owned.values()]
        for handle in owned.values():
            handle.close()
    return dict(command=command, launcher_pid=child.pid, launcher_exit_code=child.returncode,
        worker_pid=worker_pid, ready=ready, acknowledged=acked,
        processes=records, timed_out=timed_out, ready_timed_out=ready_timed_out,
        terminated_pids=stopped, errors=errors, elapsed_seconds=time.monotonic()-start,
        all_observed_processes_exited=all(v["exited"] for v in records),
        debugger_used=False, child_periodic_stack_dump=False, speed_evidence=False)
