"""Read-only Windows x64 exception capture for owned diagnostic children.

DEBUG_PROCESS follows the venv launcher's children. Only the initial loader
breakpoint is handled; all other exceptions retain Windows' normal handling.
No code, registers, numerical state, or library bindings are written.
"""

import ctypes as C
from ctypes import wintypes as W
import os
import struct
import subprocess
import time


class ExceptionRecord(C.Structure):
    _fields_ = [("code", W.DWORD), ("flags", W.DWORD), ("nested", C.c_void_p),
        ("address", C.c_void_p), ("count", W.DWORD), ("information", C.c_size_t*15)]


class ExceptionInfo(C.Structure):
    _fields_ = [("record", ExceptionRecord), ("first", W.DWORD)]


class CreateProcess(C.Structure):
    _fields_ = [("file", W.HANDLE), ("process", W.HANDLE), ("thread", W.HANDLE),
        ("base", C.c_void_p), ("offset", W.DWORD), ("size", W.DWORD),
        ("local", C.c_void_p), ("start", C.c_void_p), ("image", C.c_void_p), ("unicode", W.WORD)]


class CreateThread(C.Structure):
    _fields_ = [("thread", W.HANDLE), ("local", C.c_void_p), ("start", C.c_void_p)]


class LoadDll(C.Structure):
    _fields_ = [("file", W.HANDLE), ("base", C.c_void_p), ("offset", W.DWORD),
        ("size", W.DWORD), ("image", C.c_void_p), ("unicode", W.WORD)]


class EventData(C.Union):
    _fields_ = [("exception", ExceptionInfo), ("create", CreateProcess),
        ("thread", CreateThread), ("load", LoadDll), ("exit_code", W.DWORD), ("base", C.c_void_p)]


class DebugEvent(C.Structure):
    _fields_ = [("kind", W.DWORD), ("pid", W.DWORD), ("tid", W.DWORD), ("data", EventData)]


class ThreadEntry(C.Structure):
    _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("tid", W.DWORD),
        ("pid", W.DWORD), ("priority", W.LONG), ("delta", W.LONG), ("flags", W.DWORD)]


def api():
    if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
        raise RuntimeError("Explicit Windows x64 diagnostic only")
    k = C.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "WaitForDebugEvent": ([C.POINTER(DebugEvent), W.DWORD], W.BOOL),
        "ContinueDebugEvent": ([W.DWORD, W.DWORD, W.DWORD], W.BOOL),
        "CloseHandle": ([W.HANDLE], W.BOOL),
        "GetFinalPathNameByHandleW": ([W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD], W.DWORD),
        "GetThreadContext": ([W.HANDLE, C.c_void_p], W.BOOL),
        "ReadProcessMemory": ([W.HANDLE, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)], W.BOOL),
        "TerminateProcess": ([W.HANDLE, W.UINT], W.BOOL),
        "CreateToolhelp32Snapshot": ([W.DWORD, W.DWORD], W.HANDLE),
        "Thread32First": ([W.HANDLE, C.POINTER(ThreadEntry)], W.BOOL),
        "Thread32Next": ([W.HANDLE, C.POINTER(ThreadEntry)], W.BOOL),
    }
    for name, (args, result) in signatures.items():
        function = getattr(k, name)
        function.argtypes, function.restype = args, result
    return k


def own_threads():
    k = api()
    handle = k.CreateToolhelp32Snapshot(4, 0)
    if handle == C.c_void_p(-1).value:
        raise C.WinError(C.get_last_error())
    entry, result = ThreadEntry(), []
    entry.size = C.sizeof(entry)
    try:
        good = k.Thread32First(handle, C.byref(entry))
        while good:
            if entry.pid == os.getpid():
                result.append(entry.tid)
            good = k.Thread32Next(handle, C.byref(entry))
    finally:
        k.CloseHandle(handle)
    return sorted(result)


def file_name(k, handle):
    if not handle:
        return None
    data = C.create_unicode_buffer(32768)
    count = k.GetFinalPathNameByHandleW(handle, data, len(data), 0)
    return data.value if 0 < count < len(data) else None


def registers(k, handle):
    # Windows x64 CONTEXT,16-byte aligned. Read CONTROL and INTEGER only.
    storage = C.create_string_buffer(1232+16)
    pointer = (C.addressof(storage)+15) & ~15
    C.c_uint32.from_address(pointer+48).value = 0x100003
    if not k.GetThreadContext(handle, pointer):
        return dict(error=C.get_last_error())
    data = C.string_at(pointer, 256)
    offsets = dict(rax=120, rcx=128, rdx=136, rbx=144, rsp=152, rbp=160,
        rsi=168, rdi=176, r8=184, r9=192, r10=200, r11=208, r12=216,
        r13=224, r14=232, r15=240, rip=248)
    return {name: struct.unpack_from("<Q", data, offset)[0] for name, offset in offsets.items()}


def image_size(k, handle, base):
    data, count = C.create_string_buffer(4096), C.c_size_t()
    if not k.ReadProcessMemory(handle, base, data, len(data), C.byref(count)):
        return None
    raw = data.raw[:count.value]
    if raw[:2] != b"MZ":
        return None
    offset = struct.unpack_from("<I", raw, 60)[0]
    if offset+84 > len(raw) or raw[offset:offset+4] != b"PE\0\0":
        return None
    return struct.unpack_from("<I", raw, offset+80)[0]


def debug_child(command, *, cwd, env, log, timeout_seconds):
    """Run only newly-created descendants; preserve exception disposition.

    On timeout only handles delivered for these owned processes are stopped.
    This debugger changes scheduling and must never produce speed evidence.
    """
    if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
        raise ValueError("Positive explicit diagnostic timeout required")
    k = api()
    assert C.sizeof(DebugEvent) == 176 and C.sizeof(ExceptionRecord) == 152
    events, processes, threads, modules, initial_breaks, exits = [], {}, {}, {}, set(), {}
    timed_out, started = False, time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=1)  # DEBUG_PROCESS,including venv redirector's child.
    error = None
    try:
        while True:
            if not timed_out and time.monotonic()-started > timeout_seconds:
                timed_out = True
                for handle in processes.values():
                    if not k.TerminateProcess(handle, 124):
                        raise C.WinError(C.get_last_error())
            event = DebugEvent()
            if not k.WaitForDebugEvent(C.byref(event), 100):
                code = C.get_last_error()
                if code not in (0, 121):
                    raise C.WinError(code)
                if exits and not processes:
                    break
                continue
            pid, tid, kind = event.pid, event.tid, event.kind
            row = dict(event=kind, pid=pid, tid=tid, elapsed_ns=int((time.monotonic()-started)*1e9))
            status = 0x10002  # DBG_CONTINUE
            try:
                if kind == 3:
                    info = event.data.create
                    processes[pid], threads[(pid, tid)] = info.process, info.thread
                    path = file_name(k, info.file)
                    size = image_size(k, info.process, int(info.base))
                    modules[pid] = {int(info.base): (path, size)}
                    row.update(path=path, main_tid=tid, base=int(info.base), image_size=size)
                    if info.file:
                        k.CloseHandle(info.file)
                elif kind == 2:
                    threads[(pid, tid)] = event.data.thread.thread
                    row["start_address"] = int(event.data.thread.start or 0)
                elif kind == 6:
                    info = event.data.load
                    path, base = file_name(k, info.file), int(info.base)
                    size = image_size(k, processes[pid], base)
                    modules[pid][base] = (path, size)
                    row.update(path=path, base=base, image_size=size)
                    if info.file:
                        k.CloseHandle(info.file)
                elif kind == 7:
                    row["base"] = int(event.data.base)
                    modules[pid].pop(int(event.data.base), None)
                elif kind == 4:
                    row["exit_code"] = int(event.data.exit_code)
                    threads.pop((pid, tid), None)
                elif kind == 5:
                    row["exit_code"] = exits[pid] = int(event.data.exit_code)
                    processes.pop(pid)
                    for key in [key for key in threads if key[0] == pid]:
                        threads.pop(key)
                elif kind == 1:
                    info, record = event.data.exception, event.data.exception.record
                    code, address = int(record.code), int(record.address or 0)
                    row.update(code=code, address=address, first_chance=bool(info.first),
                        information=[int(v) for v in record.information[:min(record.count, 15)]])
                    status = 0x80010001  # DBG_EXCEPTION_NOT_HANDLED
                    if code == 0x80000003 and pid not in initial_breaks:
                        initial_breaks.add(pid)
                        status = 0x10002
                    if code == 0xC0000005:
                        candidates = [base for base, (_, size) in modules[pid].items()
                            if size is not None and base <= address < base+size]
                        base = max(candidates) if candidates else None
                        row.update(module_base=base, module=modules[pid][base][0] if base is not None else None,
                            module_offset=address-base if base is not None else None,
                            registers=registers(k, threads[(pid, tid)]))
                        data, count = C.create_string_buffer(32), C.c_size_t()
                        if k.ReadProcessMemory(processes[pid], address, data, len(data), C.byref(count)):
                            row["instruction_bytes"] = data.raw[:count.value].hex()
                    row["continue_status"] = status
                events.append(row)
            finally:
                if not k.ContinueDebugEvent(pid, tid, status):
                    raise C.WinError(C.get_last_error())
                # Windows closes debug process/thread handles on EXIT events.
                # Do not close them twice after ContinueDebugEvent.
            if exits and not processes:
                break
        exit_code = child.wait(timeout=10)
    except BaseException as caught:
        error = type(caught).__name__+":"+str(caught)
        # Debugger exit also terminates owned debuggees by Windows' default.
        # Do not detach a partially observed child or report it as completed.
        raise
    finally:
        for handle in threads.values():
            k.CloseHandle(handle)
        for handle in processes.values():
            k.CloseHandle(handle)
    return dict(launcher_pid=child.pid, exit_code=exit_code, process_exit_codes=exits,
        timed_out=timed_out, debugger_error=error, events=events,
        speed_evidence=False, exception_disposition_unchanged=True)
