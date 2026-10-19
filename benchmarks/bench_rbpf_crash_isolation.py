"""Pure-Python and unchanged RTK factorial safety diagnostics;not speed tests."""

import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import FunctionType

from rbpf_windows_debug import own_threads

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def pin():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = k.GetCurrentProcess()
    assert k.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
    assert k.SetProcessAffinityMask(handle, 1 << (allowed.value.bit_length()-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    trial = plan["trials"][args.trial]
    assert trial["trial"] == args.trial
    for name, digest in plan["source_sha256"].items():
        assert ".claude" not in [p.casefold() for p in Path(name).parts]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    ctypes.windll.kernel32.SetErrorMode(2)
    faulthandler.enable()
    pin()
    folder = ROOT / trial["metadata"]
    assert not folder.exists()
    folder.mkdir(parents=True)
    before = own_threads()
    metadata = dict(trial=trial, pid=os.getpid(), parent_pid=os.getppid(), main_tid=threading.get_native_id(),
        Python=sys.version, executable=sys.executable, before_arm_native_tids=before,
        periodic_seconds=trial["watchdog_seconds"], speed_evidence=False)
    put(folder / "before_arm.json", metadata)
    trace = (folder / "periodic_trace.log").open("xb")
    try:
        if trial["watchdog_seconds"]:
            faulthandler.dump_traceback_later(trial["watchdog_seconds"], repeat=True, file=trace)
        after = own_threads()
        put(folder / "armed.json", dict(metadata, after_arm_native_tids=after,
            watchdog_new_native_tids=sorted(set(after)-set(before))))
        if trial["workload"] == "pure":
            assert not any(n.split(".")[0] in {"numpy", "numba", "gtsam", "rbpf_fgo"} for n in sys.modules)
            # Exercise temporary Python frames and callbacks from builtin map.
            # No NumPy,GNSS/CUDA library,device operation,or numeric fixture.
            code = compile("def frame(n):\n return sum(map(lambda v:v+n,range(16)))\n", "<pure-frame-churn>", "exec")
            scope = {}
            exec(code, scope)
            template = scope["frame"]
            start, count, total = time.monotonic(), 0, 0
            while time.monotonic()-start < plan["pure_seconds"]:
                function = FunctionType(template.__code__, template.__globals__)
                total += function(count & 7)
                count += 1
            result = dict(verified=True, pid=os.getpid(), workload="pure", iterations=count,
                checksum=total, no_numpy_GNSS_or_CUDA_imports=True, speed_evidence=False)
            output = ROOT / trial["output"]
            output.mkdir(parents=True)
            put(output / "result.json", result)
        else:
            assert trial["workload"] == "field" and not trial["accuracy"]
            import bench_rbpf_transition_wake_field as original
            assert original.main() == 0
        put(folder / "completed.json", dict(completed=True, pid=os.getpid(), speed_evidence=False))
    finally:
        faulthandler.cancel_dump_traceback_later()
        trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
