"""Unchanged 400-epoch caller after a parent-only lifecycle handshake."""

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

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    trial = plan["trials"][args.trial]
    assert trial["trial"] == args.trial
    ctypes.windll.kernel32.SetErrorMode(2)
    faulthandler.enable()
    folder = ROOT / trial["metadata"]
    ready = dict(pid=os.getpid(), parent_pid=os.getppid(), argv=sys.argv,
        main_native_tid=threading.get_native_id(), Python=sys.version,
        periodic_stack_dump=False, speed_evidence=False)
    temporary = folder / "ready.tmp"
    put(temporary, ready)
    temporary.rename(folder / "ready.json")
    deadline = time.monotonic() + plan["ready_seconds"] + 10
    while not (folder / "go.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Parent did not acknowledge worker identity")
        time.sleep(0.01)
    gate = json.loads((folder / "go.json").read_text(encoding="utf8"))
    assert gate["pid"] == os.getpid()
    for name, digest in plan["source_sha256"].items():
        assert ".claude" not in [p.casefold() for p in Path(name).parts]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    import bench_rbpf_transition_wake_field as original
    assert original.main() == 0
    put(folder / "completed.json", dict(pid=os.getpid(), returned=True,
        normal_process_exit_not_yet_proven=True, speed_evidence=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
