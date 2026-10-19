"""Portable exit-evidence policy for the new RTK measurement protocol."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_parent_measurement import normal_numeric_exit  # noqa: E402

VERSION = ["C:/Windows/System32/cmd.exe", "/c", "ver"]


def evidence(extra):
    command = ["venv/python.exe", "worker.py"]
    result = dict(launcher_pid=1, worker_pid=2, command=command, launcher_exit_code=0,
        acknowledged=True, all_observed_processes_exited=True, timed_out=False,
        ready_timed_out=False, errors=[], terminated_pids=[], processes=[
            dict(pid=1, parent_pid=100, command=command, creation_filetime=10, exited=True, exit_code=0),
            dict(pid=2, parent_pid=1, command=["python.exe", "worker.py"], creation_filetime=20, exited=True, exit_code=0)])
    if extra:
        result["processes"].append(dict(pid=3, parent_pid=2, command=VERSION,
            creation_filetime=30, exited=True, exit_code=0))
    return result


@pytest.mark.parametrize("extra", [False, True])
def test_observed_known_processes_must_all_exit0_without_mutation(extra):
    record = evidence(extra)
    saved = deepcopy(record)
    assert normal_numeric_exit(record, VERSION)
    assert record == saved


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("change", ["exit", "running", "parent", "birth", "timeout", "error", "forced", "ack", "duplicate"])
def test_bad_worker_or_monitor_evidence_is_rejected(extra, change):
    record = evidence(extra)
    if change == "exit":
        record["processes"][1]["exit_code"] = 1
    elif change == "running":
        record["processes"][1]["exited"] = False
    elif change == "parent":
        record["processes"][1]["parent_pid"] = 100
    elif change == "birth":
        record["processes"][1]["creation_filetime"] = 5
    elif change == "timeout":
        record["timed_out"] = True
    elif change == "error":
        record["errors"] = ["error"]
    elif change == "forced":
        record["terminated_pids"] = [2]
    elif change == "ack":
        record["acknowledged"] = False
    else:
        record["processes"][1]["pid"] = 1
    assert not normal_numeric_exit(record, VERSION)


@pytest.mark.parametrize("change", ["command", "parent", "exit", "unknown"])
def test_observed_extra_is_never_silently_ignored(change):
    record = evidence(True)
    if change == "command":
        record["processes"][2]["command"] = ["cmd.exe", "/c", "other"]
    elif change == "parent":
        record["processes"][2]["parent_pid"] = 1
    elif change == "exit":
        record["processes"][2]["exit_code"] = 1
    else:
        record["processes"].append(dict(pid=4))
    assert not normal_numeric_exit(record, VERSION)
