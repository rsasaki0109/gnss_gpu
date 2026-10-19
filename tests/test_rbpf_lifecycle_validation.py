"""Saved-process evidence policy; portable, no native API or numerical work."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_lifecycle_validation import normal_field_exit  # noqa: E402

VERSION = ["C:/Windows/System32/cmd.exe", "/c", "ver"]


def evidence():
    command = ["venv/python.exe", "worker.py"]
    return dict(launcher_pid=1, worker_pid=2, command=command, launcher_exit_code=0,
        acknowledged=True, all_observed_processes_exited=True, timed_out=False,
        ready_timed_out=False, errors=[], terminated_pids=[], processes=[
            dict(pid=1, parent_pid=100, command=command, creation_filetime=10, exited=True, exit_code=0),
            dict(pid=2, parent_pid=1, command=["python.exe", "worker.py"], creation_filetime=20, exited=True, exit_code=0),
            dict(pid=3, parent_pid=2, command=VERSION, creation_filetime=30, exited=True, exit_code=0)])


def test_all_three_normal_exits_are_required_and_input_is_preserved():
    record = evidence()
    saved = deepcopy(record)
    assert normal_field_exit(record, VERSION)
    assert record == saved


@pytest.mark.parametrize("change", [
    "launcher_failure", "worker_failure", "version_failure", "unknown_command",
    "wrong_parent", "still_running", "timeout", "forced", "error", "no_ack",
    "unobserved_exit", "extra_process", "duplicate_pid", "birth_before_parent"])
def test_rejects_incomplete_or_failed_observed_process_evidence(change):
    record = evidence()
    if change == "launcher_failure":
        record["launcher_exit_code"] = 1
    elif change == "worker_failure":
        record["processes"][1]["exit_code"] = 1
    elif change == "version_failure":
        record["processes"][2]["exit_code"] = 259
    elif change == "unknown_command":
        record["processes"][2]["command"] = ["cmd.exe", "/c", "other"]
    elif change == "wrong_parent":
        record["processes"][2]["parent_pid"] = 100
    elif change == "still_running":
        record["processes"][2]["exited"] = False
    elif change == "timeout":
        record["timed_out"] = True
    elif change == "forced":
        record["terminated_pids"] = [2]
    elif change == "error":
        record["errors"] = [dict(message="error")]
    elif change == "no_ack":
        record["acknowledged"] = False
    elif change == "unobserved_exit":
        record["all_observed_processes_exited"] = False
    elif change == "extra_process":
        record["processes"].append(dict(pid=4))
    elif change == "duplicate_pid":
        record["processes"][2]["pid"] = 2
    elif change == "birth_before_parent":
        record["processes"][2]["creation_filetime"] = 15
    assert not normal_field_exit(record, VERSION)
