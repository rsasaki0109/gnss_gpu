"""Native exit status, timeout ownership, and launcher/worker separation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

if os.name != "nt":
    pytest.skip("Windows process handles required", allow_module_level=True)
psutil = pytest.importorskip("psutil")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_process_lifecycle import monitor_child  # noqa: E402


@pytest.fixture
def worker(tmp_path):
    source = tmp_path / "worker.py"
    source.write_text('''import json,os,pathlib,subprocess,sys,time
folder=pathlib.Path(sys.argv[1]);mode=sys.argv[2]
if mode=="launcher":
 child=subprocess.Popen([sys.executable,"-B",__file__,str(folder),sys.argv[3]])
 raise SystemExit(child.wait())
temporary=folder/"ready.tmp"
temporary.write_text(json.dumps(dict(pid=os.getpid(),parent_pid=os.getppid(),argv=sys.argv)))
temporary.rename(folder/"ready.json")
while not (folder/"go.json").exists():time.sleep(.01)
if mode=="marker_hang":
 (folder/"completed.json").write_text("{}")
 (folder/"result.json").write_text("{}")
 time.sleep(60)
elif mode=="nested_hang":
 child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(60)"])
 (folder/"nested.json").write_text(json.dumps(dict(pid=child.pid)))
 time.sleep(60)
else:raise SystemExit(int(mode))
''', encoding="utf8")
    return source


def run(tmp_path, worker, mode, *, launcher=False, timeout=10):
    folder = tmp_path / "metadata"
    argv = [str(worker), str(folder), mode]
    command = [sys.executable, "-B", *argv]
    if launcher:
        command = [sys.executable, "-B", str(worker), str(folder), "launcher", mode]
    with (tmp_path / "child.log").open("wb") as stream:
        return monitor_child(command, cwd=tmp_path, env=dict(os.environ), log=stream,
            metadata=folder, worker_argv=argv, timeout_seconds=timeout, poll_seconds=0.02)


@pytest.mark.parametrize("code", [0, 37, 259])
def test_preserves_real_exit_including_259(tmp_path, worker, code):
    result = run(tmp_path, worker, str(code))
    assert result["acknowledged"] and not result["timed_out"]
    assert result["launcher_exit_code"] == code
    assert len(result["processes"]) == 1 and result["processes"][0]["exit_code"] == code
    assert result["all_observed_processes_exited"] and not result["errors"]
    assert not result["terminated_pids"]


def test_launcher_and_worker_exit_are_observed_independently(tmp_path, worker):
    result = run(tmp_path, worker, "37", launcher=True)
    assert result["worker_pid"] != result["launcher_pid"]
    assert result["launcher_exit_code"] == 37
    assert len(result["processes"]) == 2
    assert all(v["exited"] and v["exit_code"] == 37 for v in result["processes"])
    assert not result["errors"] and not result["terminated_pids"]


def test_completed_marker_and_result_do_not_hide_a_hang(tmp_path, worker):
    result = run(tmp_path, worker, "marker_hang", launcher=True, timeout=2)
    assert (tmp_path / "metadata/result.json").exists()
    assert (tmp_path / "metadata/completed.json").exists()
    assert result["timed_out"] and result["all_observed_processes_exited"]
    assert result["worker_pid"] in result["terminated_pids"]
    assert not result["errors"]
    assert all(v["exit_code"] == 124 for v in result["processes"])


def test_timeout_cleans_observed_descendants_and_leaves_unrelated_child(tmp_path, worker):
    unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(60)"])
    try:
        result = run(tmp_path, worker, "nested_hang", launcher=True, timeout=2)
        nested = json.loads((tmp_path / "metadata/nested.json").read_text())["pid"]
        assert len(result["processes"]) == 3
        assert nested in result["terminated_pids"] and unrelated.poll() is None
        assert result["timed_out"] and result["all_observed_processes_exited"]
        assert not result["errors"]
    finally:
        unrelated.kill()
        unrelated.wait(timeout=10)
