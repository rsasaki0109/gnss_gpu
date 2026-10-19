"""Exit evidence for new independent-process RTK clock measurements.

The launcher and worker must both exit normally. Parent polling can also see
one synchronous OS-version shell query; if observed, its command, ancestry,
creation order, and exit0 are required. This is not process containment.
Parent-monitor durations and ready/completed metadata are not speed metrics.
"""

from rbpf_lifecycle_validation import normal_field_exit


def normal_numeric_exit(event, version_command):
    if len(event["processes"]) == 3:
        return normal_field_exit(event, version_command)
    processes = {v["pid"]: v for v in event["processes"]}
    if len(event["processes"]) != 2 or len(processes) != 2:
        return False
    launcher = processes.get(event["launcher_pid"])
    worker = processes.get(event["worker_pid"])
    if not launcher or not worker or launcher is worker:
        return False
    return bool(event["acknowledged"] and event["all_observed_processes_exited"] and
        not event["timed_out"] and not event["ready_timed_out"] and
        not event["errors"] and not event["terminated_pids"] and
        event["launcher_exit_code"] == launcher["exit_code"] == 0 and
        launcher["command"] == event["command"] and
        worker["parent_pid"] == launcher["pid"] and
        launcher["creation_filetime"] <= worker["creation_filetime"] and
        all(v["exited"] and v["exit_code"] == 0 for v in processes.values()))
