"""Validate the observed launcher, numeric worker, and OS-version query.

This checks saved evidence only. Every observed process must have signalled
exit0; an extra process is accepted only for the explicitly supplied command
with the numeric worker as its recorded parent. No process is filtered out.
"""


def normal_field_exit(event, version_command):
    processes = {v["pid"]: v for v in event["processes"]}
    if len(processes) != len(event["processes"]) or len(processes) != 3:
        return False
    launcher = processes.get(event["launcher_pid"])
    worker = processes.get(event["worker_pid"])
    if not launcher or not worker or launcher is worker:
        return False
    extra = [v for v in processes.values() if v is not launcher and v is not worker]
    return bool(event["acknowledged"] and event["all_observed_processes_exited"] and
        not event["timed_out"] and not event["ready_timed_out"] and
        not event["errors"] and not event["terminated_pids"] and
        event["launcher_exit_code"] == launcher["exit_code"] == 0 and
        launcher["command"] == event["command"] and
        worker["parent_pid"] == launcher["pid"] and
        extra[0]["parent_pid"] == worker["pid"] and extra[0]["command"] == version_command and
        launcher["creation_filetime"] <= worker["creation_filetime"] <= extra[0]["creation_filetime"] and
        all(v["exited"] and v["exit_code"] == 0 for v in processes.values()))
