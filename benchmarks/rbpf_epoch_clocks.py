"""Three clocks around complete, unchanged RTK calls for noise diagnosis.

These are instrumented A/A observations, never an optimization speed gate.
Process CPU can include other worker threads. Wall minus thread CPU includes
blocking and preemption; it does not identify the cause of either.
"""

import math
import threading
import time


class EpochClocks:
    """Keep clock samples outside estimator state and never filter epochs."""

    def __init__(self, wall=None, process=None, thread=None):
        self.wall = wall if wall is not None else time.perf_counter_ns
        self.process = process if process is not None else time.process_time_ns
        self.thread = thread if thread is not None else time.thread_time_ns
        self.records = []
        self.active = False

    def call(self, phase, function, *args, **kwargs):
        if self.active:
            raise RuntimeError("Epoch clock calls must be serial and nonrecursive")
        self.active = True
        owner = threading.get_ident()
        try:
            # Opposite stop order keeps each CPU interval within the wall span.
            wall_start = self.wall()
            process_start = self.process()
            thread_start = self.thread()
            completed = False
            try:
                result = function(*args, **kwargs)
                completed = True
                return result
            finally:
                thread_stop = self.thread()
                process_stop = self.process()
                wall_stop = self.wall()
                if threading.get_ident() != owner:
                    raise RuntimeError("Clock sample crossed a thread boundary")
                deltas = (
                    wall_stop - wall_start,
                    process_stop - process_start,
                    thread_stop - thread_start,
                )
                if any(v < 0 for v in deltas):
                    raise ValueError("A CPU or monotonic clock moved backwards")
                self.records.append(
                    dict(
                        epoch=len(self.records),
                        phase_before=phase,
                        wall_ns=deltas[0],
                        process_cpu_ns=deltas[1],
                        thread_cpu_ns=deltas[2],
                        completed=completed,
                    )
                )
        finally:
            self.active = False

    def instrument_type(self, original):
        function = original.process
        monitor = self

        class Clocked(original):
            def process(self_rtk, *args, **kwargs):
                phase = int(self_rtk.phase)
                return monitor.call(phase, function, self_rtk, *args, **kwargs)

        return Clocked


def summarize(records):
    """Aggregate integer CPU ticks before division; retain signed differences.

    Windows CPU samples may be zero or coarsely quantized for short calls.
    Per-epoch CPU/wall ratios and clipped/median CPU samples are not used.
    """
    if not records:
        raise ValueError("No epoch records")
    if [v["epoch"] for v in records] != list(range(len(records))):
        raise ValueError("Missing, duplicated or reordered epochs")
    if any(
        type(v["phase_before"]) is not int or v["phase_before"] not in (1, 2)
        for v in records
    ):
        raise ValueError("Unexpected RTK phase")
    if not all(v["completed"] is True for v in records):
        raise ValueError("An RTK call did not complete")
    output = {}
    for phase in (None, 1, 2):
        rows = [v for v in records if phase is None or v["phase_before"] == phase]
        if not rows:
            continue
        sums = {
            key: sum(v[key] for v in rows)
            for key in ("wall_ns", "process_cpu_ns", "thread_cpu_ns")
        }
        if any(
            type(v) is not int or v < 0
            for row in rows
            for v in (row["wall_ns"], row["process_cpu_ns"], row["thread_cpu_ns"])
        ):
            raise ValueError("Clock deltas must be nonnegative integers")
        wall = sums["wall_ns"]
        output["all" if phase is None else str(phase)] = dict(
            epochs=len(rows),
            **sums,
            wall_minus_thread_cpu_ns=wall - sums["thread_cpu_ns"],
            wall_minus_process_cpu_ns=wall - sums["process_cpu_ns"],
            process_minus_thread_cpu_ns=sums["process_cpu_ns"] - sums["thread_cpu_ns"],
            thread_cpu_over_wall=(sums["thread_cpu_ns"] / wall if wall else None),
            process_cpu_over_wall=(sums["process_cpu_ns"] / wall if wall else None),
            observed_positive_CPU_delta_gcd_ns={
                key: math.gcd(*(v[key] for v in rows if v[key] > 0))
                for key in ("process_cpu_ns", "thread_cpu_ns")
            },
            zero_CPU_epochs={
                key: sum(v[key] == 0 for v in rows)
                for key in ("process_cpu_ns", "thread_cpu_ns")
            },
        )
    return output
