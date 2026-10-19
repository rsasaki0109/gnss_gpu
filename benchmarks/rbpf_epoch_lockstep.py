"""Serialize two independent RTK calls at each matching epoch.

The protocol and its blocking I/O are outside all three estimator clocks.
After reporting a sample, a child must wait for the pair's advance barrier;
its next-epoch preparation cannot run during the other child's timed call.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

from rbpf_epoch_clocks import EpochClocks

PREFIX = "@@RBPF_EPOCH_LOCKSTEP@@"
LABELS = ("A", "AA")


class ProtocolError(RuntimeError):
    pass


class PipeEndpoint:
    """A caller-owned child; drain stdout continuously and bound all waits."""

    def __init__(self, process, log, *, timeout=180):
        self.process, self.log, self.timeout = process, log, timeout
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
                self.lines.put(line)
        except BaseException as error:
            self.lines.put(error)
        finally:
            self.lines.put(None)

    def _line(self, deadline):
        try:
            line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as error:
            raise ProtocolError("Timed out waiting for the owned child") from error
        if isinstance(line, BaseException):
            raise ProtocolError("Child output reader failed") from line
        return line

    def event(self):
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._line(deadline)
            if line is None:
                raise ProtocolError("Child closed stdout before the expected event")
            if line.startswith(PREFIX):
                try:
                    return json.loads(line[len(PREFIX) :])
                except ValueError as error:
                    raise ProtocolError("Malformed child protocol JSON") from error

    def command(self, text):
        try:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise ProtocolError("Cannot send to the owned child") from error

    def finish(self):
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._line(deadline)
            if line is None:
                break
            if line.startswith(PREFIX):
                raise ProtocolError("Unexpected extra event after the final epoch")
        return self.process.wait(timeout=self.timeout)

    def close(self):
        # Only this endpoint's explicit Popen child is affected on failure.
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        if self.reader.is_alive():
            raise ProtocolError("Owned child output reader did not close")
        self.process.stdin.close()
        self.process.stdout.close()


class LockstepClocks(EpochClocks):
    def __init__(self, *args, incoming=None, outgoing=None, label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.incoming = sys.stdin if incoming is None else incoming
        self.outgoing = sys.stdout if outgoing is None else outgoing
        self.label = (
            os.environ.get("GNSS_GPU_LOCKSTEP_LABEL") if label is None else label
        )
        if self.label not in LABELS:
            raise ProtocolError("Explicit independent child label required")
        self.waiting = False

    def emit(self, event, epoch, phase, **values):
        payload = dict(
            event=event, label=self.label, epoch=epoch, phase=phase, **values
        )
        self.outgoing.write(PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
        self.outgoing.flush()

    def expect(self, command, epoch):
        line = self.incoming.readline()
        if not line:
            raise ProtocolError("Parent closed the command pipe")
        if line.strip() != f"{command} {epoch}":
            raise ProtocolError("Unexpected parent command: " + line.strip())

    def call(self, phase, function, *args, **kwargs):
        if self.waiting or self.active:
            raise ProtocolError("Lockstep process calls must not reenter")
        self.waiting = True
        epoch = len(self.records)
        try:
            self.emit("ready", epoch, phase)
            self.expect("go", epoch)
            try:
                result = super().call(phase, function, *args, **kwargs)
            except BaseException as error:
                self.emit("failed", epoch, phase, error_type=type(error).__name__)
                raise
            self.emit("done", epoch, phase, sample=dict(self.records[-1]))
            self.expect("advance", epoch)
            return result
        finally:
            self.waiting = False


def validate_event(event, kind, label, epoch, phase=None):
    if not isinstance(event, dict):
        raise ProtocolError("Expected a structured child event")
    if (
        event.get("event") != kind
        or event.get("label") != label
        or type(event.get("epoch")) is not int
        or event["epoch"] != epoch
        or type(event.get("phase")) is not int
        or event["phase"] not in (1, 2)
        or (phase is not None and event["phase"] != phase)
    ):
        raise ProtocolError("Unexpected child event: " + repr(event))
    if kind == "done":
        sample = event.get("sample")
        if (
            type(sample) is not dict
            or type(sample.get("epoch")) is not int
            or sample["epoch"] != epoch
            or type(sample.get("phase_before")) is not int
            or sample["phase_before"] != event["phase"]
            or sample.get("completed") is not True
            or any(
                type(sample.get(k)) is not int or sample[k] < 0
                for k in ("wall_ns", "process_cpu_ns", "thread_cpu_ns")
            )
        ):
            raise ProtocolError("Invalid completed epoch sample")
    return event


def coordinate(endpoints, epochs, *, pair_index=0, progress=None):
    """Endpoints expose event() and command(text), with owned I/O timeouts.

    Both children reach ready before either starts. Both finish before either
    receives advance. The first child alternates every epoch and every pair.
    No estimator RNG is consumed to select measurement order.
    """
    if set(endpoints) != set(LABELS) or type(epochs) is not int or epochs <= 0:
        raise ProtocolError("Two children and a positive epoch count required")
    records = []
    for epoch in range(epochs):
        ready = {
            label: validate_event(endpoints[label].event(), "ready", label, epoch)
            for label in LABELS
        }
        phase = ready["A"]["phase"]
        if ready["AA"]["phase"] != phase:
            raise ProtocolError("Independent RTK phases diverged")
        order = LABELS if (epoch + pair_index) % 2 == 0 else LABELS[::-1]
        done = {}
        for label in order:
            endpoints[label].command(f"go {epoch}")
            done[label] = validate_event(
                endpoints[label].event(), "done", label, epoch, phase
            )["sample"]
        for label in LABELS:
            endpoints[label].command(f"advance {epoch}")
        record = dict(epoch=epoch, phase=phase, order=list(order), samples=done)
        records.append(record)
        if progress is not None:
            progress(record)
    return records
