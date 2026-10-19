"""Coarse explicit RBPF phase spans on the unchanged D236 controller.

These in-process wall spans locate candidate bottlenecks; they are not paired
speed measurements. Inclusive times overlap. Exclusive spans partition the
root interval, including tracing overhead and scheduling delays.
"""

import time

from rbpf_feedback_mass import NativeFeedbackMassStep


class _Spans:
    def __init__(self):
        self.stack, self.values = [], {}

    def start(self, name):
        frame = [name, time.perf_counter_ns(), 0]
        self.stack.append(frame)
        return frame

    def stop(self, frame):
        end = time.perf_counter_ns()
        assert self.stack.pop() is frame
        elapsed = end - frame[1]
        if self.stack:
            self.stack[-1][2] += elapsed
        value = self.values.setdefault(
            frame[0], dict(calls=0, inclusive_ns=0, exclusive_ns=0)
        )
        value["calls"] += 1
        value["inclusive_ns"] += elapsed
        value["exclusive_ns"] += elapsed - frame[2]


class PhaseProfileStep(NativeFeedbackMassStep):
    """Wrap only this caller's method view; retain all original state/math.

    No sys.setprofile, global or runtime-method replacement. A new recorder
    is owned by each step, and only aggregate integer spans survive. Nested
    spans are subtracted once from their parent. Exceptions still propagate;
    the recorder stack is unwound in finally blocks. Strict NumPy policy
    still uses the inherited literal whole-step fallback.
    """

    PHASES = (
        "_extract",
        "_lambda_topk",
        "_maintenance",
        "_ddpr_offset_impl",
        "_spawn_candidates",
        "_extend",
        "_shrink_worst",
        "_loglik",
        "_conditional_mean",
        "_ls_offset",
        "_feedback",
        "_maybe_resample",
    )

    def __init__(self, runtime, scorer=None, mode="native_cpu", **extensions):
        super().__init__(runtime, scorer, mode, **extensions)
        self._spans = None
        self.last_profile = None
        for name in self.PHASES:
            # Unit fixtures may replace external extraction/search inputs.
            if name not in self._functions:
                continue
            original = self._functions[name]
            self._functions[name] = self._wrap(name, original)

    def _wrap(self, name, original):
        def measured(view, *args, **kwargs):
            frame = self._spans.start(name)
            try:
                return original(view, *args, **kwargs)
            finally:
                self._spans.stop(frame)

        return measured

    def _moves(self, *args, **kwargs):
        frame = self._spans.start("_moves")
        try:
            return super()._moves(*args, **kwargs)
        finally:
            self._spans.stop(frame)

    def __call__(self, rtk, ctx):
        if self._spans is not None:
            raise RuntimeError("One filter cannot run concurrent profiled steps")
        spans = self._spans = _Spans()
        root = spans.start("step")
        try:
            return super().__call__(rtk, ctx)
        finally:
            spans.stop(root)
            self._spans = None
            assert not spans.stack
            assert (
                sum(v["exclusive_ns"] for v in spans.values.values())
                == spans.values["step"]["inclusive_ns"]
            )
            self.last_profile = spans.values
