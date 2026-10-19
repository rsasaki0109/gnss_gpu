"""D236 with bit-identical IRLS state reuse in an explicit likelihood DLL.

The same two-update rule remains. Reuse occurs only when EVERY Aw/bw bit
matches the already solved system. No convergence tolerance or model change.
"""

import ctypes

import numpy as np

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_batch import BatchScorer

_SCORE = BatchScorer.score
_SCORE_CODE = _SCORE.__code__


class IdentityBatchScorer(BatchScorer):
    """Unchanged pack/fallback API with explicit completed-batch counters.

    The DLL returns four extra bytes per hypothesis in its existing staging
    buffer. Reading them makes no extra CUDA call. All wrapper/query/allocation
    overhead is included in the complete-step benchmark.
    """

    def __init__(self, dll, capacity=512):
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_identity_counts.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ]
            self.dll.rbpf_identity_counts.restype = ctypes.c_int
        except BaseException:
            self.close()
            raise
        self._identity = dict(
            native_batches=0,
            native_hypotheses=0,
            identical_hypotheses=0,
            skipped_normal_solves=0,
        )
        self.last_skipped = np.empty(0, dtype=np.int32)

    def counts(self):
        return dict(self._identity)

    def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
        if BatchScorer.score is not _SCORE or _SCORE.__code__ is not _SCORE_CODE:
            raise ValueError("Original batch scorer changed after binding")
        self.last_skipped = np.empty(0, dtype=np.int32)
        output, details = super().score(hypotheses, problem, oracle, mode, diagnostics)
        # Early empty/unsupported/error/fallback paths contain no completed
        # native statuses. Never inspect counters left by a previous batch.
        if "statuses" in details:
            skipped = np.empty(len(details["statuses"]), dtype=np.int32)
            code = self.dll.rbpf_identity_counts(self.handle, len(skipped), skipped)
            if code or np.any(skipped < 0) or np.any(skipped > 2):
                raise RuntimeError("Invalid completed-batch IRLS identity counters")
            if oracle.cfg.huber_c <= 0 and np.any(skipped):
                raise RuntimeError("IRLS identity shortcut used without IRLS")
            self.last_skipped = skipped
            self._identity["native_batches"] += 1
            self._identity["native_hypotheses"] += len(skipped)
            self._identity["identical_hypotheses"] += int(np.count_nonzero(skipped))
            self._identity["skipped_normal_solves"] += int(skipped.sum(dtype=np.int64))
        return output, details


class IRLSIdentityStep(NativeFeedbackMassStep):
    """Original D236 caller; only the explicitly supplied scorer DLL changes.

    Particle/candidate/prior/order, canonical calls, all existing metadata,
    probabilities/RNG and conditional/FIX/feedback/graph code remain. New
    counters are outside runtime state. No prior rejected candidate is added.
    """

    def __init__(self, runtime, scorer=None, mode="native_cpu", **extensions):
        if mode != "scalar" and type(scorer) is not IdentityBatchScorer:
            raise ValueError(
                "Native IRLS identity mode requires its explicit owning scorer"
            )
        super().__init__(runtime, scorer, mode, **extensions)

    def __call__(self, rtk, ctx):
        before = None if self.scorer is None else self.scorer.counts()
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["irls_identity"] = (
                    dict(
                        native_batches=0,
                        native_hypotheses=0,
                        identical_hypotheses=0,
                        skipped_normal_solves=0,
                    )
                    if before is None
                    else {k: v - before[k] for k, v in self.scorer.counts().items()}
                )
