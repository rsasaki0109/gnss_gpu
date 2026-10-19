"""Explicit IRLS on/off control and lossless actual likelihood input replay.

This component harness does not install a runtime/default or replay a graph.
"""

from copy import deepcopy
import ctypes
import gzip
import json
from types import FunctionType, SimpleNamespace

from bench_rbpf_move_field import decode, encode
from rbpf_irls_identity import IdentityBatchScorer
from rbpf_likelihood_cases import make_oracle
from rbpf_native_canonical import CanonicalEngine, _CanonicalView
from rbpf_parent_score import ParentScoreEngine

PROBLEM_FIELDS = (
    "pairs",
    "y",
    "Qb",
    "pos",
    "Ppos",
    "rows_J",
    "rows_v0",
    "rows_sig",
    "rows_iscp",
    "rows_pair",
    "rows_lam",
    "rows_bmeas",
)


class ControlledIdentityScorer(IdentityBatchScorer):
    """One binary and interface, immutable explicit control per workspace.

    Off uses the original reweight operation and all original solves. On uses
    the audited binary64 identity shortcut. Both copy/query the same counters.
    Construction/control is outside timing; score includes the entire wrapper.
    """

    def __init__(self, dll, capacity=512, *, enabled):
        if type(enabled) is not bool:
            raise TypeError("An explicit bool control is required")
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_identity_enable.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_identity_enable.restype = ctypes.c_int
            if self.dll.rbpf_identity_enable(self.handle, int(enabled)):
                raise RuntimeError("IRLS control configuration failed")
        except BaseException:
            self.close()
            raise
        self.enabled = enabled


def snapshot(hypotheses, problem, oracle):
    """Copy values only; never call canonical/DD/cluster/scoring callbacks."""
    actual = oracle._cluster_state.__self__
    return deepcopy(
        dict(
            hypotheses=hypotheses,
            problem=SimpleNamespace(**{k: getattr(problem, k) for k in PROBLEM_FIELDS}),
            cfg=vars(oracle.cfg),
            cluster_states=actual.cluster_states,
        )
    )


def oracle_from_snapshot(saved, extension, parent_extension):
    """Original numeric methods and D236 native canonical/parent view bindings."""
    actual = make_oracle()
    actual.cfg = SimpleNamespace(**deepcopy(saved["cfg"]))
    actual.cluster_states = deepcopy(saved["cluster_states"])
    canonical = CanonicalEngine(extension, actual.basin_id)
    parent = ParentScoreEngine(parent_extension, actual._basin_parent_score)
    assert canonical.literal and parent.literal
    scope = dict(actual._cluster_state.__func__.__globals__)
    scope["_canon_basin_id"] = canonical.canonical
    functions = {}
    for name in ("_pair_dd", "_cluster_state", "_loglik"):
        fn = getattr(actual, name).__func__
        functions[name] = FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__)

    def parent_score(_view, child, previous):
        return parent.score(child, previous)

    functions["_basin_parent_score"] = parent_score
    view = _CanonicalView(actual, functions)
    return SimpleNamespace(
        cfg=actual.cfg,
        basin_id=canonical.canonical,
        _pair_dd=view._pair_dd,
        _cluster_state=view._cluster_state,
        _loglik=view._loglik,
    )


def load_corpus(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        records = [decode(json.loads(line)) for line in stream]
    assert records and [r["index"] for r in records] == list(range(len(records)))
    return records


def write_record(stream, index, saved, result):
    stream.write(
        json.dumps(
            encode(dict(index=index, inputs=saved, result=result)),
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()
