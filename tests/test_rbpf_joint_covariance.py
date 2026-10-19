"""Actual GTSAM writer bits, upper-triangle semantics and literal fallback."""

from copy import deepcopy
from pathlib import Path
import sys

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import rbpf_joint_covariance as selected  # noqa: E402
from rbpf_joint_covariance_cases import make_case, result  # noqa: E402


def exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert (
                a[key].dtype == b[key].dtype
                and a[key].shape == b[key].shape
                and a[key].tobytes() == b[key].tobytes()
            ), key
        elif a[key] is None:
            assert b[key] is None
        else:
            assert a[key] == b[key]


@pytest.mark.parametrize("dim", [3, 6])
@pytest.mark.parametrize("count", [0, 1, 2, 8, 25])
@pytest.mark.parametrize("options", [(False, False), (True, True)])
def test_complete_literal_covariance_writers(dim, count, options):
    held, subset = options
    cls, original, values, key_x, cp, _, _ = make_case(
        dim, count, 20261017 + dim + count, held=held, subset=subset
    )
    chosen, engine = selected.make_type(cls)
    candidate = chosen()
    candidate.__dict__.update(
        deepcopy(
            {k: v for k, v in vars(original).items() if k not in ("isam", "ecef_T_nav")}
        )
    )
    candidate.isam, candidate.ecef_T_nav = original.isam, original.ecef_T_nav
    name = "_write_back" if dim == 3 else "_write_back_tc"
    getattr(original, name)(values, key_x, cp)
    getattr(candidate, name)(values, key_x, cp)
    exact(result(original), result(candidate))
    assert all(p["restored_AST_equal"] for p in engine.proofs)
    assert engine.fallback_calls == 0
    if count >= 8:
        assert engine.full_matrix_calls == 1 and engine.copied_pairs > 0


@pytest.mark.parametrize("seed", range(6))
def test_direct_pair_copy_actual_fullmatrix_layout(seed):
    cls, original, _, key_x, _, jm, active = make_case(6, 18, 20261017 + seed)
    chosen, engine = selected.make_type(cls)
    candidate = chosen()
    candidate.nav = deepcopy(original.nav)
    fn, _, _, _, scope, literal = engine.methods["_write_back_tc"]
    literal(original, active, jm)
    engine.copy_pairs(candidate, active, jm, key_x, 6, scope, literal)
    assert original.nav.P.tobytes() == candidate.nav.P.tobytes()
    assert (
        engine.full_matrix_calls == 1
        and engine.copied_pairs == 153
        and engine.fallback_calls == 0
    )


def test_asymmetric_matrix_signed_zero_and_nan_payload_are_copied_without_arithmetic():
    bits = np.arange(64, dtype=np.uint64).reshape(8, 8) + np.uint64(0x3FF0000000000000)
    bits[2, 1], bits[1, 2] = (
        np.uint64(0x8000000000000000),
        np.uint64(0x7FF8000000001234),
    )
    bits[1, 5] = np.uint64(0x7FF8000000005678)
    dense = bits.view(np.float64)
    offsets, indices = np.array([2, 1, 5, 0]), np.array([7, 3, 9, 2])
    expected, actual = np.zeros((12, 12)), np.zeros((12, 12))
    for i in range(4):
        for j in range(i + 1, 4):
            value = dense[offsets[i], offsets[j]]
            expected[indices[i], indices[j]] = value
            expected[indices[j], indices[i]] = value
    with np.errstate(all="raise"):
        selected.scatter_pairs(actual, dense, indices, offsets)
    assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize(
    "kind",
    [
        "custom_IB",
        "duplicate_index",
        "duplicate_key",
        "float32",
        "strided",
        "read_only",
        "np_binding",
        "joint_binding",
    ],
)
def test_literal_fallback_preserves_calls_errors_and_arrays(kind, monkeypatch):
    cls, rtk, _, key_x, _, jm, active = make_case(6, 5)
    _, engine = selected.make_type(cls)
    fn, _, _, _, scope, literal = engine.methods["_write_back_tc"]
    trace = []
    original_ib = cls.IB
    if kind == "custom_IB":

        def counted(self, s, f, na):
            trace.append((s, f, na))
            return original_ib(self, s, f, na)

        monkeypatch.setattr(cls, "IB", counted)
    elif kind == "duplicate_index":
        active[1] = (active[0][0], active[0][1], active[1][2])
    elif kind == "duplicate_key":
        active[1] = (active[1][0], active[1][1], active[0][2])
    elif kind == "float32":
        rtk.nav.P = rtk.nav.P.astype(np.float32)
    elif kind == "strided":
        rtk.nav.P = np.repeat(np.repeat(rtk.nav.P, 2, axis=0), 2, axis=1)[::2, ::2]
    elif kind == "read_only":
        rtk.nav.P.flags.writeable = False
    elif kind == "np_binding":
        monkeypatch.setattr(
            np,
            "triu_indices",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("No new helper call")),
        )
    elif kind == "joint_binding":
        at = gtsam.JointMarginal.at

        def counted(self, a, b):
            trace.append((a, b))
            return at(self, a, b)

        monkeypatch.setattr(gtsam.JointMarginal, "at", counted)
    initial = rtk.nav.P.copy()
    errors = []
    try:
        literal(rtk, active, jm)
    except ValueError as error:
        errors.append(str(error))
    expected, events = rtk.nav.P.copy(), list(trace)
    rtk.nav.P = initial
    if kind == "read_only":
        rtk.nav.P.flags.writeable = False
    if kind == "strided":
        rtk.nav.P = np.repeat(np.repeat(initial, 2, axis=0), 2, axis=1)[::2, ::2]
    trace.clear()
    try:
        engine.copy_pairs(rtk, active, jm, key_x, 6, scope, literal)
    except ValueError as error:
        errors.append(str(error))
    assert trace == events and rtk.nav.P.tobytes() == expected.tobytes()
    assert engine.fallback_calls == 1 and engine.fast_calls == 0
    assert not errors or len(errors) == 2 and errors[0] == errors[1]


def test_changed_original_code_falls_back_once(monkeypatch):
    cls, rtk, values, key_x, cp, _, _ = make_case()
    chosen, engine = selected.make_type(cls)
    candidate = chosen()
    candidate.__dict__.update(vars(rtk))

    def changed(self, estimate, key, cp_amb=None):
        self.changed_calls = getattr(self, "changed_calls", 0) + 1
        return "new original"

    monkeypatch.setattr(cls._write_back_tc, "__code__", changed.__code__)
    assert candidate._write_back_tc(values, key_x, cp) == "new original"
    assert candidate.changed_calls == 1 and engine.calls == 0
