"""Actual int32 satellite indices retain original covariance writer bits."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_joint_covariance_cases import make_case, result  # noqa: E402
from rbpf_joint_covariance_field_schema import make_type  # noqa: E402


@pytest.mark.parametrize("dim", [3, 6])
@pytest.mark.parametrize("count", [4, 16, 25])
@pytest.mark.parametrize("high_sat", [False, True])
def test_real_int32_schema_complete_writer(dim, count, high_sat):
    cls, original, values, key_x, cp, _, _ = make_case(dim, count)
    original.amb_keys = {
        (np.int32(s + (190 if high_sat else 0)), f): k
        for (s, f), k in original.amb_keys.items()
    }
    chosen, engine = make_type(cls)
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
    a, b = result(original), result(candidate)
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert (
                a[key].dtype == b[key].dtype
                and a[key].shape == b[key].shape
                and a[key].tobytes() == b[key].tobytes()
            ), key
        else:
            assert a[key] == b[key]
    assert engine.field_schema_calls == engine.full_matrix_calls == int(count >= 16)
    assert engine.fallback_calls == 0


@pytest.mark.parametrize("kind", ["int64", "unsafe_na"])
def test_other_schema_stays_literal(kind):
    cls, rtk, _, key_x, _, jm, active = make_case(6, 16)
    _, engine = make_type(cls)
    _, _, _, _, scope, literal = engine.methods["_write_back_tc"]
    active = [
        (np.int64(s) if kind == "int64" else np.int32(s), f, k) for s, f, k in active
    ]
    if kind == "unsafe_na":
        rtk.nav.na = -1
    literal(rtk, active, jm)
    expected = rtk.nav.P.tobytes()
    engine.copy_pairs(rtk, active, jm, key_x, 6, scope, literal)
    assert expected == rtk.nav.P.tobytes()
    assert engine.fallback_calls == 1 and engine.full_matrix_calls == 0
