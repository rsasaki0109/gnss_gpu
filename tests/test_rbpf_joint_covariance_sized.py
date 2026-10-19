"""Size dispatch retains original complete covariance-writer results."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_joint_covariance_cases import make_case, result  # noqa: E402
from rbpf_joint_covariance_sized import make_type  # noqa: E402


@pytest.mark.parametrize("dim", [3, 6])
@pytest.mark.parametrize("count", [0, 1, 4, 15, 16, 25])
def test_size_dispatch_complete_writer_bits(dim, count):
    cls, original, values, key_x, cp, _, _ = make_case(dim, count)
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
    left, right = result(original), result(candidate)
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert (
                left[key].dtype == right[key].dtype
                and left[key].shape == right[key].shape
                and left[key].tobytes() == right[key].tobytes()
            ), key
        else:
            assert left[key] == right[key]
    assert engine.small_literal_calls == int(0 < count < 16)
    assert engine.full_matrix_calls == int(count >= 16)
    assert engine.fallback_calls == 0


@pytest.mark.parametrize("count", [4, 16])
def test_custom_index_callback_uses_original_order(count, monkeypatch):
    cls, rtk, _, key_x, _, jm, active = make_case(6, count)
    _, engine = make_type(cls)
    _, _, _, _, scope, literal = engine.methods["_write_back_tc"]
    calls = []
    ib = cls.IB

    def custom(self, s, f, na):
        calls.append((s, f, na))
        return ib(self, s, f, na)

    monkeypatch.setattr(cls, "IB", custom)
    literal(rtk, active, jm)
    expected, trace = rtk.nav.P.tobytes(), list(calls)
    calls.clear()
    engine.copy_pairs(rtk, active, jm, key_x, 6, scope, literal)
    assert rtk.nav.P.tobytes() == expected and calls == trace
    assert engine.full_matrix_calls == 0
    assert engine.small_literal_calls == int(count < 16)
    assert engine.fallback_calls == int(count >= 16)
