"""Retain the initial negative control and verify conversion-error partial state."""

from pathlib import Path
import sys
from types import FunctionType

import numpy as np
import pytest
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import test_rbpf_ambiguity_initx as previous  # noqa: E402
from rbpf_ambiguity_initx_checked import make_type  # noqa: E402

# Reuse every unchanged test body/parameter from the initial version, with
# only its explicit implementation factory substituted in a cloned scope.
for _name, _fn in vars(previous).items():
    if _name.startswith("test_") and isinstance(_fn, FunctionType):
        _clone = FunctionType(
            _fn.__code__,
            dict(_fn.__globals__, make_type=make_type),
            _fn.__name__,
            _fn.__defaults__,
            _fn.__closure__,
        )
        _clone.__dict__.update(_fn.__dict__)
        _clone.__module__ = __name__
        globals()[_name] = _clone


def test_initial_overflow_negative_control_is_retained():
    cls, _ = previous.make_type(rtkpos)
    a, b = previous.instance(rtkpos), previous.instance(cls)
    for value in (a, b):
        with pytest.raises(OverflowError):
            value.initx(3.25, 10**400, 8)
    assert a.nav.P.tobytes() != b.nav.P.tobytes()


@pytest.mark.parametrize(
    "x0,v0",
    [
        (3.25, 10**400),
        (10**400, 100.0),
        (3.25, -(10**400)),
        (3.25, np.longdouble(10)),
        (3.25, np.float32(100)),
    ],
)
def test_unusual_scalars_use_original_and_preserve_partial_arrays(x0, v0):
    cls, engine = make_type(rtkpos)
    a, b = previous.instance(rtkpos), previous.instance(cls)
    errors = []
    for value in (a, b):
        try:
            value.initx(x0, v0, 8)
        except OverflowError as exc:
            errors.append(type(exc))
        else:
            errors.append(None)
    assert errors[0] is errors[1]
    assert a.nav.x.tobytes() == b.nav.x.tobytes()
    assert a.nav.P.tobytes() == b.nav.P.tobytes()
    assert engine.calls == engine.fallback_calls == 1 and engine.fast_calls == 0


def test_explicit_nan_payload_bits():
    cls, engine = make_type(rtkpos)
    payload = np.array([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    a, b = previous.instance(rtkpos), previous.instance(cls)
    a.initx(payload, payload, 8)
    b.initx(payload, payload, 8)
    assert a.nav.x.tobytes() == b.nav.x.tobytes()
    assert a.nav.P.tobytes() == b.nav.P.tobytes() and engine.fast_calls == 1
