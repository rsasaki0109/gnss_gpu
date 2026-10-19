"""Field threshold types, preserved Nav values and fallback partial state."""

from pathlib import Path
import sys

import numpy as np
import pytest
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_qcedit_batch_schema import (  # noqa: E402
    capture_input,
    capture_output,
    exact,
    make_type,
    restore,
)
from test_rbpf_qcedit_batch import fixture  # noqa: E402


@pytest.mark.parametrize("base_list", [False, True])
@pytest.mark.parametrize(
    "thresholds", [(30, 35), (30.0, 35), (-(2**53), 2**53), (-0.0, 35)]
)
def test_integer_metadata_thresholds_preserved(base_list, thresholds):
    a, o, args = fixture()
    a.nav.cnr_min, a.nav.cnr_min_gpy = thresholds
    if base_list:
        args["rr"] = a.nav.x[:3].tolist()
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    ra, rb = a.qcedit(o, **args), b.qcedit(ob, **ab)
    exact(capture_output(a, o, ra), capture_output(b, ob, rb))
    assert engine.fast_calls == 1 and engine.fallback_calls == 0
    for name in ("cnr_min", "cnr_min_gpy"):
        x, y = getattr(a.nav, name), getattr(b.nav, name)
        assert type(x) is type(y) and np.float64(x).tobytes() == np.float64(y).tobytes()


@pytest.mark.parametrize("value", [2**53 + 1, 10**400])
def test_unsafe_integer_conversion_original_partial_state(value):
    a, o, args = fixture()
    a.nav.cnr_min_gpy = value
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    outputs, errors = [], []
    for rtk, obs, ar in ((a, o, args), (b, ob, ab)):
        try:
            r = rtk.qcedit(obs, **ar)
        except (OverflowError, ValueError) as exc:
            errors.append((type(exc), str(exc)))
            r = np.empty(0, dtype=int)
        else:
            errors.append(None)
        outputs.append(capture_output(rtk, obs, r))
    assert errors[0] == errors[1]
    exact(*outputs)
    assert engine.fast_calls == 0 and engine.fallback_calls == 1
