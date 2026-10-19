"""Actual base-position representation, including scalar/callback fallbacks."""

from pathlib import Path
import sys

import numpy as np
import pytest
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_qcedit_batch_list import (  # noqa: E402
    capture_input,
    capture_output,
    exact,
    make_type,
    restore,
)
from test_rbpf_qcedit_batch import fixture  # noqa: E402


@pytest.mark.parametrize("nf", [1, 2])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
def test_base_list_eight_epochs_all_bits(nf, seed):
    a, oa, aa = fixture(nf, seed)
    aa["rr"] = a.nav.x[:3].tolist()
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, oa, **aa))
    for epoch in range(8):
        for obs in (oa, ob):
            obs.L[:, 0] += float(epoch) * 0.0625
        ra, rb = a.qcedit(oa, **aa), b.qcedit(ob, **ab)
        exact(capture_output(a, oa, ra), capture_output(b, ob, rb))
    assert engine.fast_calls == 8 and engine.fallback_calls == 0


@pytest.mark.parametrize(
    "kind",
    ["custom_list", "integers", "wrong_size", "huge_integer", "readonly", "strict"],
)
def test_original_errors_and_partial_state(kind):
    a, o, args = fixture()
    args["rr"] = a.nav.x[:3].tolist()
    if kind == "custom_list":

        class Custom(list):
            pass

        args["rr"] = Custom(args["rr"])
    elif kind == "integers":
        args["rr"] = [int(v) for v in args["rr"]]
    elif kind == "wrong_size":
        args["rr"] = args["rr"][:2]
    elif kind == "huge_integer":
        args["rr"][0] = 10**400
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    if kind == "readonly":
        a.nav.el.flags.writeable = b.nav.el.flags.writeable = False
    results, errors = [], []
    with np.errstate(under="raise" if kind == "strict" else "ignore"):
        for rtk, obs, ar in ((a, o, args), (b, ob, ab)):
            try:
                result = rtk.qcedit(obs, **ar)
            except (ValueError, IndexError, TypeError, OverflowError) as exc:
                errors.append((type(exc), str(exc)))
                result = np.empty(0, dtype=int)
            else:
                errors.append(None)
            results.append(capture_output(rtk, obs, result))
    assert errors[0] == errors[1]
    exact(*results)
    assert engine.fast_calls == 0 and engine.fallback_calls == 1
