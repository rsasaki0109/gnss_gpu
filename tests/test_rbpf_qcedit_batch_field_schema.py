"""Actual three-band inputs must use the new loop and match all state bits."""

from pathlib import Path
import gzip
import hashlib
import pickle
import sys

import numpy as np
import pytest
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_qcedit_batch_field_schema import (  # noqa: E402
    GUARD_PROOF,
    capture_input,
    capture_output,
    exact,
    make_type,
    restore,
)
from test_rbpf_qcedit_batch import fixture  # noqa: E402


@pytest.mark.parametrize("nf", [1, 2, 3])
@pytest.mark.parametrize("base_list", [False, True])
def test_cached_signal_strings_and_integer_thresholds(nf, base_list):
    a, o, args = fixture(nf)
    a.nav.cnr_min, a.nav.cnr_min_gpy = 30, 35
    for signals in o.sig.values():
        for values in signals.values():
            for signal in values:
                signal.str()
    if base_list:
        args["rr"] = a.nav.x[:3].tolist()
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    for epoch in range(8):
        for obs in (o, ob):
            obs.L[:, 0] += float(epoch) * 0.03125
        ra, rb = a.qcedit(o, **args), b.qcedit(ob, **ab)
        exact(capture_output(a, o, ra), capture_output(b, ob, rb))
    assert engine.fast_calls == 8 and engine.fallback_calls == 0
    assert GUARD_PROOF["undo_AST_equal"] and not GUARD_PROOF["numerical_body_changed"]


@pytest.mark.parametrize("kind", ["extra_callback", "custom_cache", "huge_integer"])
def test_unsafe_metadata_uses_original(kind):
    a, o, args = fixture()
    signal = next(iter(o.sig.values()))[next(iter(next(iter(o.sig.values()))))][0]
    if kind == "extra_callback":
        signal.callback = lambda: None
    elif kind == "custom_cache":
        signal._str_cache = object()
    else:
        a.nav.cnr_min_gpy = 10**400
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    outputs, errors = [], []
    for rtk, obs, ar in ((a, o, args), (b, ob, ab)):
        try:
            r = rtk.qcedit(obs, **ar)
        except OverflowError as exc:
            errors.append(type(exc))
            r = np.empty(0, dtype=int)
        else:
            errors.append(None)
        outputs.append(capture_output(rtk, obs, r))
    assert errors[0] == errors[1]
    exact(*outputs)
    assert engine.fast_calls == 0 and engine.fallback_calls == 1


@pytest.mark.parametrize(
    "ci,digest",
    [
        (0, "24b832d8a221fd7234c27a501c3c1559e4e7f149ddf83b771e5ea8876c6b7e13"),
        (1, "f6063d54b75668ab090858c207d3ffc689bc88b1f983c84f6f9c736a7988df09"),
    ],
)
def test_every_saved_actual_qc_call_uses_new_loop(ci, digest):
    path = (
        ROOT
        / f"dist/rbpf_qcedit_batch_capture_20261017/r3_seed0_qc_reference_c{ci}.qc.pkl.gz"
    )
    if not path.exists():
        pytest.skip("Explicit local field capture is not available")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    cls, engine = make_type(rtkpos)
    count = 0
    with gzip.open(path, "rb") as stream:
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            rtk, obs, args = restore(cls, record["input"])
            result = rtk.qcedit(obs, **args)
            exact(capture_output(rtk, obs, result), record["output"])
            count += 1
    assert count == engine.fast_calls == 400 and engine.fallback_calls == 0
