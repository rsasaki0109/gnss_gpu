"""Regression for fresh tuple array shapes versus stored JSON list shapes."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_inprocess_pair_checked_field import serialized_encode, without_rtk  # noqa: E402
from bench_rbpf_move_field import encode  # noqa: E402


def test_fresh_array_shape_container_difference_preserves_exact_bits():
    value = np.array([0.0, -0.0, np.nan, np.inf, -np.inf], dtype=np.float64).reshape(1, 5)
    fresh = encode(value)
    stored = json.loads(json.dumps(fresh))
    assert fresh != stored and fresh["shape"] == (1, 5) and stored["shape"] == [1, 5]
    fixed = serialized_encode(value)
    assert fixed == stored and fixed["array"] == value.tobytes().hex()
    assert fixed["dtype"] == value.dtype.str and fixed["shape"] == [1, 5]


def test_nested_tags_dict_order_tuple_keys_and_large_integer_preserved():
    value = {(3, 1): [np.arange(6, dtype=np.int64).reshape(2, 3), -0.0, 2**100],
        "rtk": SimpleNamespace(nav=SimpleNamespace(P=np.eye(2)))}
    fixed = serialized_encode(value)
    assert fixed == json.loads(json.dumps(encode(value)))
    assert fixed["mapping"][0][0] == {"tuple": [3, 1]}
    assert fixed["mapping"][0][1][1] == {"float64": "-0x0.0p+0"}
    assert fixed["mapping"][0][1][2] == 2**100
    assert without_rtk(fixed) == {"mapping": [fixed["mapping"][0]]}


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.uint64])
def test_single_bit_mutation_still_rejected(dtype):
    value = np.array([1, 2, 3], dtype=dtype)
    changed = value.copy()
    changed.view(np.uint8)[0] ^= 1
    assert serialized_encode(value) != serialized_encode(changed)


def test_mapping_order_shape_and_tuple_semantics_are_not_discarded():
    assert serialized_encode({"a": 1, "b": 2}) != serialized_encode({"b": 2, "a": 1})
    assert serialized_encode(np.arange(4).reshape(2, 2)) != serialized_encode(np.arange(4))
    assert serialized_encode((1, 2)) != serialized_encode([1, 2])
