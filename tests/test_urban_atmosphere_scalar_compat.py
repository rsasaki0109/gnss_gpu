"""Single-element arrays retain legacy scalar bits with NumPy 2.4."""

import struct
import numpy as np
import pytest
from gnss_gpu.urban_signal_sim import _atmospheric_scalar


@pytest.mark.parametrize("bits", [0, 1 << 63, 1, 0x7fefffffffffffff, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000037])
@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_scalar_bits_and_input_ownership(bits, shape):
    value = np.array([bits], dtype=np.uint64).view(np.float64).reshape(shape)
    saved = value.tobytes()
    actual = _atmospheric_scalar(value)
    assert struct.pack("d", actual) == saved
    assert value.tobytes() == saved


def test_custom_float_dispatch_is_preserved():
    calls = []
    class Custom:
        def __float__(self):
            calls.append(1)
            return 3.25
    assert _atmospheric_scalar(Custom()) == 3.25
    assert calls == [1]


def test_multielement_arrays_remain_invalid():
    with pytest.raises(TypeError):
        _atmospheric_scalar(np.array([1., 2.]))
