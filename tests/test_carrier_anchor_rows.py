import numpy as np

import gnss_gpu.carrier_anchor_rows as carrier_anchor_rows
from gnss_gpu.carrier_rescue import MUPF_L1_COMPAT_SYSTEM_IDS, MUPF_L1_WAVELENGTH_M
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, MupfConfig


def test_select_carrier_anchor_rows_skips_when_anchor_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        carrier_anchor_rows,
        "_select_same_band_carrier_rows",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    rows = carrier_anchor_rows.select_carrier_anchor_rows(
        [],
        np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        MupfConfig(enabled=True),
        CarrierRescueConfig(anchor_enabled=False),
    )

    assert rows == {}
    assert calls == []


def test_select_carrier_anchor_rows_delegates_with_mupf_thresholds(monkeypatch):
    calls = []

    def fake_select(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {(0, 1): {"row": True}}

    monkeypatch.setattr(
        carrier_anchor_rows,
        "_select_same_band_carrier_rows",
        fake_select,
    )

    rows = carrier_anchor_rows.select_carrier_anchor_rows(
        ["m0"],
        np.asarray([1.0], dtype=np.float64),
        np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        MupfConfig(enabled=True, snr_min=28.0, elev_min=0.2),
        CarrierRescueConfig(anchor_enabled=True),
    )

    assert rows == {(0, 1): {"row": True}}
    assert calls[0]["args"][0] == ["m0"]
    assert calls[0]["kwargs"]["snr_min"] == 28.0
    assert calls[0]["kwargs"]["elev_min"] == 0.2
    assert calls[0]["kwargs"]["wavelength_m"] == MUPF_L1_WAVELENGTH_M
    assert calls[0]["kwargs"]["allowed_system_ids"] == MUPF_L1_COMPAT_SYSTEM_IDS
