"""Tests for DD-carrier geometry NLOS weight scaling."""

from __future__ import annotations

import numpy as np

from gnss_gpu.dd_carrier import DDResult
from gnss_gpu.dd_carrier_observation import compute_dd_carrier_observation
from gnss_gpu.nlos_mask import NlosMaskTables
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, DDCarrierConfig


class _FakeCarrierComputer:
    def compute_dd(self, tow, measurements, pf_estimate):
        return DDResult(
            dd_carrier_cycles=np.array([0.1], dtype=np.float64),
            sat_ecef_k=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            sat_ecef_ref=np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
            base_range_k=np.array([1.0], dtype=np.float64),
            base_range_ref=np.array([2.0], dtype=np.float64),
            dd_weights=np.array([2.0], dtype=np.float64),
            wavelengths_m=np.array([0.19], dtype=np.float64),
            ref_sat_ids=("G01",),
            n_dd=1,
            sat_ids=("G02@L1",),
        )


def test_compute_dd_carrier_applies_nlos_mask_to_dd_weights():
    tables = NlosMaskTables(weak={3: {"G02"}}, strong={})
    decision = compute_dd_carrier_observation(
        _FakeCarrierComputer(),
        100.0,
        [],
        None,
        DDCarrierConfig(enabled=True),
        CarrierRescueConfig(),
        dd_pseudorange_result=None,
        ess_ratio=None,
        spread_m=None,
        nlos_tables=tables,
        epoch_idx=3,
        nlos_k_weak=2.0,
    )
    assert decision.result is not None
    assert decision.result.dd_weights[0] == np.float64(1.0)
