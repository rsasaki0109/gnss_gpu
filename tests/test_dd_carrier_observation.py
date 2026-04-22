import numpy as np

from gnss_gpu.dd_carrier import DDResult, GPS_L1_WAVELENGTH
from gnss_gpu.dd_carrier_observation import compute_dd_carrier_observation
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, DDCarrierConfig


class FakeDDCarrierComputer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def compute_dd(self, tow, measurements, pf_estimate):
        self.calls.append(
            {
                "tow": tow,
                "measurements": measurements,
                "pf_estimate": pf_estimate,
            }
        )
        return self.result


class _DummyDDPseudorangeResult:
    def __init__(self, n_dd: int):
        self.n_dd = n_dd


def _dd_result(values=(0.05, 0.10, -0.15)):
    n = len(values)
    sat = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), n, axis=0)
    base_range = np.full(n, 2.0, dtype=np.float64)
    return DDResult(
        dd_carrier_cycles=np.asarray(values, dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(n, dtype=np.float64),
        wavelengths_m=np.full(n, GPS_L1_WAVELENGTH, dtype=np.float64),
        ref_sat_ids=tuple("G01" for _ in range(n)),
        n_dd=n,
    )


def test_compute_dd_carrier_observation_computes_and_gates_result():
    computer = FakeDDCarrierComputer(_dd_result())

    decision = compute_dd_carrier_observation(
        computer,
        10.0,
        ["m0"],
        np.zeros(3, dtype=np.float64),
        DDCarrierConfig(enabled=True),
        CarrierRescueConfig(),
        dd_pseudorange_result=None,
        ess_ratio=None,
        spread_m=None,
        collect_diagnostics=True,
    )

    assert decision.result is not None
    assert decision.result.n_dd == 3
    assert decision.gate_stats is not None
    assert decision.input_pairs == 3
    assert decision.raw_abs_afv_median_cycles is not None
    assert decision.gate_pairs_rejected == 0
    assert decision.gate_epoch_skipped is False
    assert computer.calls[0]["tow"] == 10.0


def test_compute_dd_carrier_observation_reports_pair_rejections():
    decision = compute_dd_carrier_observation(
        FakeDDCarrierComputer(_dd_result((0.05, 0.40, -0.10, 0.12))),
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        DDCarrierConfig(enabled=True, gate_afv_cycles=0.2),
        CarrierRescueConfig(),
        dd_pseudorange_result=None,
        ess_ratio=None,
        spread_m=None,
        collect_diagnostics=True,
    )

    assert decision.result is not None
    assert decision.result.n_dd == 3
    assert decision.gate_pairs_rejected == 1
    assert decision.gate_epoch_skipped is False
    np.testing.assert_allclose(decision.result.dd_carrier_cycles, [0.05, -0.10, 0.12])


def test_compute_dd_carrier_observation_reports_epoch_rejection():
    decision = compute_dd_carrier_observation(
        FakeDDCarrierComputer(_dd_result((0.24, -0.26, 0.23))),
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        DDCarrierConfig(enabled=True, gate_afv_cycles=0.4, gate_epoch_median_cycles=0.2),
        CarrierRescueConfig(),
        dd_pseudorange_result=None,
        ess_ratio=None,
        spread_m=None,
    )

    assert decision.result is None
    assert decision.gate_stats is not None
    assert decision.gate_epoch_skipped is True


def test_compute_dd_carrier_observation_tightens_epoch_gate_for_low_ess_context():
    decision = compute_dd_carrier_observation(
        FakeDDCarrierComputer(_dd_result((0.12, 0.13, 0.14))),
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        DDCarrierConfig(
            enabled=True,
            gate_epoch_median_cycles=0.5,
            gate_low_ess_epoch_median_cycles=0.1,
            gate_low_ess_max_ratio=0.2,
        ),
        CarrierRescueConfig(),
        dd_pseudorange_result=_DummyDDPseudorangeResult(2),
        ess_ratio=0.1,
        spread_m=None,
    )

    assert decision.result is None
    assert decision.gate_epoch_skipped is True


def test_compute_dd_carrier_observation_collects_raw_summary_for_support_guard():
    decision = compute_dd_carrier_observation(
        FakeDDCarrierComputer(_dd_result((0.05, 0.10, 0.15))),
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        DDCarrierConfig(enabled=True),
        CarrierRescueConfig(skip_low_support_min_raw_afv_median_cycles=0.2),
        dd_pseudorange_result=None,
        ess_ratio=None,
        spread_m=None,
        collect_diagnostics=False,
    )

    assert decision.raw_abs_afv_median_cycles == 0.1
    assert decision.raw_abs_afv_max_cycles == 0.15
