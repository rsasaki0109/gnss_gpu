import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.dd_pseudorange_observation import compute_dd_pseudorange_observation
from gnss_gpu.pf_smoother_config import DDPseudorangeConfig


class FakeDDPseudorangeComputer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def compute_dd(self, tow, measurements, pf_estimate, *, rover_weights):
        self.calls.append(
            {
                "tow": tow,
                "measurements": measurements,
                "pf_estimate": pf_estimate,
                "rover_weights": rover_weights,
            }
        )
        return self.result


def _dd_result(values=(0.2, 0.3, 0.4)):
    n = len(values)
    sat = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), n, axis=0)
    base_range = np.full(n, 2.0, dtype=np.float64)
    return DDPseudorangeResult(
        dd_pseudorange_m=np.asarray(values, dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(n, dtype=np.float64),
        ref_sat_ids=tuple("G01" for _ in range(n)),
        n_dd=n,
    )


def test_compute_dd_pseudorange_observation_computes_and_gates_result():
    computer = FakeDDPseudorangeComputer(_dd_result())
    decision = compute_dd_pseudorange_observation(
        computer,
        10.0,
        ["m0"],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        DDPseudorangeConfig(enabled=True),
        collect_diagnostics=True,
    )

    assert decision.result is not None
    assert decision.result.n_dd == 3
    assert decision.gate_stats is not None
    assert decision.input_pairs == 3
    assert decision.raw_abs_res_median_m is not None
    assert decision.gate_pairs_rejected == 0
    assert decision.gate_epoch_skipped is False
    assert computer.calls[0]["tow"] == 10.0


def test_compute_dd_pseudorange_observation_uses_existing_result_without_compute():
    computer = FakeDDPseudorangeComputer(_dd_result((99.0, 99.0, 99.0)))
    existing = _dd_result()

    decision = compute_dd_pseudorange_observation(
        computer,
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        DDPseudorangeConfig(enabled=True),
        existing_result=existing,
        existing_input_pairs=3,
    )

    assert decision.result is not None
    assert computer.calls == []
    assert decision.input_pairs == 3
    np.testing.assert_allclose(decision.result.dd_pseudorange_m, [0.2, 0.3, 0.4])


def test_compute_dd_pseudorange_observation_reports_pair_rejections():
    decision = compute_dd_pseudorange_observation(
        None,
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        DDPseudorangeConfig(enabled=True, gate_residual_m=1.0),
        existing_result=_dd_result((0.2, 0.3, 8.0, 0.4)),
        existing_input_pairs=4,
    )

    assert decision.result is not None
    assert decision.result.n_dd == 3
    assert decision.gate_pairs_rejected == 1
    assert decision.gate_epoch_skipped is False


def test_compute_dd_pseudorange_observation_reports_epoch_rejection():
    decision = compute_dd_pseudorange_observation(
        None,
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        DDPseudorangeConfig(enabled=True, gate_epoch_median_m=1.0),
        existing_result=_dd_result((3.0, 4.0, 5.0)),
        existing_input_pairs=3,
    )

    assert decision.result is None
    assert decision.gate_stats is not None
    assert decision.gate_epoch_skipped is True
