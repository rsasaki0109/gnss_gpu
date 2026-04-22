import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.pf_smoother_config import WidelaneConfig
from gnss_gpu.widelane import WidelaneDDStats
from gnss_gpu.widelane_observation import compute_widelane_observation


class FakeWidelaneComputer:
    def __init__(self, result, stats):
        self.result = result
        self.stats = stats
        self.calls = []

    def compute_dd(self, tow, measurements, pf_est, *, rover_weights, min_fix_rate):
        self.calls.append(
            {
                "tow": tow,
                "measurements": measurements,
                "pf_est": pf_est,
                "rover_weights": rover_weights,
                "min_fix_rate": min_fix_rate,
            }
        )
        return self.result, self.stats


def _dd_result(values=(2.0, 3.0, 4.0)):
    n = len(values)
    return DDPseudorangeResult(
        dd_pseudorange_m=np.asarray(values, dtype=np.float64),
        sat_ecef_k=np.array([[10.0 + i, 0.0, 0.0] for i in range(n)], dtype=np.float64),
        sat_ecef_ref=np.array([[8.0 + i, 0.0, 0.0] for i in range(n)], dtype=np.float64),
        base_range_k=np.zeros(n, dtype=np.float64),
        base_range_ref=np.zeros(n, dtype=np.float64),
        dd_weights=np.ones(n, dtype=np.float64),
        ref_sat_ids=tuple("G01" for _ in range(n)),
        n_dd=n,
    )


def test_compute_widelane_observation_accepts_valid_result():
    stats = WidelaneDDStats(
        n_candidate_pairs=4,
        n_fixed_pairs=3,
        n_dd=3,
        fix_rate=0.75,
        reason="ok",
    )
    computer = FakeWidelaneComputer(_dd_result(), stats)
    config = WidelaneConfig(enabled=True, min_fix_rate=0.3, dd_sigma=0.12)

    decision = compute_widelane_observation(
        computer,
        10.0,
        ["m0"],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        config,
        spread_m=None,
    )

    assert decision.used is True
    assert decision.skipped is False
    assert decision.dd_pseudorange_result is not None
    assert decision.input_pairs == 4
    assert decision.fixed_pairs == 3
    assert decision.fix_rate == 0.75
    assert decision.dd_sigma_m == 0.12
    assert decision.gate_info["reason"] == "ok"
    assert computer.calls[0]["min_fix_rate"] == 0.3


def test_compute_widelane_observation_reports_gate_skip():
    stats = WidelaneDDStats(
        n_candidate_pairs=4,
        n_fixed_pairs=2,
        n_dd=3,
        fix_rate=0.5,
        reason="ok",
    )
    computer = FakeWidelaneComputer(_dd_result(), stats)
    config = WidelaneConfig(
        enabled=True,
        min_fix_rate=0.3,
        gate_min_fixed_pairs=3,
    )

    decision = compute_widelane_observation(
        computer,
        10.0,
        [],
        np.zeros(3, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        config,
        spread_m=None,
    )

    assert decision.used is False
    assert decision.skipped is True
    assert decision.gate_skipped is True
    assert decision.gate_info["reason"] == "gate_min_fixed_pairs"


def test_compute_widelane_observation_preserves_empty_gate_reason_on_no_result():
    stats = WidelaneDDStats(
        n_candidate_pairs=4,
        n_fixed_pairs=1,
        n_dd=0,
        fix_rate=0.25,
        reason="low_fix_rate",
    )
    computer = FakeWidelaneComputer(None, stats)
    config = WidelaneConfig(enabled=True, min_fix_rate=0.3)

    decision = compute_widelane_observation(
        computer,
        10.0,
        [],
        None,
        np.ones(1, dtype=np.float64),
        config,
        spread_m=None,
    )

    assert decision.used is False
    assert decision.skipped is True
    assert decision.low_fix_rate is True
    assert decision.input_pairs == 4
    assert decision.fixed_pairs == 1
    assert decision.gate_info["reason"] is None
