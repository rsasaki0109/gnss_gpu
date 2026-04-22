import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.widelane import WidelaneDDStats
from gnss_gpu.widelane_gate import _gate_widelane_pseudorange_result


def test_gate_widelane_pseudorange_result_filters_outlier_pair():
    result = DDPseudorangeResult(
        dd_pseudorange_m=np.array([2.0, 3.0, 20.0], dtype=np.float64),
        sat_ecef_k=np.array(
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        sat_ecef_ref=np.array(
            [[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        base_range_k=np.zeros(3, dtype=np.float64),
        base_range_ref=np.zeros(3, dtype=np.float64),
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )
    stats = WidelaneDDStats(n_candidate_pairs=4, n_fixed_pairs=3, n_dd=3, fix_rate=0.75, reason="ok")

    gated, info = _gate_widelane_pseudorange_result(
        result,
        stats,
        np.zeros(3, dtype=np.float64),
        min_fixed_pairs=3,
        min_fix_rate=0.7,
        min_spread_m=1.0,
        spread_m=2.0,
        max_epoch_median_residual_m=2.0,
        max_pair_residual_m=5.0,
        min_pairs=2,
    )

    assert gated is not None
    assert gated.n_dd == 2
    assert info["reason"] == "ok"
    assert info["pair_rejected"] == 1
    assert info["raw_abs_res_max_m"] == 18.0
    assert info["kept_abs_res_median_m"] == 0.5
