import numpy as np

from experiments.analyze_wp31_static_ddpr_integrity import (
    ddpr_pair_biases,
    ddpr_scores,
)
from gnss_gpu.local_fgo import DDPseudorangeEpoch


def test_ddpr_scores_trim_worst_satellite():
    obs = DDPseudorangeEpoch(
        dd_pseudorange_m=np.array([0.0, 20.0]),
        sat_ecef_k=np.array([[20e6, 0.0, 0.0], [0.0, 20e6, 0.0]]),
        sat_ecef_ref=np.array([[0.0, 20e6, 0.0], [0.0, 20e6, 0.0]]),
        base_range_k=np.array([20e6, 20e6]),
        base_range_ref=np.array([20e6, 20e6]),
        sat_ids=("G01", "G02"),
        ref_sat_ids=("G03", "G03"),
    )
    result = ddpr_scores(np.zeros(3), [obs], sigma_m=4.0, blocks=1)
    assert result["ddpr_rows"] == 2
    assert result["ddpr_trim1_excluded"] == ["G02"]
    assert result["ddpr_trim1_mean"] < result["ddpr_cauchy_mean"]


def test_ddpr_scores_remove_accepted_anchor_pair_bias():
    geometry = dict(
        sat_ecef_k=np.array([[20e6, 0.0, 0.0], [0.0, 20e6, 0.0]]),
        sat_ecef_ref=np.array([[0.0, 20e6, 0.0], [0.0, 20e6, 0.0]]),
        base_range_k=np.array([20e6, 20e6]),
        base_range_ref=np.array([20e6, 20e6]),
        sat_ids=("G01", "G02"),
        ref_sat_ids=("G03", "G03"),
    )
    calibration = DDPseudorangeEpoch(
        dd_pseudorange_m=np.array([5.0, 5.0]), **geometry
    )
    target = DDPseudorangeEpoch(
        dd_pseudorange_m=np.array([5.0, 7.0]), **geometry
    )
    biases = ddpr_pair_biases(np.zeros(3), [calibration] * 5, min_samples=5)
    result = ddpr_scores(
        np.zeros(3), [target], sigma_m=4.0, blocks=1, pair_bias_m=biases
    )
    assert biases == {("G01", "G03"): 5.0, ("G02", "G03"): 5.0}
    assert result["ddpr_median_abs_m"] == 1.0
