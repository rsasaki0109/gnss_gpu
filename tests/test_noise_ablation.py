from dataclasses import asdict
import pytest
from experiments.run_cpu_noise_ablation import calibrate_arms,apply_arms,ARMS
from gnss_gpu.bandlimited_multipath_v2_cpu import BandlimitedFit


def calibration():
    return [dict(profile=p,old_fit=dict(bic_improvement=a),fit=dict(bic_improvement=b))
            for p,a,b in [('white',-12.,-11.),('center',2.,-10.),('edge',-15.,-7.)] for _ in range(477)]


def test_factorial_gates_use_correct_scores_and_coverage():
    gates=calibrate_arms(calibration())
    assert {a:g['score'] for a,g in gates['arms'].items()}==dict(
        unweighted_white=-12.,unweighted_multi=2.,weighted_white=-11.,weighted_multi=-7.)
    assert 6*.99**477 <= .05


def test_four_arms_pair_same_data_without_changing_raw_fit():
    gates=calibrate_arms(calibration())
    fit=BandlimitedFit(.2,.1,.1,.2,'single_path','initial',-9.,.02,(1.,.8),True)
    row=dict(fit=asdict(fit),old_fit=asdict(fit),true_delay_chips=.1)
    output=apply_arms(row,gates)
    assert row['fit']['selected_model']=='single_path'
    for arm in ARMS:
        expected='two_path' if arm.endswith('_white') else 'single_path'
        assert output['outcomes'][arm]['fit']['selected_model']==expected


def test_incomplete_calibration_cannot_claim_six_stratum_budget():
    with pytest.raises(ValueError): calibrate_arms(calibration()[:-1])
