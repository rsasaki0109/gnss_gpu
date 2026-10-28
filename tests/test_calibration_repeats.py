from dataclasses import asdict
from experiments.run_cpu_calibration_repeats import calibrate_repeats,apply_repeats
from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit


def fixture():
    return [dict(profile=p,calibration_repeat=r,fit=dict(bic_improvement=score+r))
            for r in range(3) for p,score in [('white',-12.),('center',-10.),('edge',-8.)] for _ in range(477)]


def test_calibration_draws_are_not_pooled():
    gates=calibrate_repeats(fixture())
    for repeat in range(3):
        assert gates[str(repeat)]['white']['score']==-12+repeat
        assert gates[str(repeat)]['multi']['score']==-8+repeat
        assert gates[str(repeat)]['white']['calibration_count']==477


def test_all_gates_apply_to_same_observation_without_truth_in_selection():
    gates=calibrate_repeats(fixture())
    fit=BandlimitedFit(.2,.1,.1,.2,'single_path','initial',-9.,.02,(1.,.8),True)
    row=dict(fit=asdict(fit),true_delay_chips=.1)
    a=apply_repeats(row,gates)
    b=apply_repeats(dict(row,true_delay_chips=.9),gates)
    for repeat in gates:
        for coverage in ('white','multi'):
            assert a['outcomes'][repeat][coverage]['fit']==b['outcomes'][repeat][coverage]['fit']
    assert row['fit']==asdict(fit)
