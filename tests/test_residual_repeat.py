from dataclasses import asdict
from gnss_gpu.dynamic_multipath_cpu import DynamicFit
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_residual_repeat import apply_draws,summarize,PROFILES


def test_draws_use_identical_observations_and_four_test_family():
    draws={name:dict(gates={m:asdict(PathSelectionGate(v,1920,.01,.05)) for m,v in [('maximum',maximum),('ranked',-10.)]}) for name,maximum in [('original',0.),('new',1.)]}
    fit=DynamicFit(0.,.1,1.,.01,True,True,'ok',1.)
    score=asdict(DynamicScore(.2,2.,True,-5.,fit))
    rows=[apply_draws(dict(profile=p,echo_voltage=.8,doppler_hz=0.,phase_rad=0.,replica=r,true_delay_chips=0.,score=score),draws) for p in PROFILES for r in range(2)]
    report=summarize(rows,draws)
    assert sum(c['primary'] for c in report['contrasts'])==4
    assert all(c['attempts']==2 and c['gains']==2 for c in report['contrasts'])
    assert report['descriptive_variability']['maximum']['absolute_threshold_difference']==1
    assert report['descriptive_variability']['ranked']['echo_successes_on_same_data']['edge']==[2,2]
