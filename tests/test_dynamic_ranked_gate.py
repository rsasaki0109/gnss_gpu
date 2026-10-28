from dataclasses import asdict
from gnss_gpu.dynamic_multipath_cpu import DynamicFit
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_dynamic_ranked_gate import calibrate,apply_gates,summarize


def test_rank_fixed_by_budgets_and_shared_scores():
    gates,rank=calibrate([dict(score=dict(bic_improvement=float(i))) for i in range(1000)])
    assert rank['rank_from_largest']==5
    assert rank['actual_failure_bound']<.05
    assert gates['maximum']['score']==999 and gates['ranked']['score']==995


def test_shared_pairing_and_two_primary_tests():
    gates={k:asdict(PathSelectionGate(v,1000,.01,.05)) for k,v in [('maximum',0.),('ranked',-10.)]}
    fit=DynamicFit(0.,.1,1.,.01,True,True,'ok',1.)
    score=asdict(DynamicScore(.2,2.,True,-5.,fit))
    rows=[apply_gates(dict(echo_voltage=.8,doppler_hz=fd,phase_rad=0.,replica=r,true_delay_chips=0.,score=score),gates) for fd in (0.,.1,.7) for r in range(2)]
    report=summarize(rows,gates)
    assert len(report['cells'])==6
    assert sum(c['primary'] for c in report['contrasts'])==2
    assert all(c['gains']==2 and c['losses']==0 and c['attempts']==2 for c in report['contrasts'])
    assert report['contrasts'][2]['adjusted_primary_p'] is None
