from dataclasses import asdict,replace
from gnss_gpu.drift_carrier_paths_cpu import DriftFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_drift_selection import apply_gate,tasks


def test_drift_fit_selection_ignores_truth_and_retains_guards():
    single=DriftFit(.1,.2,.5,None,None,.01,True,True,'ok',1.)
    two=replace(single,first_delay_chips=.2,excess_chips=.1,relative_doppler_hz=.1)
    gate=PathSelectionGate(-20.,299,.01,.05)
    row=dict(kind='null',true_delay_chips=.1,uncalibrated_score=-20.,fits=dict(single=asdict(single),two=asdict(two)))
    assert apply_gate(row,gate)['selected']['model']=='single_path'
    assert apply_gate(dict(row,kind='echo',true_delay_chips=.9),gate)['selected']==apply_gate(row,gate)['selected']
    row['uncalibrated_score']=-19.
    assert apply_gate(row,gate)['selected']['model']=='two_path'
    row['fits']['two']=asdict(replace(two,locally_usable=False,reason='search boundary'))
    assert apply_gate(row,gate)['selected']['model']=='ambiguous'


def test_signed_mixture_tasks_reproducible_and_disjoint():
    a=tasks('null',980000,299)
    assert a==tasks('null',980000,299)
    assert all(-2<r[1]<2 and -1<r[2]<1 for r in a)
    assert min(r[1] for r in a)<-1.9 and max(r[1] for r in a)>1.9
    assert {r[3] for r in a}.isdisjoint(r[3] for r in tasks('null',990000,64))
