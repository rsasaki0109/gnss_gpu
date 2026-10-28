from dataclasses import replace
import pytest
from gnss_gpu.common_carrier_paths_cpu import CarrierFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from gnss_gpu.calibrated_common_carrier_cpu import select_common_carrier_paths
from experiments.run_cpu_common_carrier_selection import tasks


def test_strict_gate_both_guards_and_no_fallback():
    fit=CarrierFit(.1,.2,None,None,.01,True,True,'ok',1.)
    two=replace(fit,first_delay_chips=.2,excess_chips=.1,relative_doppler_hz=.7)
    gate=PathSelectionGate(-20.,299,.01,.05)
    score=dict(score=-20.,single=fit,two=two)
    assert select_common_carrier_paths(score,gate)['model']=='single_path'
    score['score']=-19.
    assert select_common_carrier_paths(score,gate)['model']=='two_path'
    score['two']=replace(two,locally_usable=False,reason='search boundary')
    assert select_common_carrier_paths(score,gate)['model']=='ambiguous'
    score['score']=-21.
    score['single']=replace(fit,locally_usable=False,reason='optimizer did not converge')
    assert select_common_carrier_paths(score,gate)['model']=='ambiguous'
    with pytest.raises(ValueError):select_common_carrier_paths(dict(score,score=float('nan')),gate)
    with pytest.raises(ValueError):select_common_carrier_paths(score,replace(gate,score=float('inf')))


def test_mixture_tasks_reproducible_disjoint_and_signed():
    a=tasks(0.,800000,299)
    assert a==tasks(0.,800000,299)
    assert all(-1<r[1]<1 for r in a)
    assert min(r[1] for r in a)<-.9 and max(r[1] for r in a)>.9
    assert {r[2] for r in a}.isdisjoint(r[2] for r in tasks(0.,810000,128))
