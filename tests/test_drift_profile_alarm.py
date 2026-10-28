import numpy as np
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_drift_profile_alarm import alarm_score,apply_alarm,summarize


def test_score_direction_and_strict_gate():
    assert alarm_score(10,[],100)==0
    assert alarm_score(10,[11,12],100)==0
    score=alarm_score(10,[8,9],100)
    assert np.isclose(score,200*np.log(10/8))
    row=dict(profile_alarm=dict(score=score))
    assert not apply_alarm(row,PathSelectionGate(score,299,.01,.05))['warning']
    assert apply_alarm(row,PathSelectionGate(score-1,299,.01,.05))['warning']
    assert 'warning' not in row


def test_summary_counts_unavailable_and_missed_errors_separately():
    rows=[]
    for kind,ratio in [('null',0.),('echo',.2),('echo',.8)]:
        for error,warning in [(None,False),(12,True),(-11,False),(1,True),(2,False)]:
            rows.append(dict(kind=kind,echo_voltage=ratio,error_m=error,warning=warning,
                success=error is not None and abs(error)<10,profile_alarm=dict(unresolved=False)))
    for c in summarize(rows)['cells']:
        assert (c['attempts'],c['baseline_unavailable'],c['available_large_errors'],c['caught_large_errors'],c['missed_large_errors'],c['warnings_on_success'],c['retained_success'])==(5,1,2,1,1,1,1)
