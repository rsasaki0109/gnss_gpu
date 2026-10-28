import numpy as np
from gnss_gpu.bandwidth_drift_paths_cpu import _problem
from gnss_gpu.multipath_information_cpu import CHIP_RATE
from experiments.diagnose_cpu_wideband_boundary import select_cases,dense_problem,diagnostic
from experiments.replicate_cpu_drift_bandwidth import synthesize


def test_selection_includes_all_wide_echo_boundary_cases_only():
    base=dict(kind='echo',bandwidth_chips=7.2,available=False,fits=dict(two=dict(reason='search boundary')))
    rows=[base,dict(base,kind='null'),dict(base,bandwidth_chips=3.6),dict(base,available=True),
          dict(base,fits=dict(two=dict(reason='weak local delay information')))]
    assert select_cases(rows)==[base]


def test_explicit_residual_matches_compact_problem_for_both_floors():
    b,_,truth=synthesize(('echo',7.2,.5,-.3,1260000));b=b[:10]
    weights=np.linspace(.4,1.8,b.shape[1]);solve=_problem(b,weights,True,7.2*CHIP_RATE)
    dense=dense_problem(b,weights,7.2*CHIP_RATE)
    for gap in (.01,.001):
        p=[truth,-.3,.5,gap,0.]
        detail=diagnostic(solve,dense,p,True,truth)
        assert abs(detail['compact_dense_difference'])<1e-8
        assert abs(solve(p)[0]-detail['loss'])<1e-8
        assert detail['cancellation_ratio']>=1-1e-12
        assert detail['template_condition']>=1
        assert detail['error_m']==0
