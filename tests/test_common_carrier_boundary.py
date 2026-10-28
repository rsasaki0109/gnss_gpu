import numpy as np
from experiments.diagnose_cpu_common_carrier_boundary import dense_problem,select_cases
from experiments.stress_cpu_common_carrier_phase import synthesize
from gnss_gpu.common_carrier_paths_cpu import _problem


def test_dense_objective_matches_separable_at_small_gaps():
    blocks,_,_=synthesize((.8,-.4,0.,np.pi,880000))
    weights=np.linspace(.3,2.,blocks.shape[1])
    dense=dense_problem(blocks,weights);solve=_problem(blocks,weights,True)
    for gap in (.001,.01,.1):
        p=(.12,-.5,gap,.01)
        d=dense(p)
        np.testing.assert_allclose(d['loss'],solve(p,True)[0],rtol=1e-12)
        np.testing.assert_allclose(d['loss'],solve(p)[0],rtol=1e-12)
        assert d['template_condition']>=1 and d['cancellation_ratio']>=1-1e-12


def test_selection_keeps_all_boundary_cases_only():
    rows=[dict(selected=dict(model=m,reason=r)) for m,r in [('ambiguous','search boundary'),('ambiguous','weak local delay information'),('two_path','ok'),('ambiguous','search boundary')]]
    assert select_cases(rows)==[rows[0],rows[3]]
