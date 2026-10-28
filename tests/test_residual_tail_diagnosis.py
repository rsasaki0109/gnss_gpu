import numpy as np
from experiments.diagnose_cpu_residual_tail import dense_problem,select_cases
from experiments.stress_cpu_dynamic_noise import synthesize,PROFILES
from gnss_gpu.residual_dynamic_paths_cpu import _problem
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape


def test_explicit_lstsq_agrees_with_gram_even_for_nearby_paths():
    blocks,_,_=synthesize(('center',.8,.7,0.,710000))
    f,_=ca_passband_spectrum(3.6*CHIP_RATE);weights=spectral_noise_shape(f,'center')
    dense,_=dense_problem(blocks,weights);reduced=_problem(blocks,weights)
    for p in ((.1,.01,0.),(.2,.1,.7),(-.1,.5,-8.)):
        np.testing.assert_allclose(dense(p)[0],reduced(p)[0],rtol=1e-12)


def test_case_selection_is_complete_and_deterministic():
    rows=[dict(profile=p,replica=r,score={'bic_improvement':float(r)}) for p in PROFILES for r in range(640)]
    cases=select_cases(rows[::-1])
    assert len(cases)==12
    assert [r['replica'] for r in cases]==[639,638,637,320]*3
