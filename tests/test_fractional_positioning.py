import numpy as np
import pytest
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch,acquire_ca_bit_edge
from gnss_gpu.weak_range_refinement_cpu import refine_weak_ca_delay
from gnss_gpu.weak_positioning_cpu import resolve_ca_range
import experiments.run_cpu_fractional_positioning as experiment


def test_aperture_matches_dense_independent_code_and_bit_integration(monkeypatch):
    monkeypatch.setattr(experiment,'NOISE_DB',-10000.)
    rho=22001723.;phase=.37
    iq=experiment.aperture_iq(7,rho,1.,phase,0).reshape(-1,2)
    indices=np.r_[np.arange(64),20*experiment.COUNT-experiment.START+np.arange(-2,3)]
    samples=experiment.START+indices[:,None]+(np.arange(8192)+.5)/8192-.5
    chips=np.floor((samples-rho*experiment.FS/experiment.C_LIGHT)/2).astype(int)%1023
    values=generate_ca_code(7)[chips]*(-1.)**np.floor(samples/(20*experiment.COUNT))
    expected=values.mean(axis=1)*np.exp(1j*phase)
    np.testing.assert_allclose(iq[indices,0]+1j*iq[indices,1],expected,atol=3e-4)


@pytest.mark.parametrize('prn',[1,4,8])
def test_noiseless_iq_to_assisted_fractional_range(monkeypatch,prn):
    monkeypatch.setattr(experiment,'NOISE_DB',-10000.)
    rho=21001723.41+prn*200000
    iq=experiment.aperture_iq(prn,rho,1.,.53,0)[:20]
    cfg=CaBitEdgeSearch(CaSearch(experiment.FS,(-250.,0.,250.),prn))
    candidate=acquire_ca_bit_edge(iq,cfg)
    fitted=refine_weak_ca_delay(iq,cfg,candidate)
    resolved=resolve_ca_range(fitted.delay_samples+experiment.START%2046,experiment.FS,prior_range_m=rho-1500,prior_bound_m=6000)
    assert resolved.pseudorange_m==pytest.approx(rho,abs=.001)
