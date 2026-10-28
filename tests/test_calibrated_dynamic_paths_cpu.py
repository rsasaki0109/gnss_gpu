from dataclasses import replace
import numpy as np
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore,fit_single_path,select_dynamic_paths
from gnss_gpu.dynamic_multipath_cpu import DynamicFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def test_single_path_recovers_delay_and_full_residual():
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    rng=np.random.default_rng(42)
    signal=s*np.exp(-2j*np.pi*f*.123/CHIP_RATE)*(.4+.7j)
    y=np.broadcast_to(signal,(10,len(s))).copy()
    delay,loss,ok=fit_single_path(y)
    assert ok and abs(delay-.123)<1e-7 and loss<1e-12
    y+=.01*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))
    delay,loss,ok=fit_single_path(y)
    normalized=y/np.max(np.abs(y));template=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE)
    amplitude=np.vdot(np.broadcast_to(template,y.shape),normalized)/(10*np.vdot(template,template))
    np.testing.assert_allclose(loss,np.linalg.norm(normalized-amplitude*template)**2,rtol=1e-12)


def test_strict_gate_and_unavailable_guards():
    fit=DynamicFit(.1,.1,2.,.01,True,True,'ok',1.)
    score=DynamicScore(.2,2.,True,-10.,fit);gate=PathSelectionGate(-10.,299,.01,.05)
    assert select_dynamic_paths(score,gate)['model']=='single_path'
    above=replace(score,bic_improvement=-9.)
    assert select_dynamic_paths(above,gate)['delay_chips']==.1
    assert select_dynamic_paths(replace(above,two_path=replace(fit,locally_usable=False)),gate)['model']=='ambiguous'
    assert select_dynamic_paths(replace(score,single_converged=False),gate)['model']=='ambiguous'
