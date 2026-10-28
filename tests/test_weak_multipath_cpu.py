import numpy as np
import pytest
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, acquire_ca_bit_edge
from gnss_gpu.weak_multipath_cpu import coherent_weak_period,fit_weak_ca_two_paths
from gnss_gpu.weak_range_refinement_cpu import refine_weak_ca_delay
from gnss_gpu.multipath_fit_cpu import fit_ca_two_paths
from tests.test_multipath_fit_cpu import observation


def test_coherent_two_path_wrapper_uses_acquired_edge_and_preserves_input():
    data=np.tile(observation(spc=2).iq,(20,1));data[7,900:]*=-1;data[8:]*=-1
    saved=data.copy();cfg=CaBitEdgeSearch(CaSearch(2046000.,(-250.,0.,250.)))
    candidate=acquire_ca_bit_edge(data,cfg)
    packed=coherent_weak_period(data,cfg,candidate)
    fitted=fit_weak_ca_two_paths(data,cfg,candidate)
    assert fitted==fit_ca_two_paths(packed,2046000.)
    assert fitted.single.delay_samples==pytest.approx(refine_weak_ca_delay(data,cfg,candidate).delay_samples)
    np.testing.assert_array_equal(data,saved)
    assert coherent_weak_period(np.zeros_like(data),cfg,candidate) is None
    with pytest.raises(ValueError):coherent_weak_period(data[:-1],cfg,candidate)
