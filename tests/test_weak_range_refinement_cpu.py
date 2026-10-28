from dataclasses import replace
import numpy as np
import pytest
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, acquire_ca_bit_edge
from gnss_gpu.weak_range_refinement_cpu import refine_weak_ca_delay
from tests.test_code_delay_cpu import signal


@pytest.mark.parametrize('delay',[-12.37,.17,1022.9])
def test_fractional_aperture_delay_after_unknown_edge_acquisition(delay):
    iq=np.tile(signal(delay,2).iq,(20,1))
    iq[7,900:]*=-1; iq[8:]*=-1
    saved=iq.copy()
    cfg=CaBitEdgeSearch(CaSearch(2046000.,(-250.,0.,250.)))
    candidate=acquire_ca_bit_edge(iq,cfg)
    fitted=refine_weak_ca_delay(iq,cfg,candidate)
    assert fitted.delay_samples==pytest.approx(delay,abs=2e-6)
    np.testing.assert_array_equal(iq,saved)
    with pytest.raises(ValueError):refine_weak_ca_delay(iq[:19],cfg,candidate)
    with pytest.raises(ValueError):refine_weak_ca_delay(iq,cfg,replace(candidate,excluded_period=20))
