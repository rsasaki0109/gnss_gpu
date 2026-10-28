from dataclasses import replace
import numpy as np
import pytest
from gnss_gpu.bandlimited_multipath_v2_cpu import BandlimitedFit
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate, _select_default_fit


def fixture():
    return BandlimitedFit(.2,.1,.1,.2,'single_path','BIC prefers one path',-5.,.02,(1.,.8),True)


def test_order_statistic_size_and_strict_exceedance():
    with pytest.raises(ValueError): calibrate_path_gate(np.arange(298))
    gate=calibrate_path_gate(np.full(299,-5.))
    assert .99**gate.calibration_count <= .05
    assert _select_default_fit(fixture(),gate).selected_model=='single_path'
    promoted=_select_default_fit(replace(fixture(),bic_improvement=-4.),gate)
    assert promoted.selected_model=='two_path' and promoted.selected_delay_chips==.1


@pytest.mark.parametrize('change', [dict(converged=False),dict(first_delay_chips=1.),
    dict(excess_chips=.01),dict(excess_chips=1.2),dict(first_std_chips=None),dict(first_std_chips=.2)])
def test_calibrated_gate_cannot_bypass_local_guards(change):
    gate=calibrate_path_gate(np.full(299,-10.))
    result=_select_default_fit(replace(fixture(),**change),gate)
    assert result.selected_model=='ambiguous' and result.selected_delay_chips is None


@pytest.mark.parametrize('scores', [np.full(299,np.nan),np.full(299,np.inf),np.zeros((299,1))])
def test_invalid_calibration(scores):
    with pytest.raises(ValueError): calibrate_path_gate(scores)
