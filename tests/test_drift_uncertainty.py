from experiments.validate_cpu_drift_uncertainty import uncertainty, summarize, Z
from gnss_gpu.multipath_information_cpu import CHIP_RATE, LIGHT_SPEED


def row(error, sigma, model='single_path'):
    r = dict(kind='echo', echo_voltage=.8, selected=dict(model=model),
             fits={k:dict(first_std_chips=sigma*CHIP_RATE/LIGHT_SPEED) for k in ('single','two')},
             error_m=error, success=error is not None and abs(error)<10)
    r['uncertainty'] = uncertainty(r)
    return r


def test_misspecified_precise_fit_and_unavailable():
    r = row(-12, 1)
    assert r['uncertainty']['narrow_large_error']
    assert not r['uncertainty']['covers_truth']
    r = row(None, 1, 'ambiguous')
    assert not r['uncertainty']['available']
    assert r['uncertainty']['sigma_m'] is None
    assert row(1, 2, 'two_path')['uncertainty']['covers_truth']
    assert not row(12, 10)['uncertainty']['narrow_large_error']


def test_summary_keeps_unavailable_in_success_denominator():
    rows = [row(-12, 1), row(1, 2, 'two_path'), row(None, 1, 'ambiguous')]
    c = next(c for c in summarize(rows)['cells'] if c['echo_voltage']==.8 and c['model']=='all')
    assert (c['attempts'],c['available'],c['covered'],c['successes'],c['narrow_large_error']) == (3,2,1,1,1)
    assert c['coverage_available_exact_95'] is not None
