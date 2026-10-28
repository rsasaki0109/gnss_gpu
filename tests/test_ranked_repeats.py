from dataclasses import asdict
from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_ranked_repeats import apply_repeats,summarize_repeats


def fixture():
    gates={str(r):dict(combined={m:asdict(PathSelectionGate(score+r,3000,.01,.05))
           for m,score in [('maximum',0.),('ranked',-10.)]}) for r in range(3)}
    fit=BandlimitedFit(.2,.1,.1,.2,'single_path','initial',-5.,.02,(1.,.8),True)
    raw=dict(fit=asdict(fit),true_delay_chips=.1,profile='white',echo_voltage=.8)
    return gates,[apply_repeats(dict(raw,replica=r),gates) for r in range(2)]


def test_each_draw_keeps_shared_denominator_and_pairing():
    gates,rows=fixture()
    report=summarize_repeats(rows,gates)
    assert len(report['cells'])==6 and len(report['contrasts'])==3
    assert all(c['attempts']==2 for c in report['cells'])
    assert all(c['gains']==2 and c['losses']==0 for c in report['contrasts'])


def test_observed_ranges_are_computed_across_draws_not_observations():
    gates,rows=fixture()
    report=summarize_repeats(rows,gates)
    assert report['descriptive_variability']['maximum']['threshold_range']==2.
    assert report['descriptive_variability']['ranked']['success_count_range']==0
