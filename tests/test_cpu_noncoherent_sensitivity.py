from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentCandidate, CaNoncoherentDetection
from experiments.run_cpu_noncoherent_sensitivity import score_detection, aggregate


def detection(carrier=500., lag=37, accepted=True):
    return CaNoncoherentDetection(accepted, 30., 20., CaNoncoherentCandidate(carrier, lag, 0, 30.))


def test_coarse_grid_criterion_checks_both_carrier_and_wrapped_code_lag():
    assert score_detection(detection(), 700., 37.)["passes_declared_criterion"]
    assert score_detection(detection(lag=37-2046), 700., 37.)["passes_declared_criterion"]
    assert not score_detection(detection(lag=100), 700., 37.)["passes_declared_criterion"]
    assert not score_detection(detection(carrier=0.), 700., 37.)["passes_declared_criterion"]
    assert not score_detection(detection(accepted=False), 700., 37.)["passes_declared_criterion"]
    assert not score_detection(CaNoncoherentDetection(False, 0., 20., None), 700., 37.)["passes_declared_criterion"]


def test_aggregation_retains_all_replicas_for_both_matched_detectors():
    a, b = score_detection(detection(), 700., 37.), score_detection(detection(lag=100), 700., 37.)
    rows = [dict(nominal_primary_cn0_db_hz=35., primary_doppler_hz=700., detectors={"1": a, "4": b}),
            dict(nominal_primary_cn0_db_hz=35., primary_doppler_hz=700., detectors={"1": b, "4": a})]
    result = aggregate(rows)
    assert len(result) == 2
    assert all(row["replicas"] == 2 and row["criterion_fraction"] == .5 for row in result)
