from types import SimpleNamespace

from experiments.run_cpu_ca_cancellation import cancellation_metadata, strict_detection, CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentCandidate


def test_audit_metadata_does_not_serialize_iq_arrays():
    result = SimpleNamespace(status="projected", removed_energy_fraction=.5, gram_condition=1., components=(), iq=object())
    assert cancellation_metadata(result) == dict(status="projected", removed_energy_fraction=.5, gram_condition=1., components=[])


def test_strict_gate_and_fresh_domains():
    candidate = CaNoncoherentCandidate(0., 0, 0, 20.)
    assert strict_detection(candidate, 19.).accepted
    assert not strict_detection(candidate, 20.).accepted
    assert not strict_detection(None, 20.).accepted
    assert len({CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN}) == 3
