from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaAcquisition, CaSearch, acquire_ca_signal
from gnss_gpu.carrier_support_cpu import assess_ca_carrier_support
from gnss_gpu.code_delay_cpu import CodeDelayEstimate
from gnss_gpu.tracking_cpu import refine_ca_carrier
from tests.test_tracking_cpu import observation, FS
from experiments.audit_cpu_carrier_support import audit
from experiments.run_cpu_batch_receiver import json_default


def candidate(frequency):
    return CaAcquisition(frequency, CodeDelayEstimate(0., FS, 0, 1j, .1), 10.)


@pytest.mark.parametrize("frequency,side", [(-500., "lower"), (-499.875, "lower"),
                                           (499.875, "upper"), (500., "upper")])
def test_inclusive_margin_and_both_endpoints(frequency, side):
    result = assess_ca_carrier_support(candidate(frequency), CaSearch(FS, (-500., 0., 500.)), boundary_margin_hz=.125)
    assert result.status == "boundary"
    assert result.boundary_side == side


def test_interior_and_overlapping_margin_are_explicit():
    search = CaSearch(FS, (-500., 0., 500.))
    assert assess_ca_carrier_support(candidate(170.), search).status == "interior"
    result = assess_ca_carrier_support(candidate(0.), search, boundary_margin_hz=500.)
    assert result.boundary_side == "both"


def test_zero_margin_missing_estimate_and_single_bin_are_distinct():
    search = CaSearch(FS, (-500., 0., 500.))
    assert assess_ca_carrier_support(candidate(500.), search, boundary_margin_hz=0.).status == "boundary"
    assert assess_ca_carrier_support(candidate(499.999), search, boundary_margin_hz=0.).status == "interior"
    assert assess_ca_carrier_support(None, search).status == "not_estimated"
    fixed = assess_ca_carrier_support(candidate(170.), CaSearch(FS, (170.,)))
    assert fixed.status == "fixed_frequency" and fixed.boundary_side is None


@pytest.mark.parametrize("margin", [-1., np.inf, np.nan])
def test_invalid_margin_is_rejected_even_without_candidate(margin):
    with pytest.raises(ValueError):
        assess_ca_carrier_support(None, CaSearch(FS, (0.,)), boundary_margin_hz=margin)


@pytest.mark.parametrize("frequency", [-501., 501., np.nan])
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError):
        assess_ca_carrier_support(candidate(frequency), CaSearch(FS, (-500., 500.)))


def test_sampling_mismatch_is_rejected():
    other = replace(candidate(0.), delay=replace(candidate(0.).delay, sampling_freq=2*FS))
    with pytest.raises(ValueError, match="sampling"):
        assess_ca_carrier_support(other, CaSearch(FS, (-500., 500.)))


@pytest.mark.parametrize("frequency,expected", [(-700., "boundary"), (-500., "boundary"),
                                               (-170., "interior"), (170., "interior"),
                                               (500., "boundary"), (700., "boundary")])
def test_physical_waveform_checks_include_true_edges_and_outside_grid(frequency, expected):
    search = CaSearch(FS, (-500., 0., 500.))
    iq = observation(frequency)
    coarse = acquire_ca_signal(iq, search)
    refined = refine_ca_carrier(iq, search, coarse)
    saved = asdict(refined)
    result = assess_ca_carrier_support(refined, search)
    assert result.status == expected
    assert asdict(refined) == saved
    if expected == "boundary":
        assert result.boundary_side == ("lower" if frequency < 0 else "upper")


def recording_fixture(directory):
    directory.mkdir()
    metadata = dict(threshold=dict(search=asdict(CaSearch(FS, (-500., 0., 500.)))),
                    iq_sha256={"0": "unused-by-diagnostic"}, blocks=3)
    trace = [dict(start_sample=i*2046, status="tracking" if frequency is not None else "coasting",
                  measurement_used=frequency is not None, carrier_hz=frequency,
                  refined_candidate=asdict(candidate(frequency)) if frequency is not None else None)
             for i, frequency in enumerate((500., 170., None))]
    record = dict(condition=0, nominal_primary_cn0_db_hz=55., primary_doppler_hz=700.,
                  outage_ms=2, replica=0, trace=trace)
    (directory/"metadata.json").write_text(json.dumps(metadata))
    (directory/"traces.jsonl").write_text(json.dumps(record, default=json_default)+"\n")
    return record


def test_saved_trace_audit_preserves_inputs_and_distinguishes_predictions(tmp_path):
    source = tmp_path/"source"
    recording_fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    report, rows = audit(source, tmp_path/"output")
    assert report["blocks"] == 3
    assert rows[0]["used_blocks"] == 2
    assert rows[0]["boundary_used_blocks"] == 1
    assert rows[0]["interior_used_blocks"] == 1
    assert rows[0]["boundary_conditional_carrier_mae_hz"] == 200.
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    annotations = json.loads((tmp_path/"output"/"diagnostics.jsonl").read_text())["blocks"]
    assert annotations[-1]["support"]["status"] == "not_estimated"


def test_saved_trace_audit_rejects_duplicate_scenarios_before_writing(tmp_path):
    source = tmp_path/"source"
    recording_fixture(source)
    path = source/"traces.jsonl"
    path.write_text(path.read_text()*2)
    with pytest.raises(ValueError, match="duplicate"):
        audit(source, tmp_path/"output")
    assert not (tmp_path/"output").exists()
