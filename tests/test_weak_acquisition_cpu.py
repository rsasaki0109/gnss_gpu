from dataclasses import replace

import numpy as np
import pytest
import json
import hashlib
from dataclasses import asdict

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig,cancel_other_ca
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch,detect_ca_bit_edge
from gnss_gpu.weak_acquisition_cpu import calibrate_weak_ca,detect_weak_ca,load_weak_ca_experiment,_model_hashes
from experiments.solve_cpu_weak_acquisition import summarize_signals
from tests.test_ca_cancellation_cpu import observation


@pytest.mark.parametrize('coarse_candidates',[1,3])
def test_pipeline_calibrates_streamed_nulls_and_matches_manual_detection(tmp_path,coarse_candidates):
    rng=np.random.default_rng(709)
    cancellation=CaCancellationConfig(2046000.,nuisance_prns=())
    search=CaBitEdgeSearch(CaSearch(2046000.,(0.,)),periods=4,residual_grid_hz=(0.,),shortlist_size=8)
    records=(rng.normal(size=(4,4092)) for _ in range(499))
    gate=calibrate_weak_ca({'white':records},cancellation,search,coarse_candidates=coarse_candidates)
    assert gate.coarse_candidates==coarse_candidates
    iq=rng.normal(size=(4,4092))
    output=detect_weak_ca(iq,gate)
    manual=detect_ca_bit_edge(cancel_other_ca(iq,cancellation).iq,gate.detection)
    assert output.detection==manual
    assert not detect_weak_ca(np.zeros((4,4092)),gate).detection.accepted
    with pytest.raises(ValueError,match='match'):
        replace(gate,cancellation=CaCancellationConfig(4092000.,nuisance_prns=()))
    method='multiseed_20' if coarse_candidates>1 else 'bit_edge_20'
    protocol=dict(detectors={method:asdict(search)},cancellation=asdict(cancellation),
                  local_delay_radius_samples=1,source_sha256=_model_hashes(coarse_candidates))
    if coarse_candidates>1:protocol['coarse_candidates']=coarse_candidates
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'frozen_thresholds.json').write_text(json.dumps({method:asdict(gate.detection)}))
    report={key:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name,key in
            (('protocol.json','protocol_sha256'),('frozen_thresholds.json','frozen_thresholds_sha256'))}
    (tmp_path/'report.json').write_text(json.dumps(report))
    loaded=load_weak_ca_experiment(tmp_path)
    assert loaded.coarse_candidates==coarse_candidates
    assert detect_weak_ca(iq,loaded).detection==output.detection
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        load_weak_ca_experiment(tmp_path)


def test_cancellation_preserves_controlled_target_bit_edge_for_coherent_capture():
    from gnss_gpu.bit_edge_acquisition_cpu import acquire_ca_bit_edge
    target=.1*observation(1,175.,201)
    target[7,1000:]*=-1;target[8:]*=-1
    residual=cancel_other_ca(2*observation()+target,CaCancellationConfig(2046000.,nuisance_prns=(2,)))
    search=CaBitEdgeSearch(CaSearch(2046000.,(-250.,0.,250.)))
    candidate=acquire_ca_bit_edge(residual.iq,search)
    assert candidate.carrier_hz==175. and abs(candidate.delay_samples-201)<=1
    assert candidate.excluded_period==7


def test_summary_retains_failed_replica_in_both_methods():
    rows=[dict(profile='near_far_20db',nominal_primary_cn0_db_hz=37.,primary_doppler_hz=170.,
        methods={method:dict(accepted=flag,passes_declared_criterion=flag) for method in ('noncoherent_4','bit_edge_20')}) for flag in (True,False)]
    summary=summarize_signals(rows)
    assert len(summary)==2
    assert all(row['replicas']==2 and row['criterion_fraction']==.5 for row in summary)
