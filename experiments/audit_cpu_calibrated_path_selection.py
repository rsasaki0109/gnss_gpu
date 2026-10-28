"""Verify calibration order statistic, disjoint records and held-out metrics."""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from experiments.run_cpu_calibrated_path_selection import OUTPUT, evaluate, apply_gate, summarize
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate, fit_calibrated_paths
from experiments.run_cpu_coherent_multipath import synthesize_blocks


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gate=PathSelectionGate(**json.loads((OUTPUT/'frozen_gate.json').read_text()))
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(cal)==gate.calibration_count==299 and len(rows)==811
    assert gate.score==max(r['fit']['bic_improvement'] for r in cal)
    assert (1-gate.tail_budget)**len(cal)<=gate.confidence_failure
    assert hashlib.sha256((OUTPUT/'frozen_gate.json').read_bytes()).hexdigest()==completion['frozen_gate_sha256']
    for path,digest in {**protocol['source_sha256'],**protocol['prior_artifact_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    keys=('echo_voltage','gap_chips','phase_rad','replica')
    identities=lambda group:[tuple(r[k] for k in keys) for r in group]
    assert identities(cal)==[tuple(t) for t in protocol['calibration_tasks']]
    assert identities(rows)==[tuple(t) for t in protocol['null_evaluation_tasks']+protocol['echo_tasks']]
    assert len(set(identities(rows)))==811
    assert not set(r['replica'] for r in cal)&set(r['replica'] for r in rows)
    assert not set(r['replica'] for r in rows if r['echo_voltage']==0)&set(r['replica'] for r in rows if r['echo_voltage']!=0)
    for row in rows:
        recalculated=apply_gate({k:v for k,v in row.items() if k not in
            ('calibrated_fit','old_error_m','new_error_m','score_exceeds_gate')},gate)
        assert json.loads(json.dumps(recalculated))==row
        if row['calibrated_fit']['selected_model']=='two_path':
            assert row['score_exceeds_gate'] and row['calibrated_fit']['converged']
    report=json.loads((OUTPUT/'report.json').read_text())
    assert summarize(rows)==report and len(report)==5
    for cell in report:
        selected=[r for r in rows if all(r[k]==cell[k] for k in keys[:3])]
        assert len(selected)==cell['attempts']==(299 if cell['echo_voltage']==0 else 128)
        assert sum(r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in selected)==cell['selected_successes']
    assert json.loads(json.dumps(evaluate(protocol['calibration_tasks'][0])))==cal[0]
    task=protocol['echo_tasks'][-1]
    assert json.loads(json.dumps(apply_gate(evaluate(task),gate)))==rows[-1]
    _,blocks,_=synthesize_blocks(3.6,*task)
    assert json.loads(json.dumps(asdict(fit_calibrated_paths(blocks.mean(axis=0),gate))))==rows[-1]['calibrated_fit']
    audit=dict(calibration_records=299,evaluation_records=811,independent_echo_replicas=128,
        threshold_recomputed=True,all_sources_and_prior_artifacts_unchanged=True,
        disjoint_replica_groups=True,all_selections_and_reports_recomputed=True,
        exact_replays=['first calibration','last echo including public API'],
        scope='Same-model conditional calibration; no RF/acquisition/PVT or distribution-shift guarantee')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
