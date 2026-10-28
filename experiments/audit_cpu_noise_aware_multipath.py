"""Audit stratified gate, all model choices and limited deterministic replay."""
import hashlib,json
from pathlib import Path
from experiments.run_cpu_noise_aware_multipath import OUTPUT,OLD_GATE,PROFILES,evaluate,select,summarize
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gates=json.loads((OUTPUT/'frozen_gate.json').read_text())
    gate=PathSelectionGate(**gates['combined'])
    old_gate=PathSelectionGate(**json.loads(OLD_GATE.read_text()))
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(cal)==1224 and len(rows)==768 and 3*.99**408<=.05
    for p in PROFILES:
        group=[r for r in cal if r['profile']==p]
        assert len(group)==gates['strata'][p]['calibration_count']==408
        assert max(r['fit']['bic_improvement'] for r in group)==gates['strata'][p]['score']
    assert gate.score==max(g['score'] for g in gates['strata'].values())
    assert hashlib.sha256((OUTPUT/'frozen_gate.json').read_bytes()).hexdigest()==completion['frozen_gate_sha256']
    for p,h in {**protocol['source_sha256'],**protocol['prior_sha256']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    keys=('profile','echo_voltage','gap_chips','phase_rad','replica')
    for records,tasks in ((cal,protocol['calibration_tasks']),(rows,protocol['evaluation_tasks'])):
        identities=[tuple(r[k] for k in keys) for r in records]
        assert identities==[tuple(t[:5]) for t in tasks] and len(set(identities))==len(records)
    assert not set(r['replica'] for r in cal)&set(r['replica'] for r in rows)
    assert not set(r['replica'] for r in rows if r['echo_voltage']==0)&set(r['replica'] for r in rows if r['echo_voltage']!=0)
    for row in rows:
        raw={k:v for k,v in row.items() if k not in ('new_selected','old_selected','new_error_m','old_error_m','new_score_pass')}
        assert json.loads(json.dumps(select(raw,gate,old_gate)))==row
    report=json.loads((OUTPUT/'report.json').read_text())
    assert summarize(rows)==report and len(report)==9
    assert sum(r['primary'] for r in report)==2
    for cell in report:
        group=[r for r in rows if all(r[k]==cell[k] for k in ('profile','echo_voltage','phase_rad'))]
        assert len(group)==cell['attempts']==(128 if cell['echo_voltage']==0 else 64)
        assert sum(r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in group)==cell['selected_successes']
    for i in (0,408,816):
        assert json.loads(json.dumps(evaluate(protocol['calibration_tasks'][i])))==cal[i]
    for i in (128,767):
        assert json.loads(json.dumps(select(evaluate(protocol['evaluation_tasks'][i]),gate,old_gate)))==rows[i]
    audit=dict(calibration_records=1224,evaluation_records=768,strata=3,
        all_stratum_maxima_and_joint_bound_verified=True,all_selections_reports_and_hashes_verified=True,
        exact_calibration_replay_indices=[0,408,816],exact_evaluation_replay_indices=[128,767],
        limitation='Same-model replay; finite specified noise strata at fixed power and coherence')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
