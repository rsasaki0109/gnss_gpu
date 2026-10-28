from dataclasses import asdict
import json
from experiments.run_cpu_dynamic_selection import OUTPUT,digest,evaluate,select,summarize,calibrate_path_gate,PathSelectionGate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('report','report.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    assert digest(OUTPUT/'frozen_gate.json')==c['gate_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==299 and len(rows)==c['evaluation']==539
    identity=lambda r:[r['echo_voltage'],r['doppler_hz'],r['phase_rad'],r['replica']]
    assert [identity(r) for r in cal]==p['calibration_tasks']
    assert [identity(r) for r in rows]==p['evaluation_tasks']
    assert {r['replica'] for r in cal}.isdisjoint({r['replica'] for r in rows})
    gate=calibrate_path_gate([r['score']['bic_improvement'] for r in cal])
    assert asdict(gate)==json.loads((OUTPUT/'frozen_gate.json').read_text())
    for row in rows:
        raw={k:row[k] for k in ('echo_voltage','doppler_hz','phase_rad','replica','true_delay_chips','score','observation_sha256')}
        assert select(raw,gate)==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,298):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,299,538):assert json.loads(json.dumps(select(evaluate(p['evaluation_tasks'][i]),gate)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=299,evaluation=539,frozen_sources_verified=True,gate_recomputed=True,all_selections_recomputed=True,replayed_calibration=[0,298],replayed_evaluation=[0,299,538]),indent=2))
    print('Audit passed: 299 calibration +539 evaluation, five full replays')

if __name__=='__main__':run()
