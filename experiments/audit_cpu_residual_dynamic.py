import json
from experiments.run_cpu_residual_dynamic import OUTPUT,digest,evaluate,calibrate,apply_gates,summarize
from experiments.stress_cpu_dynamic_noise import evaluate as old_evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['old_gate_path'])==p['old_gate_sha256']
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('report','report.json'),('gate','frozen_gate.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==1224 and len(rows)==c['evaluation']==528
    ident=lambda r:[r[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')]
    assert [ident(r) for r in cal]==p['calibration_tasks']
    assert [ident(r) for r in rows]==p['evaluation_tasks']
    assert len({r['replica'] for r in cal})==1224
    assert {r['replica'] for r in cal}.isdisjoint({r['replica'] for r in rows})
    gates=calibrate(cal);assert gates==json.loads((OUTPUT/'frozen_gate.json').read_text())
    for row in rows:
        assert apply_gates({k:v for k,v in row.items() if k!='outcomes'},gates['combined'],p['old_gate'])==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,408,816):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,128,256,384,527):
        assert json.loads(json.dumps(apply_gates(evaluate(p['evaluation_tasks'][i]),gates['combined'],p['old_gate'])))==rows[i]
    for i in (0,384):
        old=json.loads(json.dumps(old_evaluate(p['evaluation_tasks'][i])))
        assert old['score']==rows[i]['old_score'] and old['observation_sha256']==rows[i]['observation_sha256']
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=1224,evaluation=528,frozen_sources_and_gates=True,gate_and_choices_recomputed=True,old_baseline_reproduced=True,replayed_calibration=[0,408,816],replayed_evaluation=[0,128,256,384,527]),indent=2))
    print('Audit passed: full calibration/evaluation, eight replays, old baseline identical')

if __name__=='__main__':run()
