import json
from experiments.run_cpu_residual_ablation import OUTPUT,PRIOR,digest,calibrate,apply_gates,summarize,evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for key,name in [('evaluation','evaluation.jsonl'),('report','report.json'),('gate','frozen_gates.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    gates=calibrate(cal);assert gates==json.loads((OUTPUT/'frozen_gates.json').read_text())
    assert len(cal)==c['reused_calibration']==1224
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==len(p['evaluation_tasks'])==c['evaluation']==576
    assert {r['replica'] for r in cal}.isdisjoint({r['replica'] for r in rows})
    for row,task in zip(rows,p['evaluation_tasks']):
        assert [row[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')]==task
        assert apply_gates({k:v for k,v in row.items() if k!='outcomes'},gates)==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,128,256,384,448,512):
        assert json.loads(json.dumps(apply_gates(evaluate(p['evaluation_tasks'][i]),gates)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(reused_calibration=1224,evaluation=576,frozen_sources_and_prior=True,all_gates_choices_report_recomputed=True,replayed_indices=[0,128,256,384,448,512]),indent=2))
    print('Audit passed: 576 fresh observations, four arms, six full replays')

if __name__=='__main__':run()
