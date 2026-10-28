import json
from experiments.run_cpu_residual_repeat import OUTPUT,PRIOR,digest,calibrate,apply_draws,summarize,evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for key,name in [('new_calibration','new_calibration.jsonl'),('evaluation','evaluation.jsonl'),('report','report.json'),('gate','frozen_draws.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    old=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    cal=[json.loads(x) for x in (OUTPUT/'new_calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(old)==c['reused_calibration']==1920 and len(cal)==c['new_calibration']==1920
    assert len(rows)==c['evaluation']==576
    ident=lambda r:[r[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')]
    assert [ident(r) for r in cal]==p['new_calibration_tasks']
    assert [ident(r) for r in rows]==p['evaluation_tasks']
    assert len({r['replica'] for r in cal})==1920
    groups=[{r['replica'] for r in group} for group in (old,cal,rows)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    draws={'original':calibrate(old),'new':calibrate(cal)}
    assert draws==json.loads((OUTPUT/'frozen_draws.json').read_text())
    assert draws['original']==json.loads((PRIOR/'frozen_gates.json').read_text())
    for row in rows:assert apply_draws({k:v for k,v in row.items() if k!='outcomes'},draws)==row
    assert summarize(rows,draws)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,640,1280):assert json.loads(json.dumps(evaluate(p['new_calibration_tasks'][i])))==cal[i]
    for i in (0,128,256,384,448,512):assert json.loads(json.dumps(apply_draws(evaluate(p['evaluation_tasks'][i]),draws)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(new_calibration=1920,reused_calibration=1920,evaluation=576,independent_calibration_ids=True,original_gates_preserved=True,all_gates_choices_report_recomputed=True,replayed_calibration=[0,640,1280],replayed_evaluation=[0,128,256,384,448,512]),indent=2))
    print('Audit passed: independent calibration, preserved original gates, nine replays')

if __name__=='__main__':run()
