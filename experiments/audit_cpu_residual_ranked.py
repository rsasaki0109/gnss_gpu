import json
from experiments.run_cpu_residual_ranked import OUTPUT,digest,calibrate,apply_gates,summarize,evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('report','report.json'),('gate','frozen_gates.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==1920 and len(rows)==c['evaluation']==576
    ident=lambda r:[r[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')]
    assert [ident(r) for r in cal]==p['calibration_tasks']
    assert [ident(r) for r in rows]==p['evaluation_tasks']
    assert len({r['replica'] for r in cal})==1920
    assert {r['replica'] for r in cal}.isdisjoint({r['replica'] for r in rows})
    result=calibrate(cal);assert result==json.loads((OUTPUT/'frozen_gates.json').read_text())
    assert all(s['ranked']['rank_from_largest']==p['rank_from_largest']==2 for s in result['strata'].values())
    gates=result['gates']
    for row in rows:assert apply_gates({k:v for k,v in row.items() if k!='outcomes'},gates)==row
    assert summarize(rows,gates)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,640,1280):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,128,256,384,448,512):assert json.loads(json.dumps(apply_gates(evaluate(p['evaluation_tasks'][i]),gates)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=1920,evaluation=576,frozen_sources=True,rank_gates_choices_report_recomputed=True,replayed_calibration=[0,640,1280],replayed_evaluation=[0,128,256,384,448,512]),indent=2))
    print('Audit passed: 1920 calibration +576 evaluation; nine complete replays')

if __name__=='__main__':run()
