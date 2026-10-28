import json
from experiments.run_cpu_dynamic_ranked_gate import OUTPUT,digest,calibrate,apply_gates,summarize,evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['prior_gate'])==p['prior_gate_sha256']
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('report','report.json'),('gate','frozen_gates.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==1000 and len(rows)==c['evaluation']==683
    identity=lambda r:[r['echo_voltage'],r['doppler_hz'],r['phase_rad'],r['replica']]
    assert [identity(r) for r in cal]==p['calibration_tasks']
    assert [identity(r) for r in rows]==p['evaluation_tasks']
    assert {r['replica'] for r in cal}.isdisjoint({r['replica'] for r in rows})
    gates,rank=calibrate(cal)
    from pathlib import Path
    gates['prior299']=json.loads(Path(p['prior_gate']).read_text())
    assert dict(gates=gates,rank=rank)==json.loads((OUTPUT/'frozen_gates.json').read_text())
    assert rank['rank_from_largest']==p['rank_from_largest']==5
    for row in rows:
        raw={k:v for k,v in row.items() if k!='outcomes'}
        assert apply_gates(raw,gates)==row
    assert summarize(rows,gates)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,999):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,299,682):assert json.loads(json.dumps(apply_gates(evaluate(p['evaluation_tasks'][i]),gates)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=1000,evaluation=683,frozen_sources_and_gates=True,rank_recomputed=True,all_choices_and_report_recomputed=True,replayed_calibration=[0,999],replayed_evaluation=[0,299,682]),indent=2))
    print('Audit passed: 1000 calibration +683 evaluation; five full replays')

if __name__=='__main__':run()
