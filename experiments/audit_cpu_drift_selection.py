import json
from dataclasses import asdict
from experiments.run_cpu_drift_selection import OUTPUT,PRIOR,digest,tasks,evaluate,apply_gate,summarize,calibrate_path_gate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('gate','frozen_gate.json'),('report','report.json')]:assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==299 and len(rows)==c['evaluation']==96
    ident=lambda r:[r[k] for k in ('kind','drift_hz_s','initial_carrier_hz','replica')]
    assert [ident(r) for r in cal]==p['calibration_tasks']==json.loads(json.dumps(tasks('null',980000,299)))
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(tasks('null',990000,64)+tasks('echo',1000000,32)))
    prior=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    ids=[{r['replica'] for r in group} for group in (cal,rows,prior)]
    assert len(ids[0])==299 and len(ids[1])==96
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i))
    gate=calibrate_path_gate([r['uncalibrated_score'] for r in cal])
    assert asdict(gate)==json.loads((OUTPUT/'frozen_gate.json').read_text())
    for r in rows:
        raw={k:v for k,v in r.items() if k not in ('selected','error_m','success','score_exceeds')}
        assert apply_gate(raw,gate)==r
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,298):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,63,64,95):assert json.loads(json.dumps(apply_gate(evaluate(p['evaluation_tasks'][i]),gate)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=299,evaluation=96,source_hashes=True,disjoint_ids=True,
        gate_choices_summary_recomputed=True,replayed_calibration=[0,298],replayed_evaluation=[0,63,64,95]),indent=2))
    print('Audit passed: gate,96 choices, disjoint IDs and six full replays')


if __name__=='__main__':run()
