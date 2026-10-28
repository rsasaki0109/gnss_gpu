import json
from experiments.run_cpu_common_carrier_selection import OUTPUT,PRIOR,digest,evaluate,apply_gate,summarize,tasks,calibrate_path_gate,PathSelectionGate
from dataclasses import asdict


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('gate','frozen_gate.json'),('report','report.json')]:
        assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==299 and len(rows)==c['evaluation']==192
    ident=lambda r:[r[k] for k in ('echo_voltage','residual_carrier_hz','replica')]
    assert [ident(r) for r in cal]==p['calibration_tasks']==json.loads(json.dumps(tasks(0.,800000,299)))
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(tasks(0.,810000,128)+tasks(.8,820000,64)))
    development=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    groups=[{r['replica'] for r in g} for g in (cal,rows,development)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert len(groups[0])==299 and len(groups[1])==192
    gate=calibrate_path_gate([r['uncalibrated_score'] for r in cal])
    assert asdict(gate)==json.loads((OUTPUT/'frozen_gate.json').read_text())
    for r in rows:
        raw={k:v for k,v in r.items() if k not in ('selected','error_m','success','score_exceeds')}
        assert apply_gate(raw,gate)==r
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,149,298):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in (0,127,128,191):assert json.loads(json.dumps(apply_gate(evaluate(p['evaluation_tasks'][i]),gate)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=299,evaluation=192,disjoint_ids=True,
        source_hashes_verified=True,gate_choices_report_recomputed=True,replayed_calibration=[0,149,298],replayed_evaluation=[0,127,128,191]),indent=2))
    print('Audit passed: frozen sources, disjoint IDs, recomputed gate and 192 choices, seven full replays')


if __name__=='__main__':run()
