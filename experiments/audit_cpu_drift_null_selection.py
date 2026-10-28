import json
from dataclasses import asdict
from experiments.run_cpu_drift_null_selection import OUTPUT,OLD,RATES,digest,evaluate,summarize,tasks,apply_gate,PathSelectionGate,calibrate_path_gate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(OLD)==p['old_gate_sha256']
    for key,name in [('calibration','calibration.jsonl'),('evaluation','evaluation.jsonl'),('gate','frozen_gates.json'),('report','report.json')]:assert digest(OUTPUT/name)==c[key+'_sha256']
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==c['calibration']==299 and len(rows)==c['evaluation']==192
    ident=lambda r:[r[k] for k in ('block_count','drift_hz_s','initial_carrier_hz','replica')]
    assert [ident(r) for r in cal]==p['calibration_tasks']==json.loads(json.dumps([(40,0.,o,i) for _,o,i in tasks(0.,940000,299)]))
    expected=[(count,rate,o,i) for count in (10,40) for rate in RATES for _,o,i in tasks(0.,950000,32)]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    assert len({r['replica'] for r in cal})==299
    assert {r['replica'] for r in cal}.isdisjoint(r['replica'] for r in rows)
    oldrows=[json.loads(x) for x in (OLD.parent/'calibration.jsonl').read_text().splitlines()]
    assert {r['replica'] for r in oldrows}.isdisjoint(r['replica'] for r in cal+rows)
    gates={'10':json.loads(OLD.read_text()),'40':asdict(calibrate_path_gate([r['uncalibrated_score'] for r in cal]))}
    assert gates==json.loads((OUTPUT/'frozen_gates.json').read_text())
    for row in rows:
        raw={k:v for k,v in row.items() if k not in ('selected','error_m','success','score_exceeds')}
        assert apply_gate(raw,PathSelectionGate(**gates[str(row['block_count'])]))==row
    for rate in RATES:
        for replica in range(950000,950032):
            g=[r for r in rows if r['drift_hz_s']==rate and r['replica']==replica]
            assert len(g)==2 and len({r['first10_sha256'] for r in g})==1
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,298):assert json.loads(json.dumps(evaluate(p['calibration_tasks'][i])))==cal[i]
    for i in range(0,192,32):
        raw=evaluate(p['evaluation_tasks'][i])
        assert json.loads(json.dumps(apply_gate(raw,PathSelectionGate(**gates[str(raw['block_count'])]))))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=299,evaluation=192,frozen_sources=True,old_gate_preserved=True,disjoint_ids=True,identical_prefixes=True,gate_choices_summary_recomputed=True,replayed_calibration=[0,298],replayed_evaluation=list(range(0,192,32))),indent=2))
    print('Audit passed: gates,192 choices, identical prefixes and eight full replays')


if __name__=='__main__':run()
