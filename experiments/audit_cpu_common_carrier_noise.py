import json
from experiments.stress_cpu_common_carrier_noise import OUTPUT,PRIOR,PROFILES,evaluate,apply_gate,summarize,digest,PathSelectionGate,tasks


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']==c['gate_sha256']
    assert json.loads(open(p['gate_path']).read())==p['gate']
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['evaluation']==288
    ident=lambda r:[r[k] for k in ('profile','echo_voltage','residual_carrier_hz','replica')]
    expected=[(profile,*t) for profile in PROFILES for t in tasks(0.,840000,64)+tasks(.8,850000,32)]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    previous=[]
    for name in ('calibration.jsonl','evaluation.jsonl'):
        previous.extend(json.loads(x) for x in (PRIOR/name).read_text().splitlines())
    assert {r['replica'] for r in previous}.isdisjoint({r['replica'] for r in rows})
    gate=PathSelectionGate(**p['gate'])
    for r in rows:
        raw={k:v for k,v in r.items() if k not in ('selected','error_m','success','score_exceeds')}
        assert apply_gate(raw,gate)==r
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[0,64,96,160,192,256]
    for i in indices:assert json.loads(json.dumps(apply_gate(evaluate(p['evaluation_tasks'][i]),gate)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=288,frozen_sources_and_gate=True,
        prior_ids_disjoint=True,all_choices_and_summary_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed: 288 choices, six full replays, frozen gate/sources and disjoint IDs')


if __name__=='__main__':run()
