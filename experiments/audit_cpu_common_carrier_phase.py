import json
from experiments.stress_cpu_common_carrier_phase import OUTPUT,PRIOR,digest,evaluate,apply_gate,summarize,PathSelectionGate,CONDITIONS,tasks


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']==c['gate_sha256']
    assert json.loads(open(p['gate_path']).read())==p['gate']
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['evaluation']==176
    expected=[(r,o,.7,0.,i) for r,o,i in tasks(0.,870000,32)]+[(r,o,fd,phase,i) for fd,phase in CONDITIONS for r,o,i in tasks(.8,880000,16)]
    ident=lambda r:[r[k] for k in ('echo_voltage','residual_carrier_hz','relative_doppler_hz','phase_rad','replica')]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    previous=[]
    for name in ('calibration.jsonl','evaluation.jsonl'):
        previous.extend(json.loads(x) for x in (PRIOR/name).read_text().splitlines())
    assert {r['replica'] for r in rows}.isdisjoint(r['replica'] for r in previous)
    gate=PathSelectionGate(**p['gate'])
    for r in rows:assert apply_gate({k:v for k,v in r.items() if k not in ('selected','error_m','success','score_exceeds')},gate)==r
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[0]+list(range(32,176,16))
    for i in indices:assert json.loads(json.dumps(apply_gate(evaluate(p['evaluation_tasks'][i]),gate)))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=176,sources_and_gate_preserved=True,prior_ids_disjoint=True,all_choices_and_summary_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed: 176 choices, ten full replays, frozen sources and gate')


if __name__=='__main__':run()
