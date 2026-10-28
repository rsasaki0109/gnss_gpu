import json
from experiments.run_cpu_carrier_duration import OUTPUT,PRIOR,digest,evaluate,summarize,tasks


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['evaluation']==72
    expected=[(count,fd,offset,i) for fd in (0.,.1,.7) for count in (10,20,40) for _,offset,i in tasks(.8,900000,8)]
    ident=lambda r:[r[k] for k in ('block_count','relative_doppler_hz','residual_carrier_hz','replica')]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    prior=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    assert {r['replica'] for r in rows}.isdisjoint(r['replica'] for r in prior)
    for fd in (0.,.1,.7):
        for replica in range(900000,900008):
            g=[r for r in rows if r['relative_doppler_hz']==fd and r['replica']==replica]
            assert len(g)==3 and len({r['first10_sha256'] for r in g})==1
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=list(range(0,72,8))
    for i in indices:assert json.loads(json.dumps(evaluate(p['evaluation_tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=72,frozen_sources=True,identical_prefixes=True,prior_ids_disjoint=True,summary_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed: 72 records, identical prefixes, nine full replays')


if __name__=='__main__':run()
