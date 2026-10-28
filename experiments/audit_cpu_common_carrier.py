import json
from experiments.run_cpu_common_carrier import OUTPUT,PRIOR,digest,evaluate,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for name,extension in [('evaluation','jsonl'),('report','json')]:
        assert digest(OUTPUT/f'{name}.{extension}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    old=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    ident=lambda r:[r[k] for k in ('echo_voltage','residual_carrier_hz','replica')]
    assert len(rows)==len(old)==c['evaluation']==192
    assert [ident(r) for r in rows]==[ident(r) for r in old]==p['evaluation_tasks']
    assert all(r['observation_sha256']==o['observation_sha256'] for r,o in zip(rows,old))
    assert summarize(rows,old)==json.loads((OUTPUT/'report.json').read_text())
    indices=list(range(0,192,24))
    for i in indices:
        assert json.loads(json.dumps(evaluate(p['evaluation_tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=192,sources_and_prior_preserved=True,
        all_observations_match=True,summary_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed: 192 matched observations, summary, eight full replays')


if __name__=='__main__':run()
