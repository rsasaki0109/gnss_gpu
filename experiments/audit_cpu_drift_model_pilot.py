import json
from experiments.run_cpu_drift_model_pilot import OUTPUT,digest,select_cases,evaluate,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    old=select_cases();rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    ident=lambda r:[r['kind'],r['replica'],r['drift_hz_s']]
    assert len(old)==len(rows)==c['evaluation']==48
    assert [ident(r) for r in old]==[ident(r) for r in rows]==p['cases']
    assert all(r['observation_sha256']==o['observation_sha256'] and r['old_score']==o['uncalibrated_score'] for r,o in zip(rows,old))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=list(range(0,48,8))
    for i in indices:assert json.loads(json.dumps(evaluate(old[i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=48,prior_and_sources_preserved=True,all_observations_match=True,summary_recomputed=True,full_replays=indices),indent=2))
    print('Audit passed:48 matched observations, sources, summary and six full replays')


if __name__=='__main__':run()
