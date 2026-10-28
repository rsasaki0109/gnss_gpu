import json,statistics
from experiments.benchmark_cpu_drift_cache import OUTPUT,PRIOR,digest,replay


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for key,name in [('replay','replay.jsonl'),('timings','timings.jsonl'),('report','report.json')]:assert digest(OUTPUT/name)==c[key+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'replay.jsonl').read_text().splitlines()]
    timings=[json.loads(x) for x in (OUTPUT/'timings.jsonl').read_text().splitlines()]
    assert len(rows)==c['replays']==395 and len(timings)==c['timing_pairs']==8
    cal=[json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    evaluation=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    items=[('calibration',i,r) for i,r in enumerate(cal)]+[('evaluation',i,r) for i,r in enumerate(evaluation)]
    for row,(stage,index,old) in zip(rows,items):
        assert (row['stage'],row['index'],row['replica'])==(stage,index,old['replica'])
        assert row['exact'] and row['score']==old['uncalibrated_score'] and row['weights_sha256']==old['weights_sha256']
    for repeat in range(2):
        assert [[r['stage'],r['index'],r['replica']] for r in timings if r['repeat']==repeat]==p['benchmark_cases']
    for r in timings:assert r['baseline_seconds']>0 and r['cached_seconds']>0 and r['speedup']==r['baseline_seconds']/r['cached_seconds']
    report=json.loads((OUTPUT/'report.json').read_text())
    assert report['median_paired_speedup']==statistics.median(r['speedup'] for r in timings)
    for i in (0,298,299,363,390,394):
        check=replay(items[i]);saved=rows[i]
        assert {k:v for k,v in check.items() if k!='cached_elapsed_seconds'}=={k:v for k,v in saved.items() if k!='cached_elapsed_seconds'}
    (OUTPUT/'audit.json').write_text(json.dumps(dict(exact_rows=395,timing_pairs=8,source_and_prior_hashes=True,full_replays=[0,298,299,363,390,394]),indent=2))
    print('Audit passed:395 exact rows,8 timing pairs,six full cached replays')


if __name__=='__main__':run()
