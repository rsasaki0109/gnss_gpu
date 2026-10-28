import json
from experiments.diagnose_cpu_common_carrier_boundary import OUTPUT,PRIOR,digest,select_cases,diagnose,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    for key,name in [('diagnostics','diagnostics.jsonl'),('report','report.json')]:assert digest(OUTPUT/name)==c[key+'_sha256']
    cases=select_cases([json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()])
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    ident=lambda r:[r['replica'],r['relative_doppler_hz'],r['phase_rad']]
    assert len(cases)==len(rows)==c['cases']==13
    assert [ident(r) for r in cases]==[ident(r) for r in rows]==p['cases']
    for row,old in zip(rows,cases):
        assert row['observation_sha256']==old['observation_sha256'] and row['weights_sha256']==old['weights_sha256']
        for name in ('same_bounds','relaxed_floor'):
            assert row[name]['loss']<=row['original']['loss']
            assert row[name]['attempted_starts']==33
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,12):assert json.loads(json.dumps(diagnose(cases[i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=13,all_prior_boundary_cases_included=True,sources_and_prior_preserved=True,summary_recomputed=True,full_replays=[0,12]),indent=2))
    print('Audit passed: all13 cases, preserved sources/inputs, summary and two full replays')


if __name__=='__main__':run()
