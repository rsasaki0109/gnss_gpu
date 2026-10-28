import json
from experiments.diagnose_cpu_path_boundary import OUTPUT,PRIOR,digest,evaluate,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    assert digest(OUTPUT/'diagnostics.jsonl')==c['diagnostics_sha256']
    assert digest(OUTPUT/'report.json')==c['report_sha256']
    prior=[json.loads(line) for line in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    chosen=[r for r in prior if r['arms']['dynamic']['error_m'] is None]
    rows=[json.loads(line) for line in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    identities=lambda rows:[[r['doppler_hz'],r['phase_rad'],r['replica']] for r in rows]
    assert identities(rows)==identities(chosen)==p['cases']
    assert len(rows)==c['cases']==29
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,28):assert json.loads(json.dumps(evaluate(chosen[i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=29,complete_failure_selection=True,sources_and_prior_verified=True,summary_recomputed=True,replayed_indices=[0,28]),indent=2))
    print('Audit passed: 29 cases, frozen lineage and two full replays')

if __name__=='__main__':run()
