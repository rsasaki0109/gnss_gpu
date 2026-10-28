import json
from experiments.diagnose_cpu_residual_tail import OUTPUT,PRIOR,digest,select_cases,diagnose,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    assert digest(OUTPUT/'diagnostics.jsonl')==c['diagnostics_sha256']
    assert digest(OUTPUT/'report.json')==c['report_sha256']
    cases=select_cases([json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()])
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    identity=lambda rows:[[r['profile'],r['replica'],r['selected_rank']] for r in rows]
    assert identity(rows)==identity(cases)==p['cases']
    assert len(rows)==c['cases']==12
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,4,8):assert json.loads(json.dumps(diagnose(cases[i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(cases=12,frozen_sources_and_prior=True,selection_and_summary_recomputed=True,replayed_indices=[0,4,8]),indent=2))
    print('Audit passed: all12 cases, three profile maxima fully replayed')

if __name__=='__main__':run()
