import json
from experiments.run_cpu_dynamic_path_fit import OUTPUT,PRIOR,digest,evaluate,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['prior_sha256']}.items())
    assert digest(OUTPUT/'evaluation.jsonl')==c['evaluation_sha256']
    assert digest(OUTPUT/'report.json')==c['report_sha256']
    rows=[json.loads(line) for line in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['records']==len(p['tasks'])==101
    for row,task in zip(rows,p['tasks']):
        assert [row['panel'],row['doppler_hz'],row['phase_rad'],row['replica']]==task
        assert row['raw_success']==(abs(row['error_m'])<10)
        assert row['locally_usable_success']==(row['fit']['locally_usable'] and row['raw_success'])
    previous=[json.loads(line) for line in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    chosen=[r for r in previous if r['arms']['dynamic']['error_m'] is None]
    assert len(chosen)==29
    for row,old in zip(rows[:29],chosen):
        assert row['old']==old['arms']['dynamic']
        assert row['observation_sha256']==old['observation_sha256']['dynamic']
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,29,100):assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(records=101,frozen_lineage=True,prior_failures_reproduced=29,summary_recomputed=True,replayed_indices=[0,29,100]),indent=2))
    print('Audit passed: 101 records, prior failures reproduced, three full replays')

if __name__=='__main__':run()
