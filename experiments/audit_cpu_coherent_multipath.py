"""Audit nested integration identities, denominators and limited exact replay."""
import hashlib
import json
from pathlib import Path
from scipy.stats import binomtest
from experiments.run_cpu_coherent_multipath import OUTPUT, KEYS, evaluate, summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    records=[json.loads(line) for line in (OUTPUT/'records.jsonl').read_text().splitlines()]
    report=json.loads((OUTPUT/'report.json').read_text())
    for path,digest in {**protocol['source_sha256'],**protocol['prior_artifact_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    identities=[tuple(r[k] for k in KEYS)+(r['replica'],r['duration_ms']) for r in records]
    expected=[tuple(task)+(duration,) for task in protocol['tasks'] for duration in (20,80,200)]
    assert identities==expected and len(set(identities))==len(records)==1920
    cells,comparisons=summarize(records)
    assert dict(cells=cells,comparisons=comparisons)==report
    assert len(cells)==30 and len(comparisons)==20
    for cell in cells:
        rows=[r for r in records if all(r[k]==cell[k] for k in KEYS) and r['duration_ms']==cell['duration_ms']]
        assert len(rows)==cell['attempts']==64
        successes=sum(r['selected_error_m'] is not None and abs(r['selected_error_m'])<10 for r in rows)
        assert successes==cell['selected_successes']
    for comparison in comparisons:
        rows=[r for r in records if all(r[k]==comparison[k] for k in KEYS)]
        lookup={(r['replica'],r['duration_ms']):r for r in rows}
        gains=losses=0
        for replica in range(64):
            pair=[lookup[(replica,d)] for d in (20,comparison['duration_ms'])]
            a,b=[r['selected_error_m'] is not None and abs(r['selected_error_m'])<10 for r in pair]
            gains+=int(not a and b);losses+=int(a and not b)
        assert (gains,losses)==(comparison['gains'],comparison['losses'])
        p=binomtest(gains,gains+losses).pvalue if gains+losses else 1.
        assert p==comparison['paired_exact_p']
    assert sum(c['primary'] for c in comparisons)==2
    for index in (0,len(protocol['tasks'])-1):
        actual=json.loads(json.dumps(evaluate(protocol['tasks'][index])))
        assert actual==records[index*3:index*3+3]
    audit=dict(records=1920,cells=30,paired_comparisons=20,source_and_prior_hashes_verified=True,
        all_identities_denominators_and_paired_tables_verified=True,
        exact_replay_task_indices=[0,len(protocol['tasks'])-1],
        limitation='Six estimator outputs replayed using same model; perfect carrier/nav coherence remains assumed')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
