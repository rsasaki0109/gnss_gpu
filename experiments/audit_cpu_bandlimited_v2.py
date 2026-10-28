"""Audit serialized outputs, including the archived pre-audit-fix runner."""
import hashlib
import json
from pathlib import Path
from experiments.validate_cpu_bandlimited_v2 import OUTPUT, KEYS, evaluate_pair, report_records


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    rows=[json.loads(line) for line in (OUTPUT/'records.jsonl').read_text().splitlines()]
    report=json.loads((OUTPUT/'report.json').read_text())
    assert len(rows)==1280 and len(report)==20
    archived=[]
    for path,digest in protocol['source_and_prior_sha256'].items():
        source=Path(path)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            assert source.name=='validate_cpu_bandlimited_v2.py'
            snapshot=OUTPUT/'executed_validation_source.py'
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
            # The only post-run change normalizes tuples on BOTH sides of replay.
            old_line='assert json.loads(json.dumps(evaluate_pair(tasks[index])))==records[index]'
            new_line='assert json.loads(json.dumps(evaluate_pair(tasks[index])))==json.loads(json.dumps(records[index]))'
            assert snapshot.read_text().replace(old_line,new_line)==source.read_text()
            archived.append(str(snapshot))
    assert report_records(rows)==report
    identities=[tuple(r['old'][k] for k in KEYS)+(r['old']['replica'],) for r in rows]
    assert identities==[tuple(t) for t in protocol['tasks']] and len(set(identities))==1280
    for row in rows:
        if row['old']['selected_error_m'] is not None:
            assert row['new_error_m']==row['old']['selected_error_m']
    for i in (0,1279):
        assert json.loads(json.dumps(evaluate_pair(protocol['tasks'][i])))==rows[i]
    audit=dict(records=1280,cells=20,source_hashes_verified=True,
        archived_executed_sources=archived,only_post_run_change='Normalize list/tuple replay comparison',
        prior_experiment_sources_and_artifacts_unchanged=True,
        previously_available_outputs_exactly_preserved=True,report_recomputed=True,
        replay_indices=[0,1279],gains=sum(r['gains'] for r in report),losses=sum(r['losses'] for r in report),
        old_unavailable=sum(r['old_unavailable'] for r in report),
        new_unavailable=sum(r['new_unavailable'] for r in report),
        limitation='Same-model limited replay; pooled counts are correlated across conditions')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
