"""Verify fixed gate, all false-selection counts, corrected tests and replay."""
import hashlib
import json
from pathlib import Path
from scipy.stats import binomtest
from experiments.stress_cpu_path_selection import OUTPUT,GATE,evaluate,summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    rows=[json.loads(x) for x in (OUTPUT/'records.jsonl').read_text().splitlines()]
    report=json.loads((OUTPUT/'report.json').read_text())
    assert len(rows)==1536 and len(report)==12
    identities=[(r['reference_cn0_dbhz'],r['profile'],r['replica']) for r in rows]
    assert identities==[tuple(t[:3]) for t in protocol['tasks']] and len(set(identities))==1536
    assert hashlib.sha256(GATE.read_bytes()).hexdigest()==protocol['frozen_gate_sha256']
    assert json.loads(GATE.read_text())==protocol['frozen_gate']
    for path,digest in {**protocol['source_sha256'],**protocol['prior_artifact_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert summarize(rows)==report
    for cell in report:
        group=[r for r in rows if r['reference_cn0_dbhz']==cell['reference_cn0_dbhz'] and r['profile']==cell['profile']]
        assert len(group)==cell['attempts']==128
        false=sum(r['calibrated_fit']['selected_model']=='two_path' for r in group)
        assert false==cell['false_two_path_selections']
        assert false<=cell['score_exceedances']
        p=binomtest(false,128,p=.01,alternative='greater').pvalue
        assert cell['bonferroni_12_p']==min(1.,12*p)
        assert sum(r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in group)==cell['new_under_10m']
    for i in (0,128,256,1535):
        assert json.loads(json.dumps(evaluate(protocol['tasks'][i])))==rows[i]
    audit=dict(records=1536,cells=12,frozen_gate_unchanged=True,
        source_and_prior_artifact_hashes_verified=True,all_denominators_and_tests_verified=True,
        exact_replay_indices=[0,128,256,1535],
        limitation='Same synthetic model; four replayed records; colored noise profiles are not measured RF data')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
