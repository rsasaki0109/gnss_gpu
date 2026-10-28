"""Audit fixed rank, exact tolerance budget, identities and held-out outputs."""
import hashlib,json
from pathlib import Path
from scipy.stats import binom
from experiments.run_cpu_ranked_calibration import OUTPUT,evaluate,calibrate,apply_gates,summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gates=json.loads((OUTPUT/'frozen_gates.json').read_text())
    report=json.loads((OUTPUT/'report.json').read_text())
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(cal)==3000 and len(rows)==1281 and protocol['rank_from_largest']==4
    assert gates==calibrate(cal)
    total_bound=0.
    for profile,stratum in gates['strata'].items():
        scores=sorted([r['fit']['bic_improvement'] for r in cal if r['profile']==profile],reverse=True)
        assert len(scores)==1000
        assert stratum['ranked']['score']==scores[3] and stratum['maximum']['score']==scores[0]
        bound=float(binom.cdf(3,1000,.01));total_bound+=bound
        assert bound==stratum['ranked']['actual_failure_bound']
        assert binom.cdf(4,1000,.01)>.05/3
    assert total_bound<=.05
    assert gates['combined']['ranked']['score']<=gates['combined']['maximum']['score']
    assert hashlib.sha256((OUTPUT/'frozen_gates.json').read_bytes()).hexdigest()==completion['gate_sha256']
    for p,h in {**protocol['source_sha256'],**protocol['prior_sha256']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    keys=('profile','echo_voltage','gap_chips','phase_rad','replica')
    for records,tasks in ((cal,protocol['calibration_tasks']),(rows,protocol['evaluation_tasks'])):
        ids=[tuple(r[k] for k in keys) for r in records]
        assert ids==[tuple(t[:5]) for t in tasks] and len(set(ids))==len(records)
    assert not set(r['replica'] for r in cal)&set(r['replica'] for r in rows)
    assert not set(r['replica'] for r in rows if r['echo_voltage']==0)&set(r['replica'] for r in rows if r['echo_voltage']!=0)
    for row in rows:
        assert json.loads(json.dumps(apply_gates({k:v for k,v in row.items() if k!='outcomes'},gates)))==row
    assert summarize(rows)==report and len(report['cells'])==12 and len(report['contrasts'])==6
    assert sum(c['primary'] for c in report['contrasts'])==1
    for c in report['cells']:
        group=[r for r in rows if r['profile']==c['profile'] and r['echo_voltage']==c['echo_voltage']]
        assert len(group)==c['attempts']==(299 if c['echo_voltage']==0 else 128)
        outcomes=[r['outcomes'][c['method']] for r in group]
        assert sum(o['error_m'] is not None and abs(o['error_m'])<10 for o in outcomes)==c['successes']
    for i in (0,1000,2000):
        assert json.loads(json.dumps(evaluate(protocol['calibration_tasks'][i])))==cal[i]
    for i in (299,1280):
        assert json.loads(json.dumps(apply_gates(evaluate(protocol['evaluation_tasks'][i]),gates)))==rows[i]
    audit=dict(calibration=3000,evaluation_observations=1281,rank=4,
        joint_calibration_failure_bound=total_bound,all_gates_identities_outcomes_and_hashes_verified=True,
        calibration_replay_indices=[0,1000,2000],evaluation_replay_indices=[299,1280],
        limitation='Same synthetic model; one calibration realization; no empirical across-calibration stability claim')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
