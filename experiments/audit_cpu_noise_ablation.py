"""Audit the 2x2 gate mapping, paired records and all saved contrasts."""
import hashlib,json
from pathlib import Path
from experiments.run_cpu_noise_ablation import OUTPUT,ARMS,evaluate,calibrate_arms,apply_arms,summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gates=json.loads((OUTPUT/'frozen_gates.json').read_text())
    report=json.loads((OUTPUT/'report.json').read_text())
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(cal)==1431 and len(rows)==768 and 6*.99**477<=.05
    assert calibrate_arms(cal)==gates
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
        assert json.loads(json.dumps(apply_arms({k:v for k,v in row.items() if k!='outcomes'},gates)))==row
    assert summarize(rows)==report
    assert len(report['cells'])==36 and len(report['contrasts'])==36
    assert sum(c['primary'] for c in report['contrasts'])==2
    for c in report['cells']:
        group=[r for r in rows if all(r[k]==c[k] for k in ('profile','echo_voltage','phase_rad'))]
        assert len(group)==c['attempts']==(128 if c['echo_voltage']==0 else 64)
        outcomes=[r['outcomes'][c['arm']] for r in group]
        assert sum(o['error_m'] is not None and abs(o['error_m'])<10 for o in outcomes)==c['successes']
    for i in (0,477,954):
        assert json.loads(json.dumps(evaluate(protocol['calibration_tasks'][i])))==cal[i]
    for i in (128,767):
        assert json.loads(json.dumps(apply_arms(evaluate(protocol['evaluation_tasks'][i]),gates)))==rows[i]
    audit=dict(calibration_records=1431,evaluation_observations=768,selection_outcomes=3072,
        all_gates_and_joint_calibration_bound_verified=True,all_arm_mappings_and_reports_verified=True,
        source_and_prior_artifact_hashes_verified=True,exact_calibration_replays=[0,477,954],
        exact_evaluation_replays=[128,767],limitation='One calibration draw; shared evaluation replicas; same-model limited replay')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__': run()
