"""Audit independent calibration groups and shared paired evaluation."""
import hashlib,json
from pathlib import Path
from experiments.run_cpu_calibration_repeats import OUTPUT,evaluate,calibrate_repeats,apply_repeats,summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gates=json.loads((OUTPUT/'frozen_gates.json').read_text())
    report=json.loads((OUTPUT/'report.json').read_text())
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(cal)==4293 and len(rows)==512
    assert calibrate_repeats(cal)==gates
    assert hashlib.sha256((OUTPUT/'frozen_gates.json').read_bytes()).hexdigest()==completion['gate_sha256']
    for p,h in {**protocol['source_sha256'],**protocol['prior_sha256']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    keys=('profile','echo_voltage','gap_chips','phase_rad','replica')
    for records,tasks in ((cal,protocol['calibration_tasks']),(rows,protocol['evaluation_tasks'])):
        ids=[tuple(r[k] for k in keys) for r in records]
        assert ids==[tuple(t[:5]) for t in tasks] and len(set(ids))==len(records)
    groups=[set(r['replica'] for r in cal if r['calibration_repeat']==i) for i in range(3)]
    assert all(len(group)==1431 for group in groups)
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert not set.union(*groups)&set(r['replica'] for r in rows)
    assert not set(r['replica'] for r in rows if r['echo_voltage']==0)&set(r['replica'] for r in rows if r['echo_voltage']!=0)
    for row in rows:
        assert json.loads(json.dumps(apply_repeats({k:v for k,v in row.items() if k!='outcomes'},gates)))==row
    assert summarize(rows)==report and len(report['cells'])==24 and len(report['contrasts'])==12
    assert sum(r['primary'] for r in report['contrasts'])==3
    for c in report['cells']:
        group=[r for r in rows if r['profile']==c['profile'] and r['echo_voltage']==c['echo_voltage']]
        assert len(group)==c['attempts']==128
        outcomes=[r['outcomes'][str(c['repeat'])][c['coverage']] for r in group]
        assert sum(r['error_m'] is not None and abs(r['error_m'])<10 for r in outcomes)==c['successes']
    for i in (0,1431,2862):
        raw=evaluate(protocol['calibration_tasks'][i]);raw['calibration_repeat']=i//1431
        assert json.loads(json.dumps(raw))==cal[i]
    for i in (0,511):
        assert json.loads(json.dumps(apply_repeats(evaluate(protocol['evaluation_tasks'][i]),gates)))==rows[i]
    audit=dict(calibration_records=4293,independent_calibration_repeats=3,shared_evaluation_records=512,
        selection_outcomes=3072,all_thresholds_identities_disjointness_and_denominators_verified=True,
        all_source_and_prior_hashes_verified=True,exact_calibration_replays=[0,1431,2862],
        exact_evaluation_replays=[0,511],limitation='Only three calibration draws; one shared synthetic evaluation set')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
