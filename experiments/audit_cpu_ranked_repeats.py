"""Audit repeated rank calibration without inflating shared evaluation counts."""
import hashlib,json
from pathlib import Path
from experiments.run_cpu_ranked_repeats import OUTPUT,INITIAL,evaluate,calibrate,apply_repeats,summarize_repeats


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    initial=[json.loads(x) for x in (INITIAL/'calibration.jsonl').read_text().splitlines()]
    cal=[json.loads(x) for x in (OUTPUT/'new_calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gates=json.loads((OUTPUT/'frozen_gates.json').read_text())
    report=json.loads((OUTPUT/'report.json').read_text())
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert len(initial)==3000 and len(cal)==6000 and len(rows)==512
    assert gates['0']==calibrate(initial)==json.loads((INITIAL/'frozen_gates.json').read_text())
    groups=[initial]+[[r for r in cal if r['calibration_repeat']==repeat] for repeat in (1,2)]
    for repeat,group in enumerate(groups):assert len(group)==3000 and calibrate(group)==gates[str(repeat)]
    ids=[set(r['replica'] for r in group) for group in groups]
    assert all(len(group)==3000 for group in ids)
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    assert not set.union(*ids)&set(r['replica'] for r in rows)
    assert not set(r['replica'] for r in rows if r['echo_voltage']==0)&set(r['replica'] for r in rows if r['echo_voltage']!=0)
    keys=('profile','echo_voltage','gap_chips','phase_rad','replica')
    for records,tasks in ((cal,protocol['new_calibration_tasks']),(rows,protocol['evaluation_tasks'])):
        identities=[tuple(r[k] for k in keys) for r in records]
        assert identities==[tuple(t[:5]) for t in tasks] and len(set(identities))==len(records)
    for p,h in {**protocol['source_sha256'],**protocol['prior_sha256']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUTPUT/'frozen_gates.json').read_bytes()).hexdigest()==completion['gate_sha256']
    for row in rows:
        assert json.loads(json.dumps(apply_repeats({k:v for k,v in row.items() if k!='outcomes'},gates)))==row
    assert summarize_repeats(rows,gates)==report
    assert len(report['cells'])==24 and len(report['contrasts'])==12
    assert sum(c['primary'] for c in report['contrasts'])==3
    for c in report['cells']:
        group=[r for r in rows if r['profile']==c['profile'] and r['echo_voltage']==c['echo_voltage']]
        assert len(group)==c['attempts']==128
        selected=[r['outcomes'][str(c['repeat'])][c['method']] for r in group]
        assert sum(r['error_m'] is not None and abs(r['error_m'])<10 for r in selected)==c['successes']
    for i in (0,3000):
        row=evaluate(protocol['new_calibration_tasks'][i]);row['calibration_repeat']=1+i//3000
        assert json.loads(json.dumps(row))==cal[i]
    for i in (0,511):
        assert json.loads(json.dumps(apply_repeats(evaluate(protocol['evaluation_tasks'][i]),gates)))==rows[i]
    # The reused single-run summary retains an unadjusted support flag. Keep
    # family-level decisions explicit and use these for the repeated study.
    decisions=[dict(repeat=c['repeat'],adjusted_p=c['adjusted_primary_p'],
                    improvement_supported=c['difference']>0 and c['adjusted_primary_p']<.05)
               for c in report['contrasts'] if c['primary']]
    (OUTPUT/'family_decisions.json').write_text(json.dumps(decisions,indent=2))
    audit=dict(new_calibration=6000,preserved_initial_calibration=3000,shared_evaluation=512,
        all_gates_groups_denominators_and_source_hashes_verified=True,
        initial_gate_exactly_preserved=True,selection_outcomes=3072,
        exact_new_calibration_replays=[0,3000],exact_evaluation_replays=[0,511],
        limitation='Two new draws plus one prior development draw; one shared synthetic evaluation set')
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
