import json,hashlib
from dataclasses import asdict
from gnss_gpu.drift_carrier_paths_cpu import score_drift_paths as baseline_score
from experiments.stress_cpu_drift_phase import OUTPUT,PRIOR,digest,tasks,CONDITIONS,evaluate,synthesize,apply_gate,summarize,PathSelectionGate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in {**p['source_sha256'],**p['cache_evidence_sha256']}.items())
    assert digest(p['gate_path'])==p['gate_sha256']==c['gate_sha256']
    assert json.loads(open(p['gate_path']).read())==p['gate']
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['evaluation']==176
    import numpy as np
    expected=[(kind,.1,np.pi,a,f,i) for kind,a,f,i in tasks('null',1020000,32)]+[(kind,fd,phase,a,f,i) for fd,phase in CONDITIONS for kind,a,f,i in tasks('echo',1030000,16)]
    ident=lambda r:[r[k] for k in ('kind','relative_doppler_hz','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    prior=[]
    for name in ('calibration.jsonl','evaluation.jsonl'):prior.extend(json.loads(x) for x in (PRIOR/name).read_text().splitlines())
    assert {r['replica'] for r in rows}.isdisjoint(r['replica'] for r in prior)
    gate=PathSelectionGate(**p['gate'])
    for row in rows:assert apply_gate({k:v for k,v in row.items() if k not in ('selected','error_m','success','score_exceeds')},gate)==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[0]+list(range(32,176,16))
    for i in indices:
        task=p['evaluation_tasks'][i]
        assert json.loads(json.dumps(apply_gate(evaluate(task),gate)))==rows[i]
        b,_,_=synthesize(task);score,weights=baseline_score(b)
        assert score['score']==rows[i]['uncalibrated_score']
        assert {k:asdict(score[k]) for k in ('single','two','preliminary')}==rows[i]['fits']
        assert hashlib.sha256(weights.tobytes()).hexdigest()==rows[i]['weights_sha256']
        print(f'Audit condition {indices.index(i)+1}/10',flush=True)
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=176,sources_and_gate_preserved=True,prior_ids_disjoint=True,
        all_choices_and_summary_recomputed=True,cached_full_replays=indices,baseline_full_replays=indices),indent=2))
    print('Audit passed:176 choices,ten cached and ten baseline exact replays')


if __name__=='__main__':run()
