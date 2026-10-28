import json,hashlib
from dataclasses import asdict
import numpy as np
from gnss_gpu.drift_carrier_paths_cpu import score_drift_paths as baseline_score
from experiments.stress_cpu_drift_gap_amplitude import OUTPUT,PRIOR,digest,tasks,CONDITIONS,evaluate,synthesize,apply_gate,summarize,PathSelectionGate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']==c['gate_sha256']
    assert json.loads(open(p['gate_path']).read())==p['gate']
    for name,ext in [('evaluation','jsonl'),('report','json')]:assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==c['evaluation']==224
    expected=[(kind,0.,.1,0.,a,f,i) for kind,a,f,i in tasks('null',1050000,32)]+[(kind,ratio,gap,phase,a,f,i) for ratio,gap,phase in CONDITIONS for kind,a,f,i in tasks('echo',1060000,16)]
    ident=lambda r:[r[k] for k in ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')]
    assert [ident(r) for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected))
    previous=[json.loads(x) for x in (PRIOR/'evaluation.jsonl').read_text().splitlines()]
    from pathlib import Path
    base=Path('output/cpu-drift-selection')
    for name in ('calibration.jsonl','evaluation.jsonl'):previous.extend(json.loads(x) for x in (base/name).read_text().splitlines())
    assert {r['replica'] for r in rows}.isdisjoint(r['replica'] for r in previous)
    gate=PathSelectionGate(**p['gate'])
    for row in rows:assert apply_gate({k:v for k,v in row.items() if k not in ('selected','error_m','success','score_exceeds')},gate)==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[0]+list(range(32,224,16));baseline_indices=[0,32,208]
    for i in indices:
        task=p['evaluation_tasks'][i]
        assert json.loads(json.dumps(apply_gate(evaluate(task),gate)))==rows[i]
        if i in baseline_indices:
            b,_,_=synthesize(task);score,weights=baseline_score(b)
            assert score['score']==rows[i]['uncalibrated_score']
            assert {k:asdict(score[k]) for k in ('single','two','preliminary')}==rows[i]['fits']
            assert hashlib.sha256(weights.tobytes()).hexdigest()==rows[i]['weights_sha256']
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=224,sources_and_gate_preserved=True,prior_ids_disjoint=True,
        all_choices_and_summary_recomputed=True,cached_full_replays=indices,baseline_full_replays=baseline_indices),indent=2))
    print('Audit passed:224 choices,13 cached and3 baseline full replays')


if __name__=='__main__':run()
