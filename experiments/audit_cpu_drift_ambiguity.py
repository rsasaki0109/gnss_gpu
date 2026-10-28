from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from experiments.run_cpu_drift_ambiguity import (OUTPUT,digest,tasklist,evaluate,apply_alarm,
    summarize,apply_gate,PathSelectionGate,calibrate_path_gate)
from gnss_gpu.drift_ambiguity_cpu import ambiguity_score,candidate_grid
from gnss_gpu.cached_drift_carrier_paths_cpu import DriftFit
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['baseline_gate_path'])==p['baseline_gate_sha256']
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    cal=[json.loads(x) for x in (OUTPUT/'calibration.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(cal)==299 and len(rows)==160
    expected_cal=tasklist('null',0.,1170000,299)
    expected_test=tasklist('null',0.,1180000,64)+tasklist('echo',.2,1190000,32)+tasklist('echo',.8,1200000,64)
    keys=('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')
    assert [[r[k] for k in keys] for r in cal]==p['calibration_tasks']==json.loads(json.dumps(expected_cal))
    assert [[r[k] for k in keys] for r in rows]==p['evaluation_tasks']==json.loads(json.dumps(expected_test))
    ids={r['replica'] for r in cal+rows};assert len(ids)==459
    for path in Path('output').glob('cpu-drift-*/*.jsonl'):
        if path.parent==OUTPUT:continue
        assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines())),str(path)
    gate=calibrate_path_gate([r['profile_alarm']['score'] for r in cal])
    assert asdict(gate)==json.loads((OUTPUT/'frozen_gate.json').read_text())
    baseline_gate=PathSelectionGate(**json.loads(Path(p['baseline_gate_path']).read_text()))
    for row in cal+rows:
        base={k:v for k,v in row.items() if k not in ('profile_alarm','warning','selected','error_m','success','score_exceeds')}
        base=apply_gate(base,baseline_gate)
        for k,v in base.items():assert row[k]==v
        diag=row['profile_alarm']
        assert diag['evaluated']==(row['selected']['model']!='ambiguous')
        if diag['evaluated']:
            grid=candidate_grid(row['selected']['delay_chips'],DriftFit(**row['fits']['single']),DriftFit(**row['fits']['two']))
            np.testing.assert_array_equal([q['delay_chips'] for q in diag['points']],grid)
        else:
            assert not diag['points']
        losses=[q['loss'] for q in diag['points'] if q['converged']]
        assert diag['unresolved']==(diag['evaluated'] and not bool(losses))
        # All studies use 40 blocks x7365 harmonics.
        assert diag['score']==ambiguity_score(row['fits']['single']['loss'],row['fits']['two']['loss'],losses,40*7365)
    for row in rows:assert apply_alarm({k:v for k,v in row.items() if k!='warning'},gate)==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    cal_indices=sorted({0,298,max(range(len(cal)),key=lambda i:cal[i]['profile_alarm']['score'])})
    for index in cal_indices:assert evaluate(expected_cal[index])==cal[index]
    for index in (0,64,96,159):assert apply_alarm(evaluate(expected_test[index]),gate)==rows[index]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(calibration=299,evaluation=160,
        sources_and_artifacts_verified=True,prior_ids_disjoint=True,all_scores_choices_and_summary_recomputed=True,
        full_calibration_replays=cal_indices,full_evaluation_replays=[0,64,96,159]),indent=2))
    print(f'Audit passed:459 records,{len(cal_indices)+4} complete replays')


if __name__=='__main__':run()
