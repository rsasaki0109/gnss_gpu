"""Recompute report and replay independently selected cases from each panel."""
import json
from pathlib import Path
from experiments.validate_cpu_drift_uncertainty import (OUTPUT, evaluation_tasks, uncertainty,
    summarize, digest, evaluate, apply_gate, PathSelectionGate)


def run():
    p = json.loads((OUTPUT/'protocol.json').read_text())
    c = json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']
    assert json.loads(Path(p['gate_path']).read_text())==p['gate']
    for name, ext in [('protocol','json'),('evaluation','jsonl'),('report','json')]:
        assert digest(OUTPUT/f'{name}.{ext}')==c[name+'_sha256']
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    expected = json.loads(json.dumps(evaluation_tasks()))
    keys = ('kind','echo_voltage','excess_chips','phase_rad','drift_hz_s','initial_carrier_hz','replica')
    assert [[r[k] for k in keys] for r in rows]==p['evaluation_tasks']==expected
    assert len(rows)==c['evaluation']==128
    ids = {r['replica'] for r in rows}
    assert len(ids)==128
    for path in Path('output').glob('cpu-drift-*/*.jsonl'):
        if path.parent==OUTPUT: continue
        prior = [json.loads(x) for x in path.read_text().splitlines()]
        assert ids.isdisjoint(r['replica'] for r in prior if 'replica' in r), str(path)
    gate = PathSelectionGate(**p['gate'])
    for row in rows:
        raw = {k:v for k,v in row.items() if k not in ('selected','error_m','success','score_exceeds','uncertainty')}
        chosen = apply_gate(raw,gate)
        chosen['uncertainty'] = uncertainty(chosen)
        assert chosen==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices = [0,31,32,63,64,127]
    for i in indices:
        fresh = apply_gate(evaluate(expected[i]),gate)
        fresh['uncertainty'] = uncertainty(fresh)
        assert json.loads(json.dumps(fresh))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=128, sources_gate_and_artifacts_verified=True,
        previous_drift_ids_disjoint=True, all_choices_uncertainties_and_summary_recomputed=True,
        full_replays=indices),indent=2))
    print('Audit passed:128 rows,6 full replays')


if __name__=='__main__': run()
