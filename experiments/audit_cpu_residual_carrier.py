"""Verify frozen inputs, every decision and summary, and eight full replays."""
import json
from experiments.stress_cpu_residual_carrier import OUTPUT, PRIOR, digest, evaluate, select, summarize, PathSelectionGate


def run():
    p = json.loads((OUTPUT/'protocol.json').read_text())
    c = json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path) == h for path, h in p['source_sha256'].items())
    assert digest(p['gate_path']) == p['gate_sha256']
    assert json.loads(open(p['gate_path']).read())['gates']['ranked'] == p['gate']
    for name in ('evaluation', 'report'):
        assert digest(OUTPUT/(name + ('.jsonl' if name == 'evaluation' else '.json'))) == c[name+'_sha256']
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows) == c['evaluation'] == 192
    assert [[r[k] for k in ('echo_voltage', 'residual_carrier_hz', 'replica')] for r in rows] == p['evaluation_tasks']
    prior = [json.loads(x) for x in (PRIOR/'calibration.jsonl').read_text().splitlines()]
    assert {r['replica'] for r in prior}.isdisjoint({r['replica'] for r in rows})
    gate = PathSelectionGate(**p['gate'])
    derived = ('selected', 'error_m', 'success', 'forced_error_m', 'forced_success', 'score_exceeds')
    for row in rows:
        assert select({k:v for k,v in row.items() if k not in derived}, gate) == row
    assert summarize(rows) == json.loads((OUTPUT/'report.json').read_text())
    indices = list(range(0, 192, 24))
    for i in indices:
        assert json.loads(json.dumps(select(evaluate(p['evaluation_tasks'][i]), gate))) == rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=192, frozen_sources_and_gate=True,
        calibration_ids_disjoint=True, all_choices_and_summary_recomputed=True, full_replays=indices), indent=2))
    print('Audit passed: 192 decisions, eight full replays, frozen gate and sources')


if __name__ == '__main__':
    run()
