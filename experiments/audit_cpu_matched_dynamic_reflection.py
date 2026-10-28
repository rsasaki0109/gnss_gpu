"""Verify paired intervention, frozen gate/source lineage, and replay."""
import json
from experiments.run_cpu_matched_dynamic_reflection import OUTPUT,digest,evaluate,apply_gate,summarize


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text())
    c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']
    assert digest(OUTPUT/'evaluation.jsonl')==c['evaluation_sha256']
    assert digest(OUTPUT/'report.json')==c['report_sha256']
    rows=[json.loads(line) for line in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==len(p['tasks'])==c['pairs']==576 and c['fits']==1152
    for row,task in zip(rows,p['tasks']):
        assert [row['doppler_hz'],row['phase_rad'],row['replica']]==task
        assert row['mean_max_difference']<1e-15
        raw=dict(row,arms={name:{k:v for k,v in arm.items() if k in ('fit','true_delay_chips')} for name,arm in row['arms'].items()})
        assert json.loads(json.dumps(apply_gate(raw,p['gate'])))==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    indices=[i*64 for i in range(9)]
    for i in indices:
        assert json.loads(json.dumps(apply_gate(evaluate(p['tasks'][i]),p['gate'])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(pairs=576,fits=1152,frozen_sources_and_gate=True,all_means_matched=True,all_selections_and_report_recomputed=True,replayed_indices=indices),indent=2))
    print('Audit passed: 576 pairs, all nine conditions replayed')

if __name__=='__main__':run()
