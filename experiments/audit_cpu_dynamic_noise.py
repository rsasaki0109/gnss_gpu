import json
from experiments.stress_cpu_dynamic_noise import OUTPUT,digest,evaluate,apply_gates,summarize
from experiments.run_cpu_dynamic_selection import evaluate as white_evaluate


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert digest(p['gate_path'])==p['gate_sha256']==c['gate_sha256']
    assert digest(OUTPUT/'evaluation.jsonl')==c['evaluation_sha256']
    assert digest(OUTPUT/'report.json')==c['report_sha256']
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==len(p['evaluation_tasks'])==c['evaluation']==672
    for row,task in zip(rows,p['evaluation_tasks']):
        assert [row[k] for k in ('profile','echo_voltage','doppler_hz','phase_rad','replica')]==task
        raw={k:v for k,v in row.items() if k!='outcomes'}
        assert apply_gates(raw,p['gates'])==row
    assert summarize(rows,p['gates'])==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,128,256,384,480,576):
        assert json.loads(json.dumps(apply_gates(evaluate(p['evaluation_tasks'][i]),p['gates'])))==rows[i]
    # White special case reproduces the previous frozen generator bit for bit.
    old=json.loads(json.dumps(white_evaluate(p['evaluation_tasks'][0][1:])))
    assert old=={k:v for k,v in rows[0].items() if k not in ('profile','outcomes')}
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=672,frozen_sources_and_gates=True,all_choices_and_summary_recomputed=True,replayed_indices=[0,128,256,384,480,576],white_generator_equivalence=True),indent=2))
    print('Audit passed: 672 records, all profiles replayed, white generator identical')

if __name__=='__main__':run()
