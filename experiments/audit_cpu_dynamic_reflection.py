"""Verify frozen inputs, complete paired records, summaries and selected replays."""
import json
from experiments.run_cpu_dynamic_reflection import OUTPUT, digest, evaluate, select, summarize


def run():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    completion=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(p)==h for p,h in protocol['source_sha256'].items())
    assert digest(protocol['gate_path'])==protocol['gate_sha256']
    assert digest(OUTPUT/'evaluation.jsonl')==completion['evaluation_sha256']
    assert digest(OUTPUT/'report.json')==completion['report_sha256']
    rows=[json.loads(line) for line in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==len(protocol['tasks'])==completion['evaluation']==640
    for row,task in zip(rows,protocol['tasks']):
        assert [row['echo_voltage'],row['doppler_hz'],row['replica']]==task
        raw={k:v for k,v in row.items() if k not in ('selected','error_m','success')}
        assert json.loads(json.dumps(select(raw,protocol['gate'])))==row
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    for i in (0,128,256,384,639):
        replay=select(evaluate(protocol['tasks'][i]),protocol['gate'])
        assert json.loads(json.dumps(replay))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(records=640,source_hashes_verified=True,gate_unchanged=True,replayed_indices=[0,128,256,384,639],all_selections_and_report_recomputed=True),indent=2))
    print('Audit passed: 640 records, frozen sources/gate, five replays')

if __name__=='__main__':run()
