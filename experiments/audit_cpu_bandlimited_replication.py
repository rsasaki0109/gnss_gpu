"""Audit frozen replication inputs, paired denominators, and limited replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_bandlimited_multipath import evaluate
from experiments.validate_cpu_bandlimited_multipath import OUTPUT, KEYS, summarize


def run():
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    records = [json.loads(line) for line in (OUTPUT/'records.jsonl').read_text().splitlines()]
    report = json.loads((OUTPUT/'report.json').read_text())
    assert len(records) == len(protocol['tasks']) == 2560
    identities = [tuple(r[k] for k in KEYS)+(r['replica'],) for r in records]
    assert len(set(identities)) == 2560
    assert identities == [tuple(t) for t in protocol['tasks']]
    assert set(r['replica'] for r in records) == set(range(10000, 10128))
    for path, digest in {**protocol['source_sha256'], **protocol['previous_artifact_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert summarize(records) == report
    # Independently reconstruct 2x2 tables including unavailable selected ranges.
    for cell in report:
        table = np.zeros((2, 2), dtype=int)
        rows = [r for r in records if all(r[k] == cell[k] for k in KEYS)]
        for row in rows:
            a = int(abs(row['single_error_m']) < 10)
            b = int(row['selected_error_m'] is not None and abs(row['selected_error_m']) < 10)
            table[a, b] += 1
        assert table.sum() == cell['attempts'] == 128
        assert table[0, 1] == cell['gains'] and table[1, 0] == cell['losses']
        assert table[1].sum() == cell['baseline_successes']
        assert table[:, 1].sum() == cell['selected_successes']
        discordant = int(table[0, 1]+table[1, 0])
        p = binomtest(int(table[0, 1]), discordant).pvalue if discordant else 1.
        assert p == cell['paired_exact_p']
    assert sum(r['primary'] for r in report) == 2
    for i in (0, 2559):
        assert json.loads(json.dumps(evaluate(protocol['tasks'][i]))) == records[i]
    audit = dict(records=2560, cells=20, independent_replica_ids_per_cell=128,
                 all_sources_and_previous_artifacts_unchanged=True,
                 all_paired_tables_and_reports_verified=True,
                 exact_replay_indices=[0,2559],
                 limitation='Limited replay with same synthesis/fit implementation; no real RF validation')
    (OUTPUT/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
