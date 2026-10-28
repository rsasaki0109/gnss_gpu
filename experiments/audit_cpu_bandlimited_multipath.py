"""Check all saved denominators/metrics and replay two deterministic records."""
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
from experiments.run_cpu_bandlimited_multipath import evaluate


def run():
    output = Path('output/cpu-bandlimited-multipath')
    protocol = json.loads((output/'protocol.json').read_text())
    records = json.loads((output/'records.json').read_text())
    summaries = json.loads((output/'report.json').read_text())
    assert len(records) == len(protocol['tasks']) == 160
    assert len(summaries) == 20
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    keys = ('bandwidth_chip_rates', 'cn0_dbhz', 'echo_voltage', 'gap_chips', 'phase_rad')
    identities = [tuple(r[k] for k in keys)+(r['replica'],) for r in records]
    assert len(set(identities)) == 160
    assert identities == [tuple(task) for task in protocol['tasks']]
    for summary in summaries:
        rows = [r for r in records if all(r[k] == summary[k] for k in keys)]
        selected = [r for r in rows if r['selected_error_m'] is not None]
        assert len(rows) == summary['attempts'] == 8
        assert len(selected) == summary['available']
        assert sum(abs(r['single_error_m']) < 10 for r in rows) == summary['single_under_10m']
        assert sum(abs(r['selected_error_m']) < 10 for r in selected) == summary['selected_under_10m']
        assert sum(r['fit']['selected_model']=='ambiguous' for r in rows) == summary['ambiguous']
        assert sum(r['fit']['selected_model']=='two_path' for r in rows) == summary['two_path_selected']
        for metric, field, subset in (
            ('single_rmse_m', 'single_error_m', rows),
            ('raw_two_path_rmse_m', 'raw_two_path_error_m', rows),
            ('selected_rmse_available_m', 'selected_error_m', selected)):
            if subset:
                assert np.isclose(summary[metric], np.sqrt(np.mean([r[field]**2 for r in subset])))
            else:
                assert summary[metric] is None
    for i in (0, 159):
        replayed = json.loads(json.dumps(evaluate(protocol['tasks'][i])))
        assert replayed == records[i]
    report = dict(records=160, cells=20, source_hashes_verified=True,
                  all_success_denominators_and_rmse_verified=True,
                  exact_replay_record_indices=[0, 159], numpy_version=np.__version__,
                  scipy_version=scipy.__version__,
                  limitation='Two replays only; same implementation, not independent receiver validation')
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
