from pathlib import Path
import json,hashlib
import numpy as np
from experiments.run_cpu_drift_bandwidth import OUTPUT,digest,tasklist,summarize,synthesize,evaluate,BANDS,CHIP_RATE,LIGHT_SPEED


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text());c=json.loads((OUTPUT/'completion.json').read_text())
    assert all(digest(path)==h for path,h in p['source_sha256'].items())
    assert all(digest(OUTPUT/name)==h for name,h in c.items())
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==48 and [r['task'] for r in rows]==p['tasks']==json.loads(json.dumps(tasklist()))
    assert summarize(rows)==json.loads((OUTPUT/'report.json').read_text())
    ids={r['replica'] for r in rows};assert len(ids)==16
    for pattern in ('cpu-drift-*/*.jsonl','cpu-dominant-*/*.jsonl'):
        for path in Path('output').glob(pattern):
            if path.parent==OUTPUT:continue
            assert ids.isdisjoint(r.get('replica') for r in map(json.loads,path.read_text().splitlines()))
    for row in rows:
        b,_,truth=synthesize(row['task'])
        assert hashlib.sha256(b.tobytes()).hexdigest()==row['observation_sha256']
        fit=row['fits']['two']
        error=(fit['first_delay_chips']-truth)*LIGHT_SPEED/CHIP_RATE
        assert error==row['raw_error_m'] and row['available']==fit['locally_usable']
        assert row['success']==bool(fit['locally_usable'] and abs(error)<10)
    from gnss_gpu.multipath_information_cpu import ca_passband_spectrum
    fw,_=ca_passband_spectrum(7.2*CHIP_RATE)
    for i in range(16):
        wide,ws,wt=synthesize(rows[32+i]['task'])
        for offset in (0,16):
            row=rows[offset+i];b,s,t=synthesize(row['task']);keep=abs(fw)<=row['bandwidth_chips']*CHIP_RATE
            assert t==wt
            np.testing.assert_array_equal(b,wide[:,keep]);np.testing.assert_array_equal(s,ws[:,keep])
    timings=json.loads((OUTPUT/'timings.json').read_text())
    assert [(r['replica'],r['bandwidth_chips']) for r in timings]==[(1240000,b) for b in BANDS]+[(1240001,b) for b in BANDS[::-1]]
    assert all(np.isfinite(r['elapsed_s']) and r['elapsed_s']>0 for r in timings)
    for i in (0,16,32):assert json.loads(json.dumps(evaluate(p['tasks'][i])))==rows[i]
    (OUTPUT/'audit.json').write_text(json.dumps(dict(evaluation=48,sources_and_artifacts_verified=True,
        prior_ids_disjoint=True,nested_band_checks=32,all_endpoints_recomputed=True,full_replays=[0,16,32],timing_records_checked=6),indent=2))
    print('Audit passed:48 observations,32 exact nested-band checks,3 full replays')


if __name__=='__main__':run()
