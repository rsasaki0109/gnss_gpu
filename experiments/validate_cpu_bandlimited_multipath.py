"""Frozen-receiver replication with fresh seed IDs and paired success inference."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import binomtest

from experiments.run_cpu_bandlimited_multipath import evaluate

OUTPUT = Path('output/cpu-bandlimited-multipath-replication')
KEYS = ('bandwidth_chip_rates', 'cn0_dbhz', 'echo_voltage', 'gap_chips', 'phase_rad')


def paired_summary(baseline, selected):
    baseline, selected = np.asarray(baseline, dtype=bool), np.asarray(selected, dtype=bool)
    if baseline.ndim != 1 or baseline.shape != selected.shape or not len(baseline):
        raise ValueError('paired nonempty one-dimensional observations required')
    n = len(baseline)
    gains = int(np.sum(~baseline & selected))
    losses = int(np.sum(baseline & ~selected))
    # Exact two-sided McNemar test, conditioned on discordant observations.
    p = float(binomtest(gains, gains+losses).pvalue) if gains+losses else 1.
    intervals = [binomtest(int(x.sum()), n).proportion_ci(confidence_level=.95, method='exact')
                 for x in (baseline, selected)]
    # Paired percentile bootstrap; three categories represent each replica's
    # selected-minus-baseline binary difference. Descriptive pointwise CI only.
    rng = np.random.default_rng(0x42535450)
    draws = rng.multinomial(n, [gains/n, losses/n, 1-(gains+losses)/n], size=20000)
    ci = np.quantile((draws[:, 0]-draws[:, 1])/n, [.025, .975])
    return dict(attempts=n, baseline_successes=int(baseline.sum()),
                selected_successes=int(selected.sum()), gains=gains, losses=losses,
                difference=float((gains-losses)/n), paired_exact_p=p,
                baseline_exact_95=list(intervals[0]), selected_exact_95=list(intervals[1]),
                paired_bootstrap_pointwise_95=ci.tolist())


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    summaries = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: r['replica'])
        baseline = [abs(r['single_error_m']) < 10 for r in rows]
        selected = [r['selected_error_m'] is not None and abs(r['selected_error_m']) < 10 for r in rows]
        paired = paired_summary(baseline, selected)
        primary = key[0] == 3.6 and key[1] == 37. and key[2] == .8 and key[3] == .5
        summaries.append(dict(zip(KEYS, key), **paired,
            available=sum(r['selected_error_m'] is not None for r in rows),
            two_path_selected=sum(r['fit']['selected_model']=='two_path' for r in rows),
            ambiguous=sum(r['fit']['selected_model']=='ambiguous' for r in rows),
            primary=primary,
            primary_bonferroni_p=min(1., 2*paired['paired_exact_p']) if primary else None,
            primary_improvement_supported=(paired['difference'] > 0 and paired['paired_exact_p'] < .025) if primary else None,
            single_rmse_m=float(np.sqrt(np.mean([r['single_error_m']**2 for r in rows]))),
            raw_two_path_rmse_m=float(np.sqrt(np.mean([r['raw_two_path_error_m']**2 for r in rows])))))
    return summaries


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError('output must be empty')
    old_protocol = json.loads(Path('output/cpu-bandlimited-multipath/protocol.json').read_text())
    for path, digest in old_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    cells = [(0., 0., 0.)]+[(.8, g, p) for g in (.1, .5) for p in (0., np.pi/2)]
    tasks = [(b, cn0, a, g, p, r) for b in (.9, 3.6) for cn0 in (37., 55.)
             for a, g, p in cells for r in range(10000, 10128)]
    sources = dict(old_protocol['source_sha256'])
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path('output/cpu-bandlimited-multipath').iterdir() if p.is_file()}
    protocol = dict(tasks=tasks, replicas_per_cell=128, replica_ids=[10000,10127],
        development_replica_ids=[0,7], seed_domain=old_protocol['seed_domain'],
        source_sha256=sources, previous_artifact_sha256=prior_hashes,
        primary='Selected vs single; B=3.6 chip rates, C/N0=37, gap=.5 chip, phases 0 and pi/2',
        family='Two primary paired exact two-sided McNemar tests, Bonferroni family alpha=.05',
        success='Available absolute direct range error <10 m; all attempts in denominator',
        intervals='Pointwise exact binomial 95%; paired bootstrap percentile 95%, 20000 draws, seed 0x42535450; not simultaneous',
        independence='128 independent replica IDs per cell, shared across cells; not 2560 independent replicas',
        scope='Same ideal assisted spectral simulation and estimator; new randomness, no external physical validation',
        numpy_version=np.__version__, scipy_version=scipy.__version__)
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False))
    records = []
    with (OUTPUT/'records.jsonl').open('w') as stream, ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate, tasks):
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(records) % 128 == 0:
                stream.flush()
                print(f'{len(records)}/{len(tasks)} records', flush=True)
    report = summarize(records)
    for path, digest in {**sources, **prior_hashes}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cn0 in zip(axes, (37., 55.)):
        for j, bandwidth in enumerate((.9, 3.6)):
            rows = [r for r in report if r['cn0_dbhz']==cn0 and r['bandwidth_chip_rates']==bandwidth]
            y = np.array([r['selected_successes']/r['attempts'] for r in rows])
            bounds = np.array([r['selected_exact_95'] for r in rows]).T
            ax.errorbar(np.arange(5)+(j-.5)*.15, y, yerr=[y-bounds[0], bounds[1]-y],
                        fmt='o', capsize=3, label=f'B={bandwidth*1.023:.3f} MHz')
        ax.set_xticks(np.arange(5), ['Direct only', '.1 chip\nphase 0', '.1 chip\npi/2', '.5 chip\nphase 0', '.5 chip\npi/2'])
        ax.set_title(f'{cn0:g} dB-Hz, 128 fresh replicas per cell')
        ax.set_ylim(-.04, 1.04)
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Available range error <10 m: proportion')
    fig.suptitle('Fixed assisted estimator; pointwise exact 95% intervals; failures retained')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'comparison.{extension}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run()
