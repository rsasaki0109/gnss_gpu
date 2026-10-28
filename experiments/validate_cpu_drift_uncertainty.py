"""Fresh replication of close destructive echoes and local uncertainty coverage."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from scipy.stats import binomtest
from experiments.stress_cpu_drift_gap_amplitude import evaluate, tasks, apply_gate, digest, GATE, PathSelectionGate
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED, CHIP_RATE

OUTPUT = Path('output/cpu-drift-uncertainty')
PRIOR = Path('output/cpu-drift-gap-amplitude')
Z = 1.959963984540054


def evaluation_tasks():
    return ([(k, 0., .03, 0., a, f, i) for k, a, f, i in tasks('null', 1080000, 32)]
            + [(k, .2, .03, np.pi, a, f, i) for k, a, f, i in tasks('echo', 1090000, 32)]
            + [(k, .8, .03, np.pi, a, f, i) for k, a, f, i in tasks('echo', 1100000, 64)])


def uncertainty(row):
    """Report local Gaussian intervals; never use truth to select/guard a fit."""
    model = row['selected']['model']
    if model == 'ambiguous':
        return dict(available=False, sigma_m=None, half_width_m=None, covers_truth=False,
                    narrow=False, narrow_large_error=False)
    fit = row['fits']['two' if model == 'two_path' else 'single']
    sigma = fit['first_std_chips'] * LIGHT_SPEED / CHIP_RATE
    half = Z * sigma
    error = abs(row['error_m'])
    return dict(available=True, sigma_m=sigma, half_width_m=half,
                covers_truth=error <= half, narrow=half < 10,
                narrow_large_error=half < 10 and error >= 10)


def summarize(rows):
    cells = []
    for kind, ratio in [('null', 0.), ('echo', .2), ('echo', .8)]:
        group = [r for r in rows if r['kind'] == kind and r['echo_voltage'] == ratio]
        for model in ('all', 'single_path', 'two_path', 'ambiguous'):
            g = [r for r in group if model == 'all' or r['selected']['model'] == model]
            n = len(g)
            available = sum(r['uncertainty']['available'] for r in g)
            covered = sum(r['uncertainty']['covers_truth'] for r in g)
            successes = sum(r['success'] for r in g)
            cells.append(dict(kind=kind, echo_voltage=ratio, model=model, attempts=n,
                available=available, successes=successes,
                success_exact_95=list(binomtest(successes,n).proportion_ci(method='exact')) if n else None,
                covered=covered,
                coverage_available_exact_95=list(binomtest(covered,available).proportion_ci(method='exact')) if available else None,
                narrow=sum(r['uncertainty']['narrow'] for r in g),
                narrow_large_error=sum(r['uncertainty']['narrow_large_error'] for r in g)))
    return dict(cells=cells, interpretation='Local Hessian Gaussian intervals are diagnostic, not calibrated confidence intervals. Coverage conditional on selected model is descriptive; unavailable cases count as failures in all-attempt range success. Pointwise exact intervals only.')


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError('output must be empty')
    sources = dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())] = digest(__file__)
    assert all(digest(p) == h for p, h in sources.items())
    gate = PathSelectionGate(**json.loads(GATE.read_text()))
    tasklist = evaluation_tasks()
    protocol = dict(source_sha256=sources, gate_path=str(GATE), gate_sha256=digest(GATE),
        gate=asdict(gate), evaluation_tasks=tasklist, gaussian_multiplier=Z,
        endpoints='Frozen receiver and gate. Fresh 64 strong destructive close echoes,32 weak controls,32 nulls; disjoint IDs, no paired noise between cells. No retuning. Report |error|<10m, availability, local +/-1.95996 sigma coverage, and |error|>=10m despite half-width<10m.',
        conditions='800ms,PRN1,white37dBHz,one-sided3.6828MHz,relative Doppler0Hz,echo gap.03chip and phasepi. Random common carrier[-1,1]Hz and drift[-2,2]Hz/s.',
        limits='Synthetic isolated satellite,ideal coherent integration,no RF/geometry/PVT. Local curvature excludes model ambiguity/bias. Coverage by selected model is not a population guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool, (OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate, tasklist):
            row = apply_gate(raw, gate)
            row['uncertainty'] = uncertainty(row)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(rows) % 16 == 0:
                stream.flush()
                print(f'Evaluation {len(rows)}/{len(tasklist)}', flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows), indent=2))
    assert all(digest(p) == h for p,h in sources.items()) and digest(GATE) == protocol['gate_sha256']
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),
        protocol_sha256=digest(OUTPUT/'protocol.json'), evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),
        report_sha256=digest(OUTPUT/'report.json')), indent=2))


if __name__ == '__main__':
    run()
