"""Frozen residual-weighted receiver under common residual carrier mismatch."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from gnss_gpu.dynamic_multipath_cpu import temporal_echo
from gnss_gpu.multipath_information_cpu import CHIP_RATE, ca_passband_spectrum
from gnss_gpu.residual_dynamic_paths_cpu import score_residual_dynamic_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.stress_cpu_dynamic_noise import synthesize as baseline
from experiments.run_cpu_dynamic_selection import select, summarize as summarize_single, digest

OUTPUT = Path('output/cpu-residual-carrier-stress')
PRIOR = Path('output/cpu-residual-ranked')
OFFSETS = (0., .1, .5, 1.)


def synthesize(task):
    ratio, offset, replica = task
    original, old_signal, delay = baseline(('white', ratio, .7, 0., replica))
    if offset == 0:
        return original, old_signal, delay
    # Reproduce only nuisance truth. Absolute echo frequency is common + relative.
    rng = np.random.default_rng(np.random.SeedSequence([20261026, 0x44594741, replica]))
    assert float(rng.uniform(-.3, .3)) == delay
    common = float(rng.uniform(-np.pi, np.pi))
    f, s = ca_passband_spectrum(3.6 * CHIP_RATE)
    direct = s * np.exp(-2j*np.pi*f*delay/CHIP_RATE + 1j*common)
    signal = direct * (temporal_echo(offset)[0][:, None] + ratio *
        temporal_echo(offset + .7)[0][:, None] * np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    return signal + (original - old_signal), signal, delay


def evaluate(task):
    blocks, _, delay = synthesize(task)
    score, preliminary, weights = score_residual_dynamic_paths(blocks)
    ratio, offset, replica = task
    return dict(echo_voltage=ratio, residual_carrier_hz=offset, doppler_hz=.7,
        phase_rad=0., replica=replica, true_delay_chips=delay,
        score=asdict(score), preliminary=asdict(preliminary),
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells = []
    for offset in OFFSETS:
        group = [r for r in rows if r['residual_carrier_hz'] == offset]
        cells.extend(dict(c, residual_carrier_hz=offset) for c in summarize_single(group)['cells'])
    return dict(cells=cells)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError('output must be empty')
    sources = dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())] = digest(__file__)
    assert all(digest(p) == h for p, h in sources.items())
    gatepath = PRIOR/'frozen_gates.json'
    gate = json.loads(gatepath.read_text())['gates']['ranked']
    tasks = [(ratio, offset, 760000 + 1000*j + r) for j, ratio in enumerate((0., .8))
             for offset in OFFSETS for r in range(24)]
    protocol = dict(evaluation_tasks=tasks, source_sha256=sources,
        gate_path=str(gatepath), gate_sha256=digest(gatepath), gate=gate,
        context='PRN1, white noise,37 reference dB-Hz,10x20ms,3.6828MHz; echo gap .1chip, voltage .8, relative Doppler .7Hz, initial phase0.',
        intervention='Common residual carrier 0/.1/.5/1Hz. Integrate direct at offset and echo at offset+.7Hz independently using exact rectangular-block sinc factors. Preserve the same additive noise and delay across offsets.',
        inference='Exploratory descriptive 24 trials/cell. No significance or 1% certification claims. Same replicas paired across offsets; do not pool.',
        scope='Frozen original ranked gate and receiver, no retuning. Zero-carrier calibration does not guarantee false-path control for nonzero residual carrier.',
        limits='Narrowband frozen code delay, independent harmonic noise; no interharmonic leakage, tracking loop, navigation bits, PVT or real forest/rain.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool, (OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate, tasks):
            row = select(raw, PathSelectionGate(**gate))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(rows) % 24 == 0:
                stream.flush()
                print(f'Evaluation {len(rows)}/{len(tasks)}', flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows), indent=2))
    assert all(digest(p) == h for p, h in sources.items())
    assert digest(gatepath) == protocol['gate_sha256']
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'), report_sha256=digest(OUTPUT/'report.json')), indent=2))


if __name__ == '__main__':
    run()
