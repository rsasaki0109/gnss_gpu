"""Fresh paired signals and nulls with a common conditional false-alarm budget.

The two true false-alarm rates need not be equal. Tolerance calibration controls
an upper budget under IID representative white-noise null records only.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT, render
from experiments.run_cpu_receiver_stress import NOISE_DB, make_scene, seed_for
from experiments.run_cpu_noncoherent_sensitivity import score_detection, aggregate
from experiments.run_cpu_receiver_sensitivity import CN0_LEVELS, DOPPLERS
from experiments.validate_cpu_noncoherent_calibration import summarize_null
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.signal_sim_cpu import L1_HZ
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, acquire_ca_noncoherent
from gnss_gpu.noncoherent_tolerance_cpu import calibrate_ca_noncoherent_tolerance, detect_ca_noncoherent_tolerance
from gnss_gpu.input_validation import positive_int

CALIBRATION_DOMAIN, EVALUATION_DOMAIN = 0x42544341, 0x42544556
SIGNAL_DOMAIN, OFFSET_DOMAIN = 0x42545349, 0x42544F46
HORIZON_MS, REPLICA_OFFSET = 20, 1000


def signal_scene(cn0, doppler, replica, bit_mode):
    if bit_mode not in ("constant", "flip"):
        raise ValueError("bit_mode must be constant or flip")
    base, _, _, _ = make_scene(cn0, doppler, 8, replica, 64)
    rng = np.random.default_rng(seed_for(OFFSET_DOMAIN, replica, 0))
    offset = int(rng.integers(1, 20*COUNT))
    while offset % COUNT == 0:
        offset = int(rng.integers(1, 20*COUNT))
    channels = []
    for source in base[0]:
        channel = dict(source)
        channel.pop("nav_bits", None)
        if channel["prn"] == 1 and bit_mode == "flip":
            channel["nav_bits"] = [1, -1, 1]
        channels.append(channel)
    rows = [[dict(channel) for channel in channels] for _ in range(HORIZON_MS)]
    starts = [offset+i*COUNT for i in range(HORIZON_MS)]
    seeds = [seed_for(SIGNAL_DOMAIN, replica, i) for i in range(HORIZON_MS)]
    nominal_delay = -2*channels[0]["code_phase"]-offset*(1+doppler/L1_HZ)
    return rows, starts, seeds, nominal_delay, 20*COUNT-offset


def exact_paired_sign_p(gains, losses):
    if gains < 0 or losses < 0 or int(gains) != gains or int(losses) != losses:
        raise ValueError("discordant counts must be nonnegative integers")
    n = int(gains+losses)
    if not n:
        return 1.
    return min(1., 2*sum(math.comb(n, j) for j in range(int(min(gains, losses))+1))/2**n)


def summarize_signals(records):
    grouped, paired = [], []
    for mode in ("constant", "flip"):
        selected = [row for row in records if row["bit_mode"] == mode]
        grouped.extend(dict(bit_mode=mode, **row) for row in aggregate(selected))
        for cn0 in CN0_LEVELS:
            for doppler in DOPPLERS:
                subset = [row for row in selected if row["nominal_primary_cn0_db_hz"] == cn0 and row["primary_doppler_hz"] == doppler]
                if not subset:
                    continue
                gains = sum(not row["detectors"]["1"]["passes_declared_criterion"] and row["detectors"]["4"]["passes_declared_criterion"] for row in subset)
                losses = sum(row["detectors"]["1"]["passes_declared_criterion"] and not row["detectors"]["4"]["passes_declared_criterion"] for row in subset)
                p = exact_paired_sign_p(gains, losses)
                primary = mode == "flip" and cn0 == 37.
                paired.append(dict(bit_mode=mode, nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                    replicas=len(subset), only_4ms=gains, only_1ms=losses, exact_two_sided_p=p,
                    primary_endpoint=primary, primary_bonferroni_p=min(1., 2*p) if primary else None))
    return grouped, paired


def run(output, *, calibration_records=4999, evaluation_records=4999, replicas=32):
    calibration_records = positive_int("calibration_records", calibration_records)
    evaluation_records = positive_int("evaluation_records", evaluation_records)
    replicas = positive_int("replicas", replicas)
    # Fail before any generation if the requested confidence cannot be supported.
    from gnss_gpu.noncoherent_tolerance_cpu import tolerance_order
    tolerance_order(calibration_records)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    configs = {k: CaNoncoherentSearch(CaSearch(FS, WIDE_GRID), k, HORIZON_MS//k) for k in (1, 4)}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__)]
    paths += [root/"experiments"/(name+".py") for name in ("run_cpu_batch_receiver", "run_cpu_receiver_stress", "run_cpu_noncoherent_sensitivity", "run_cpu_receiver_sensitivity", "validate_cpu_noncoherent_calibration")]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in ("noncoherent_tolerance_cpu", "noncoherent_acquisition_cpu", "signal_sim_cpu", "code_delay_cpu", "code_detection_cpu", "acquisition_cpu")]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    protocol = dict(calibration_records=calibration_records, evaluation_records=evaluation_records, replicas=replicas,
        replica_offset=REPLICA_OFFSET, horizon_ms=HORIZON_MS, configs={k: asdict(value) for k, value in configs.items()},
        noise_floor_db=NOISE_DB, sampling_freq_hz=FS,
        null_definition="All signals absent; IID simulated Gaussian white-noise records.",
        max_conditional_false_alarm_probability=.01, failure_probability_per_method=.025,
        joint_calibration_failure_budget=.05, noise_domains=dict(calibration=CALIBRATION_DOMAIN, evaluation=EVALUATION_DOMAIN,
                                                               signal=SIGNAL_DOMAIN, offset=OFFSET_DOMAIN),
        bit_modes=["constant", "flip"], signal_levels_db_hz=CN0_LEVELS, doppler_hz=DOPPLERS,
        bit_challenge="One forced PRN-1 sign flip inside a 1-ms period; other PRNs have constant nav polarity.",
        primary_endpoints=dict(cn0_db_hz=37., doppler_hz=DOPPLERS, bit_mode="flip", test="two-sided exact paired sign test, Bonferroni factor 2"),
        criterion=dict(max_carrier_error_hz=250., max_nominal_phase_lag_error_samples=2.),
        source_sha256=source_hashes, evaluation_policy="Freeze thresholds before evaluation; no retuning on evaluation results.")
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    def noise_record(domain, index):
        rows = []
        for block in range(HORIZON_MS):
            simulator.noise_seed = seed_for(domain, index, block)
            rows.append(simulator.generate_epoch([], COUNT))
        return np.stack(rows)
    scores = {k: [] for k in configs}
    with (output/"calibration.jsonl").open("w") as handle:
        for index in range(calibration_records):
            iq = noise_record(CALIBRATION_DOMAIN, index)
            values = {k: acquire_ca_noncoherent(iq, config).statistic for k, config in configs.items()}
            for k in scores:
                scores[k].append(values[k])
            handle.write(json.dumps(dict(record=index, scores=values, iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))+"\n")
            if (index+1) % 500 == 0:
                handle.flush(); print(f"calibration {index+1}/{calibration_records}", flush=True)
    thresholds = {k: calibrate_ca_noncoherent_tolerance(values, configs[k]) for k, values in scores.items()}
    frozen_path = output/"frozen_thresholds.json"
    frozen_path.write_text(json.dumps({k: asdict(value) for k, value in thresholds.items()}, indent=2)+"\n")
    frozen_hash = digest(frozen_path)
    outcomes = []
    with (output/"evaluation.jsonl").open("w") as handle:
        for index in range(evaluation_records):
            iq = noise_record(EVALUATION_DOMAIN, index)
            detections = {k: detect_ca_noncoherent_tolerance(iq, threshold) for k, threshold in thresholds.items()}
            outcomes.append({str(k): value.accepted for k, value in detections.items()})
            handle.write(json.dumps(dict(record=index, detectors={k: asdict(value) for k, value in detections.items()},
                                        iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))+"\n")
            if (index+1) % 500 == 0:
                handle.flush(); print(f"evaluation {index+1}/{evaluation_records}", flush=True)
    signals = []
    with (output/"signals.jsonl").open("w") as handle:
        for cn0 in CN0_LEVELS:
            for doppler in DOPPLERS:
                for replica in range(REPLICA_OFFSET, REPLICA_OFFSET+replicas):
                    for mode in ("constant", "flip"):
                        rows, starts, seeds, nominal, flip = signal_scene(cn0, doppler, replica, mode)
                        iq = render("batch", rows, starts, seeds, NOISE_DB)
                        result = dict(nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                            replica=replica, bit_mode=mode, start_sample=starts[0], hypothetical_flip_sample=flip,
                            iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),
                            detectors={str(k): score_detection(detect_ca_noncoherent_tolerance(iq, threshold), doppler, nominal)
                                       for k, threshold in thresholds.items()})
                        signals.append(result); handle.write(json.dumps(result)+"\n")
                handle.flush(); print(f"signals {cn0:g} dB-Hz / {doppler:g} Hz complete", flush=True)
    grouped, paired = summarize_signals(signals)
    (output/"aggregate.json").write_text(json.dumps(grouped, indent=2)+"\n")
    (output/"paired.json").write_text(json.dumps(paired, indent=2)+"\n")
    if digest(frozen_path) != frozen_hash or any(digest(root/path) != expected for path, expected in source_hashes.items()):
        raise ValueError("source or frozen thresholds changed during the experiment")
    report = dict(protocol_sha256=digest(output/"protocol.json"), frozen_thresholds_sha256=frozen_hash,
        null_validation=summarize_null(outcomes), signal_records=len(signals),
        primary_results=[row for row in paired if row["primary_endpoint"]],
        limitations=["Common upper false-alarm budget under IID white noise, not equality of true false-alarm rates.",
          "One calibration set per method; theoretical tolerance guarantee assumes representative IID nulls.",
          "Forced single PRN-1 bit reversal is a stress control, not random complete navigation messages.",
          "Other PRNs have constant bit signs; no target-absent-with-interferers false-alarm guarantee.",
          "Coarse carrier/nominal phase criterion, not fractional ranging, tracking, PVT, rain or RF validation.",
          "Primary endpoints were chosen from earlier exploration and fixed before these fresh signals; no novelty claim."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-noncoherent-bit-validation"))
    parser.add_argument("--calibration-records", type=int, default=4999)
    parser.add_argument("--evaluation-records", type=int, default=4999)
    parser.add_argument("--replicas", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output, calibration_records=args.calibration_records,
                         evaluation_records=args.evaluation_records, replicas=args.replicas), indent=2))
