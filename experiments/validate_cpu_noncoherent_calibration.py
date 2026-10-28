"""Larger, independently seeded null validation of fixed noncoherent searches.

Write the protocol before calibration and freeze thresholds before evaluation.
Signal scores are rethresholded from the previous experiment, not new trials.
"""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT
from experiments.run_cpu_receiver_stress import NOISE_DB, seed_for
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from experiments.run_cpu_noncoherent_sensitivity import HORIZON_MS, aggregate
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, acquire_ca_noncoherent, calibrate_ca_noncoherent
from gnss_gpu.input_validation import positive_int

CALIBRATION_DOMAIN, EVALUATION_DOMAIN = 0x4E435632, 0x4E435633


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_null(outcomes):
    if not outcomes:
        raise ValueError("at least one evaluation record required")
    n = len(outcomes)
    summary = {str(k): dict(records=n, accepted=sum(row[str(k)] for row in outcomes)) for k in (1, 4)}
    for values in summary.values():
        values["fraction"] = values["accepted"]/n
        values["pointwise_wilson_95"] = wilson_interval(values["accepted"], n)
    paired = {"neither": 0, "both": 0, "only_1ms": 0, "only_4ms": 0}
    for row in outcomes:
        a, b = row["1"], row["4"]
        paired["both" if a and b else "only_1ms" if a else "only_4ms" if b else "neither"] += 1
    return dict(per_detector=summary, paired_counts=paired,
                observed_rate_difference_4ms_minus_1ms=summary["4"]["fraction"]-summary["1"]["fraction"])


def rethreshold(records, thresholds, criterion):
    updated = []
    for row in records:
        detectors = {}
        for k, threshold in thresholds.items():
            before = row["detectors"][str(k)]
            after = dict(before)
            candidate = before["candidate"]
            if candidate is not None and candidate["statistic"] != before["statistic"]:
                raise ValueError("candidate and decision statistics differ")
            after["accepted"] = candidate is not None and before["statistic"] > threshold.calibration.score
            after["passes_declared_criterion"] = bool(after["accepted"] and
                before["carrier_error_hz"] <= criterion["max_carrier_error_hz"] and
                before["initial_phase_lag_error_samples"] <= criterion["max_initial_phase_lag_error_samples"])
            detectors[str(k)] = after
        updated.append(dict(row, detectors=detectors))
    return updated


def run(prior_dir, output, *, calibration_records=1999, evaluation_records=1999):
    calibration_records = positive_int("calibration_records", calibration_records)
    evaluation_records = positive_int("evaluation_records", evaluation_records)
    if calibration_records < 99:
        raise ValueError("at least 99 calibration records required for 1% rank target")
    prior_dir, output = Path(prior_dir), Path(output)
    if prior_dir.resolve() == output.resolve():
        raise ValueError("output must be separate from the prior experiment")
    prior = json.loads((prior_dir/"report.json").read_text())
    root = Path(__file__).resolve().parents[1]
    # Reusing cached candidate scores is valid only with the original algorithm.
    for path, expected in prior["source_sha256"].items():
        if digest(root/path) != expected:
            raise ValueError(f"prior experiment source changed: {path}")
    configs = {k: CaNoncoherentSearch(CaSearch(FS, WIDE_GRID), k, HORIZON_MS//k) for k in (1, 4)}
    for k, config in configs.items():
        if json.loads(json.dumps(asdict(config))) != prior["thresholds"][str(k)]["config"]:
            raise ValueError("prior decision configuration differs")
    inputs = {name: digest(prior_dir/name) for name in ("report.json", "decisions.jsonl")}
    sources = {"experiments/validate_cpu_noncoherent_calibration.py": digest(Path(__file__))}
    sources.update({path: digest(root/path) for path in prior["source_sha256"]})
    protocol = dict(calibration_records=calibration_records, evaluation_records=evaluation_records,
        horizon_ms=HORIZON_MS, configs={k: asdict(value) for k, value in configs.items()},
        target_false_alarm_probability=.01, sampling_freq_hz=FS, noise_floor_db=NOISE_DB,
        noise_domains=dict(calibration=CALIBRATION_DOMAIN, evaluation=EVALUATION_DOMAIN),
        prior_input_sha256=inputs, source_sha256=sources,
        signal_data_role="Previously observed signal records, rethresholding only; no new signal validation.",
        evaluation_policy="Freeze thresholds before evaluation; do not tune on evaluation outcomes.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    def noise_record(domain, record):
        rows = []
        for block in range(HORIZON_MS):
            simulator.noise_seed = seed_for(domain, record, block)
            rows.append(simulator.generate_epoch([], COUNT))
        return np.stack(rows)
    calibration = {k: [] for k in configs}
    with (output/"calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record", "iq_sha256", "score_1ms", "score_4ms"))
        writer.writeheader()
        for index in range(calibration_records):
            iq = noise_record(CALIBRATION_DOMAIN, index)
            values = {k: acquire_ca_noncoherent(iq, config).statistic for k, config in configs.items()}
            for k, value in values.items():
                calibration[k].append(value)
            writer.writerow(dict(record=index, iq_sha256=digest_bytes(iq.tobytes()),
                                 score_1ms=values[1], score_4ms=values[4]))
            if (index+1) % 250 == 0:
                handle.flush()
                print(f"calibration {index+1}/{calibration_records}", flush=True)
    thresholds = {k: calibrate_ca_noncoherent(values, configs[k]) for k, values in calibration.items()}
    frozen_path = output/"frozen_thresholds.json"
    frozen_path.write_text(json.dumps({k: asdict(value) for k, value in thresholds.items()}, indent=2)+"\n")
    frozen_hash = digest(frozen_path)
    outcomes = []
    with (output/"evaluation.jsonl").open("w") as handle:
        for index in range(evaluation_records):
            iq = noise_record(EVALUATION_DOMAIN, index)
            candidates = {k: acquire_ca_noncoherent(iq, config) for k, config in configs.items()}
            accepted = {str(k): candidate.statistic > thresholds[k].calibration.score for k, candidate in candidates.items()}
            outcomes.append(accepted)
            handle.write(json.dumps(dict(record=index, iq_sha256=digest_bytes(iq.tobytes()), accepted=accepted,
                                        candidates={k: asdict(candidate) for k, candidate in candidates.items()}))+"\n")
            if (index+1) % 250 == 0:
                handle.flush()
                print(f"evaluation {index+1}/{evaluation_records}", flush=True)
    if digest(frozen_path) != frozen_hash:
        raise ValueError("frozen thresholds changed during evaluation")
    original_records = [json.loads(line) for line in (prior_dir/"decisions.jsonl").read_text().splitlines()]
    signals = rethreshold(original_records, thresholds, prior["criterion"])
    (output/"signal_decisions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in signals))
    (output/"aggregate.json").write_text(json.dumps(aggregate(signals), indent=2)+"\n")
    for path, expected in sources.items():
        if digest(root/path) != expected:
            raise ValueError(f"source changed during experiment: {path}")
    for name, expected in inputs.items():
        if digest(prior_dir/name) != expected:
            raise ValueError("prior input changed during experiment")
    report = dict(protocol_sha256=digest(output/"protocol.json"), frozen_thresholds_sha256=frozen_hash,
        null_validation=summarize_null(outcomes), rethresholded_signal_records=len(signals),
        prior_thresholds={k: prior["thresholds"][str(k)]["calibration"]["score"] for k in configs},
        new_thresholds={k: value.calibration.score for k, value in thresholds.items()},
        limitations=["One new calibration set per method; no repeated-calibration population validation.",
          "Wilson intervals describe each frozen threshold's null rate under this simulated white noise.",
          "Intervals are pointwise and overlap is not proof of equivalence or matched false-alarm rates.",
          "Signal scores were previously observed and only rethresholded, not independently validated.",
          "Same integer-delay and sampling-model limitations; no sustained tracking, PVT, rain or RF claim."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, default=Path("output/cpu-noncoherent-sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("output/cpu-noncoherent-calibration-validation"))
    parser.add_argument("--calibration-records", type=int, default=1999)
    parser.add_argument("--evaluation-records", type=int, default=1999)
    args = parser.parse_args()
    print(json.dumps(run(args.prior_dir, args.output, calibration_records=args.calibration_records,
                         evaluation_records=args.evaluation_records), indent=2))
