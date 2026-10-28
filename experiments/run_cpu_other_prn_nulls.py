"""Target-absent satellite cross-correlation stress and fixed stratified gates."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT, render
from experiments.run_cpu_receiver_stress import NOISE_DB, seed_for
from experiments.run_cpu_noncoherent_sensitivity import score_detection, aggregate
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.signal_sim_cpu import L1_HZ
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, CaNoncoherentDetection, acquire_ca_noncoherent
from gnss_gpu.stratified_tolerance_cpu import calibrate_ca_stratified_tolerance
from gnss_gpu.noncoherent_tolerance_cpu import tolerance_order
from gnss_gpu.input_validation import positive_int

PROFILES = {"white": None, "other_prns": 1., "near_far_20db": 10.}
CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN = 0x4F504341, 0x4F504556, 0x4F505349
HORIZON_MS = 20


def scene(profile, domain, index, *, target_cn0=None, target_doppler=170.):
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    rng = np.random.default_rng(seed_for(domain, index, HORIZON_MS))
    offset = int(rng.integers(1, 20*COUNT))
    while offset % COUNT == 0:
        offset = int(rng.integers(1, 20*COUNT))
    gain = PROFILES[profile]
    other = []
    for prn in range(2, 13):
        channel = dict(prn=prn, code_phase=float(rng.uniform(0., 1023.)),
            carrier_phase=float(rng.uniform(0., 2*np.pi)), doppler_hz=(-1.)**prn*prn*150.,
            amplitude=.4/prn*(gain if gain is not None else 0.), nav_bit=1)
        if gain is not None:
            other.append(channel)
    phase = float(rng.uniform(0., 2*np.pi))
    channels = other
    if target_cn0 is not None:
        channels = [dict(prn=1, code_phase=-18.5, carrier_phase=phase, doppler_hz=target_doppler,
                         amplitude=10**((target_cn0-55.)/20), nav_bit=1, nav_bits=[1, -1, 1])]+other
    rows = [[dict(channel) for channel in channels] for _ in range(HORIZON_MS)]
    starts = [offset+block*COUNT for block in range(HORIZON_MS)]
    seeds = [seed_for(domain, index, block) for block in range(HORIZON_MS)]
    nominal = 37.-offset*(1+target_doppler/L1_HZ)
    return rows, starts, seeds, nominal


def gate_candidate(candidate, threshold):
    score = candidate.statistic if candidate is not None else 0.
    return CaNoncoherentDetection(candidate is not None and score > threshold, score, threshold, candidate)


def run(output, baseline_path, *, calibration_records=999, evaluation_records=999, replicas=32):
    calibration_records = positive_int("calibration_records", calibration_records)
    evaluation_records = positive_int("evaluation_records", evaluation_records)
    replicas = positive_int("replicas", replicas)
    tolerance_order(calibration_records, failure_probability=.05/(2*len(PROFILES)))
    output, baseline_path = Path(output), Path(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    configs = {k: CaNoncoherentSearch(CaSearch(FS, WIDE_GRID), k, HORIZON_MS//k) for k in (1, 4)}
    for k, cfg in configs.items():
        if json.loads(json.dumps(asdict(cfg))) != baseline[str(k)]["config"]:
            raise ValueError("baseline search differs from fixed experiment config")
    white_scores = {k: baseline[str(k)]["score"] for k in configs}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__)]
    paths += [root/"experiments"/(name+".py") for name in ("run_cpu_batch_receiver", "run_cpu_receiver_stress", "run_cpu_noncoherent_sensitivity", "run_cpu_receiver_sensitivity")]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in ("stratified_tolerance_cpu", "noncoherent_tolerance_cpu", "noncoherent_acquisition_cpu", "signal_sim_cpu", "acquisition_cpu", "code_delay_cpu", "code_detection_cpu")]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    protocol = dict(calibration_records_per_profile=calibration_records, evaluation_records_per_profile=evaluation_records,
        signal_replicas=replicas, profiles=PROFILES, configs={k: asdict(cfg) for k, cfg in configs.items()},
        noise_domains=dict(calibration=CALIBRATION_DOMAIN, evaluation=EVALUATION_DOMAIN, signal=SIGNAL_DOMAIN),
        baseline_sha256=digest(baseline_path), source_sha256=hashes,
        signal_levels_db_hz=[37., 41., 45.], signal_dopplers_hz=[170., 700.],
        null_model="PRN 1 absent; PRNs 2..12 have fixed Dopplers and randomized phases/code delays; Gaussian white noise.",
        nuisance_polarity="Other PRN bit signs are constant; target-present controls have one forced within-period flip.",
        max_false_alarm_probability=.01, joint_calibration_failure_budget=.05,
        family_members=2*len(PROFILES), runtime_rule="One maximum score threshold per method; no profile label used in detection.",
        evaluation_policy="Freeze all thresholds before evaluation; no outcome-based retuning.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    calibration = {k: {profile: [] for profile in PROFILES} for k in configs}
    with (output/"calibration.jsonl").open("w") as handle:
        for profile in PROFILES:
            for index in range(calibration_records):
                rows, starts, seeds, _ = scene(profile, CALIBRATION_DOMAIN, index)
                iq = render("batch", rows, starts, seeds, NOISE_DB)
                values = {k: acquire_ca_noncoherent(iq, cfg).statistic for k, cfg in configs.items()}
                for k, value in values.items():
                    calibration[k][profile].append(value)
                handle.write(json.dumps(dict(profile=profile, record=index, scores=values,
                                            iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))+"\n")
                if (index+1) % 250 == 0:
                    handle.flush(); print(f"calibration {profile}: {index+1}/{calibration_records}", flush=True)
    thresholds = {k: calibrate_ca_stratified_tolerance(values, configs[k]) for k, values in calibration.items()}
    gates = {"white_calibrated": white_scores, "stratified": {k: value.score for k, value in thresholds.items()}}
    frozen_path = output/"frozen_thresholds.json"
    frozen_path.write_text(json.dumps(dict(gates=gates, stratified={k: asdict(value) for k, value in thresholds.items()}), indent=2)+"\n")
    frozen_hash = digest(frozen_path)
    counts = {gate: {profile: {str(k): 0 for k in configs} for profile in PROFILES} for gate in gates}
    with (output/"evaluation.jsonl").open("w") as handle:
        for profile in PROFILES:
            for index in range(evaluation_records):
                rows, starts, seeds, _ = scene(profile, EVALUATION_DOMAIN, index)
                iq = render("batch", rows, starts, seeds, NOISE_DB)
                candidates = {k: acquire_ca_noncoherent(iq, cfg) for k, cfg in configs.items()}
                accepted = {gate: {str(k): candidate.statistic > scores[k] for k, candidate in candidates.items()}
                            for gate, scores in gates.items()}
                for gate, values in accepted.items():
                    for k, value in values.items():
                        counts[gate][profile][k] += value
                handle.write(json.dumps(dict(profile=profile, record=index, accepted=accepted,
                    candidates={k: asdict(value) for k, value in candidates.items()}, iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest()))+"\n")
                if (index+1) % 250 == 0:
                    handle.flush(); print(f"evaluation {profile}: {index+1}/{evaluation_records}", flush=True)
    signals = []
    with (output/"signals.jsonl").open("w") as handle:
        for profile in PROFILES:
            for cn0 in (37., 41., 45.):
                for doppler in (170., 700.):
                    for replica in range(replicas):
                        rows, starts, seeds, nominal = scene(profile, SIGNAL_DOMAIN, replica, target_cn0=cn0, target_doppler=doppler)
                        iq = render("batch", rows, starts, seeds, NOISE_DB)
                        candidates = {k: acquire_ca_noncoherent(iq, cfg) for k, cfg in configs.items()}
                        result = dict(profile=profile, nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                            replica=replica, iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),
                            gates={gate: {str(k): score_detection(gate_candidate(candidates[k], score), doppler, nominal)
                                          for k, score in scores.items()} for gate, scores in gates.items()})
                        signals.append(result); handle.write(json.dumps(result)+"\n")
            handle.flush(); print(f"signals {profile} complete", flush=True)
    summaries = []
    for profile in PROFILES:
        for gate in gates:
            records = [dict(row, detectors=row["gates"][gate]) for row in signals if row["profile"] == profile]
            summaries.extend(dict(profile=profile, gate=gate, **row) for row in aggregate(records))
    (output/"aggregate.json").write_text(json.dumps(summaries, indent=2)+"\n")
    if digest(frozen_path) != frozen_hash or digest(baseline_path) != protocol["baseline_sha256"] or any(digest(root/path) != value for path, value in hashes.items()):
        raise ValueError("frozen inputs or sources changed")
    null_summary = {gate: {profile: {k: dict(accepted=count, records=evaluation_records,
        fraction=count/evaluation_records, pointwise_wilson_95=wilson_interval(count, evaluation_records))
        for k, count in values.items()} for profile, values in profiles.items()} for gate, profiles in counts.items()}
    report = dict(protocol_sha256=digest(output/"protocol.json"), frozen_thresholds_sha256=frozen_hash,
        null_results=null_summary, signal_records=len(signals),
        limitations=["Coverage applies only to the three declared IID null strata, not arbitrary near-far or RF conditions.",
          "Single fixed maximum gate trades sensitivity for stratum-wide false-alarm control.",
          "Historical white baseline and new stratified gate use different calibration counts/confidence allocations.",
          "Exploratory signal outcomes with coarse peak criteria; no positioning, tracking, rain or RF claim."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-other-prn-nulls"))
    parser.add_argument("--baseline-thresholds", type=Path, default=Path("output/cpu-noncoherent-bit-validation/frozen_thresholds.json"))
    parser.add_argument("--calibration-records", type=int, default=999)
    parser.add_argument("--evaluation-records", type=int, default=999)
    parser.add_argument("--replicas", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.baseline_thresholds, calibration_records=args.calibration_records,
                         evaluation_records=args.evaluation_records, replicas=args.replicas), indent=2))
