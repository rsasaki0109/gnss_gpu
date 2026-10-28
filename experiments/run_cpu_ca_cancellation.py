"""Calibrate and evaluate raw/cancelled PRN-1 detectors on matched fresh IQ."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import FS, render
from experiments.run_cpu_receiver_stress import NOISE_DB
from experiments.run_cpu_other_prn_nulls import scene, PROFILES
from experiments.run_cpu_noncoherent_sensitivity import aggregate, score_detection
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig, cancel_other_ca
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, acquire_ca_noncoherent, CaNoncoherentDetection
from gnss_gpu.stratified_tolerance_cpu import calibrate_ca_stratified_tolerance
from gnss_gpu.noncoherent_tolerance_cpu import tolerance_order
from gnss_gpu.input_validation import positive_int

CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN = 0x43414341, 0x43414556, 0x43415349


def cancellation_metadata(result):
    return dict(status=result.status, removed_energy_fraction=result.removed_energy_fraction,
                gram_condition=result.gram_condition, components=[asdict(value) for value in result.components])


def strict_detection(candidate, threshold):
    score = candidate.statistic if candidate is not None else 0.
    return CaNoncoherentDetection(candidate is not None and score > threshold, score, threshold, candidate)


def run(output, *, calibration_records=599, evaluation_records=499, replicas=24):
    calibration_records = positive_int("calibration_records", calibration_records)
    evaluation_records = positive_int("evaluation_records", evaluation_records)
    replicas = positive_int("replicas", replicas)
    policies = ("raw_1", "raw_4", "cancelled_1", "cancelled_4")
    tolerance_order(calibration_records, failure_probability=.05/(len(policies)*len(PROFILES)))
    configs = {k: CaNoncoherentSearch(CaSearch(FS, WIDE_GRID), k, 20//k) for k in (1, 4)}
    cancellation = CaCancellationConfig(FS)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__)]
    paths += [root/"experiments"/(name+".py") for name in ("run_cpu_batch_receiver", "run_cpu_receiver_stress", "run_cpu_other_prn_nulls", "run_cpu_noncoherent_sensitivity", "run_cpu_receiver_sensitivity")]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in ("ca_cancellation_cpu", "stratified_tolerance_cpu", "noncoherent_tolerance_cpu", "noncoherent_acquisition_cpu", "signal_sim_cpu", "acquisition_cpu", "code_delay_cpu", "code_detection_cpu", "validation/tracking_residuals")]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    protocol = dict(calibration_records_per_profile=calibration_records, evaluation_records_per_profile=evaluation_records,
        signal_replicas=replicas, profiles=PROFILES, policies=policies, horizon_ms=20,
        cancellation=asdict(cancellation), detector_configs={k: asdict(value) for k, value in configs.items()},
        noise_domains=dict(calibration=CALIBRATION_DOMAIN, evaluation=EVALUATION_DOMAIN, signal=SIGNAL_DOMAIN),
        max_false_alarm_probability=.01, joint_calibration_failure_budget=.05,
        calibration_family_size=len(PROFILES)*len(policies), signal_cn0_db_hz=[37., 41., 45.], doppler_hz=[170., 700.],
        source_sha256=hashes, evaluation_policy="Freeze complete raw/cancelled pipeline thresholds before evaluation; no retuning.",
        null_model="PRN 1 absent in three finite IID strata; other satellite bits constant and phases/code delays randomized.",
        inference_inputs="IQ and fixed code/search catalog only; no true amplitude, carrier, delay or presence label.")
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    def analyse(iq):
        result = cancel_other_ca(iq, cancellation)
        candidates = {}
        for k, cfg in configs.items():
            candidates[f"raw_{k}"] = acquire_ca_noncoherent(iq, cfg)
            candidates[f"cancelled_{k}"] = acquire_ca_noncoherent(result.iq, cfg)
        audit = cancellation_metadata(result)
        audit["raw_iq_sha256"] = hashlib.sha256(iq.tobytes()).hexdigest()
        audit["residual_iq_sha256"] = hashlib.sha256(result.iq.tobytes()).hexdigest()
        return candidates, audit
    calibration = {policy: {profile: [] for profile in PROFILES} for policy in policies}
    with (output/"calibration.jsonl").open("w") as handle:
        for profile in PROFILES:
            for index in range(calibration_records):
                rows, starts, seeds, _ = scene(profile, CALIBRATION_DOMAIN, index)
                candidates, audit = analyse(render("batch", rows, starts, seeds, NOISE_DB))
                for policy, candidate in candidates.items():
                    calibration[policy][profile].append(candidate.statistic)
                handle.write(json.dumps(dict(profile=profile, record=index,
                    candidates={p: asdict(v) for p, v in candidates.items()}, cancellation=audit))+"\n")
                if (index+1) % 100 == 0:
                    handle.flush(); print(f"calibration {profile}: {index+1}/{calibration_records}", flush=True)
    thresholds = {policy: calibrate_ca_stratified_tolerance(values, configs[int(policy[-1])], methods_in_family=len(policies))
                  for policy, values in calibration.items()}
    frozen_path = output/"frozen_thresholds.json"
    frozen_path.write_text(json.dumps({p: asdict(v) for p, v in thresholds.items()}, indent=2)+"\n")
    frozen_hash = digest(frozen_path)
    counts = {profile: {policy: 0 for policy in policies} for profile in PROFILES}
    with (output/"evaluation.jsonl").open("w") as handle:
        for profile in PROFILES:
            for index in range(evaluation_records):
                rows, starts, seeds, _ = scene(profile, EVALUATION_DOMAIN, index)
                candidates, audit = analyse(render("batch", rows, starts, seeds, NOISE_DB))
                accepted = {policy: strict_detection(candidate, thresholds[policy].score).accepted for policy, candidate in candidates.items()}
                for policy, flag in accepted.items():
                    counts[profile][policy] += flag
                handle.write(json.dumps(dict(profile=profile, record=index, accepted=accepted,
                    candidates={p: asdict(v) for p, v in candidates.items()}, cancellation=audit))+"\n")
                if (index+1) % 100 == 0:
                    handle.flush(); print(f"evaluation {profile}: {index+1}/{evaluation_records}", flush=True)
    signals = []
    with (output/"signals.jsonl").open("w") as handle:
        for profile in PROFILES:
            for cn0 in (37., 41., 45.):
                for doppler in (170., 700.):
                    for replica in range(replicas):
                        rows, starts, seeds, nominal = scene(profile, SIGNAL_DOMAIN, replica, target_cn0=cn0, target_doppler=doppler)
                        candidates, audit = analyse(render("batch", rows, starts, seeds, NOISE_DB))
                        result = dict(profile=profile, nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                            replica=replica, cancellation=audit,
                            policies={policy: score_detection(strict_detection(candidate, thresholds[policy].score), doppler, nominal)
                                      for policy, candidate in candidates.items()})
                        signals.append(result); handle.write(json.dumps(result)+"\n")
            handle.flush(); print(f"signals {profile} complete", flush=True)
    grouped = []
    for profile in PROFILES:
        for mode in ("raw", "cancelled"):
            rows = [dict(row, detectors={str(k): row["policies"][f"{mode}_{k}"] for k in (1, 4)})
                    for row in signals if row["profile"] == profile]
            grouped.extend(dict(profile=profile, mode=mode, **row) for row in aggregate(rows))
    (output/"aggregate.json").write_text(json.dumps(grouped, indent=2)+"\n")
    if digest(frozen_path) != frozen_hash or any(digest(root/path) != value for path, value in hashes.items()):
        raise ValueError("frozen threshold or source changed during experiment")
    report = dict(protocol_sha256=digest(output/"protocol.json"), frozen_thresholds_sha256=frozen_hash,
        null_results={profile: {policy: dict(accepted=count, records=evaluation_records, fraction=count/evaluation_records,
            pointwise_wilson_95=wilson_interval(count, evaluation_records)) for policy, count in values.items()}
            for profile, values in counts.items()}, signal_records=len(signals),
        limitations=["One-pass blind nuisance projection can remove target energy and noise; reduced energy is not accuracy.",
          "Full residual pipeline calibrated separately; old raw thresholds are not reused as calibrated residual gates.",
          "Finite declared IID null strata only, not arbitrary interference, dynamic satellites or RF environments.",
          "Nuisance bits constant in this experiment; within-period nuisance transitions/code drift can break the fit.",
          "Coarse target criteria, not sustained tracking, ranging, PVT, rain or algorithmic novelty."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-ca-cancellation"))
    parser.add_argument("--calibration-records", type=int, default=599)
    parser.add_argument("--evaluation-records", type=int, default=499)
    parser.add_argument("--replicas", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run(args.output, calibration_records=args.calibration_records,
                         evaluation_records=args.evaluation_records, replicas=args.replicas), indent=2))
