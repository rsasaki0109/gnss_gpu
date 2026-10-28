"""Compare 1-ms and 4-ms integer-delay detectors over matched 20-ms decisions.

This evaluates capture detection, not the existing fractional-delay tracker.
Both detectors receive their own complete-record rank calibration.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import FS, COUNT, render
from experiments.run_cpu_receiver_stress import NOISE_DB, make_scene, seed_for
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.noncoherent_acquisition_cpu import (
    CaNoncoherentSearch, acquire_ca_noncoherent, calibrate_ca_noncoherent, detect_ca_noncoherent,
)
import numpy as np

CALIBRATION_DOMAIN, EVALUATION_DOMAIN = 0x4E434341, 0x4E434556
HORIZON_MS, CALIBRATION_RECORDS, NULL_RECORDS = 20, 199, 199


def score_detection(detection, doppler, nominal_delay):
    candidate = detection.candidate
    carrier_error = abs(candidate.carrier_hz-doppler) if candidate is not None else None
    lag_error = abs((candidate.delay_samples-nominal_delay+COUNT/2) % COUNT-COUNT/2) if candidate is not None else None
    return dict(accepted=detection.accepted, statistic=detection.statistic,
        candidate=asdict(candidate) if candidate is not None else None,
        carrier_error_hz=carrier_error, initial_phase_lag_error_samples=lag_error,
        passes_declared_criterion=bool(detection.accepted and carrier_error <= 250. and lag_error <= 2.))


def aggregate(rows):
    result = []
    conditions = sorted({(row["nominal_primary_cn0_db_hz"], row["primary_doppler_hz"]) for row in rows})
    for cn0, doppler in conditions:
        selected = [row for row in rows if (row["nominal_primary_cn0_db_hz"], row["primary_doppler_hz"]) == (cn0, doppler)]
        for k in (1, 4):
            outcomes = [row["detectors"][str(k)] for row in selected]
            n = len(outcomes)
            accepted, criterion = sum(x["accepted"] for x in outcomes), sum(x["passes_declared_criterion"] for x in outcomes)
            result.append(dict(nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                integration_ms=k, replicas=n, accepted=accepted, criterion=criterion,
                accepted_fraction=accepted/n, criterion_fraction=criterion/n,
                accepted_wilson_95=wilson_interval(accepted, n), criterion_wilson_95=wilson_interval(criterion, n)))
    return result


def run(recording_dir, output):
    recording_dir, output = Path(recording_dir), Path(output)
    original = json.loads((recording_dir/"metadata.json").read_text())
    if original["sampling_freq_hz"] != FS or original["noise_floor_db"] != NOISE_DB:
        raise ValueError("recording sampling/noise settings do not match regeneration")
    search = CaSearch(FS, WIDE_GRID)
    configs = {k: CaNoncoherentSearch(search, k, HORIZON_MS//k) for k in (1, 4)}
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    def noise_record(domain, index):
        rows = []
        for block in range(HORIZON_MS):
            simulator.noise_seed = seed_for(domain, index, block)
            rows.append(simulator.generate_epoch([], COUNT))
        return np.stack(rows)
    calibration = {k: [] for k in configs}
    for index in range(CALIBRATION_RECORDS):
        iq = noise_record(CALIBRATION_DOMAIN, index)
        for k, config in configs.items():
            calibration[k].append(acquire_ca_noncoherent(iq, config).statistic)
    thresholds = {k: calibrate_ca_noncoherent(values, configs[k]) for k, values in calibration.items()}
    output.mkdir(parents=True, exist_ok=True)
    null_counts = {k: 0 for k in configs}
    with (output/"null_decisions.jsonl").open("w") as handle:
        for index in range(NULL_RECORDS):
            iq = noise_record(EVALUATION_DOMAIN, index)
            detections = {k: detect_ca_noncoherent(iq, threshold) for k, threshold in thresholds.items()}
            for k, detection in detections.items():
                null_counts[k] += detection.accepted
            handle.write(json.dumps(dict(record=index, iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),
                                        detectors={k: asdict(detection) for k, detection in detections.items()}))+"\n")
    results, hashes = [], {}
    with (output/"decisions.jsonl").open("w") as handle:
        for cn0 in original["nominal_cn0_db_hz"]:
            for doppler in original["doppler_hz"]:
                for replica in range(original["replica_offset"], original["replica_offset"]+original["replicas"]):
                    rows, starts, seeds, outage = make_scene(cn0, doppler, original["outage_ms"], replica, original["blocks"])
                    if outage[0] < HORIZON_MS:
                        raise ValueError("decision horizon must precede the outage")
                    iq = render("batch", rows, starts, seeds, NOISE_DB)
                    key = f"{cn0:g}/{doppler:g}/{replica}"
                    assert hashlib.sha256(iq.tobytes()).hexdigest() == original["iq_sha256"][key]
                    decision_iq = iq[:HORIZON_MS]
                    hashes[key] = hashlib.sha256(decision_iq.tobytes()).hexdigest()
                    # Fixed scene's initial phase implies 37 samples at 2 samples/chip.
                    # Small code drift over this horizon is not treated as absolute truth.
                    nominal_delay = -rows[0][0]["code_phase"]*2
                    detectors = {str(k): score_detection(detect_ca_noncoherent(decision_iq, threshold), doppler, nominal_delay)
                                 for k, threshold in thresholds.items()}
                    result = dict(nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler,
                                  replica=replica, detectors=detectors)
                    results.append(result)
                    handle.write(json.dumps(result)+"\n")
    grouped = aggregate(results)
    (output/"aggregate.json").write_text(json.dumps(grouped, indent=2)+"\n")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"experiments/run_cpu_receiver_stress.py", root/"experiments/run_cpu_batch_receiver.py"]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in
              ("noncoherent_acquisition_cpu", "signal_sim_cpu", "code_delay_cpu", "code_detection_cpu", "acquisition_cpu")]
    report = dict(scenarios=len(results), horizon_ms=HORIZON_MS, replicas=original["replicas"],
        thresholds={k: asdict(value) for k, value in thresholds.items()}, calibration_scores=calibration,
        calibration_records=CALIBRATION_RECORDS, null_records=NULL_RECORDS, null_accepted=null_counts,
        noise_domains=dict(calibration=CALIBRATION_DOMAIN, evaluation=EVALUATION_DOMAIN),
        decision_iq_sha256=hashes, original_full_iq_hashes_match=True,
        original_metadata_sha256=hashlib.sha256((recording_dir/"metadata.json").read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        criterion=dict(max_carrier_error_hz=250., max_initial_phase_lag_error_samples=2.),
        limitations=["Both methods are integer-delay detectors; neither is the earlier fractional-delay tracker.",
          "Matched 20-ms decision-level calibration, not per-window or track-level calibration.",
          "12 paired replicas per condition; pointwise Wilson intervals, not simultaneous population validation.",
          "No tracking, carrier refinement, nav-bit synchronization, rain, PVT or RF validation.",
          "Criterion is an offline coarse-carrier/nominal-phase check, not an accuracy or integrity bound.",
          "Constant-delay windows neglect small code drift; within-period bit transitions and dynamics can reduce gain.",
          "Whole-period polarity robustness does not imply robustness to arbitrary navigation-bit timing."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report, grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, default=Path("output/cpu-receiver-sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("output/cpu-noncoherent-sensitivity"))
    args = parser.parse_args()
    report, _ = run(args.recording_dir, args.output)
    print(json.dumps({key: report[key] for key in ("scenarios", "null_records", "null_accepted")}, indent=2))
