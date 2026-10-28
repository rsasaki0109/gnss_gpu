"""Exploratory 1-ms receiver sensitivity with a previously frozen wide gate.

Each condition has independent replicas; conditions share draws within replicas.
Confidence intervals summarize replica events, never correlated block updates.
"""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT, render, receive, json_default
from experiments.run_cpu_receiver_stress import NOISE_DB, make_scene, seed_for, summarize
from experiments.compare_cpu_carrier_grids import WIDE_GRID
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, CaSearchThreshold
from gnss_gpu.code_detection_cpu import CaDetectionThreshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import _sample_index

CN0_LEVELS = (35., 37., 39., 41., 43., 45.)
DOPPLERS = (170., 700.)
REPLICA_OFFSET, NULL_DOMAIN = 100, 0x53454E55
BLOCKS, OUTAGE_MS, CARRIER_CRITERION_HZ = 64, 8, 100.


def wilson_interval(successes, trials):
    n = positive_int("trials", trials)
    k = _sample_index("successes", successes)
    if k > n:
        raise ValueError("successes must not exceed trials")
    z = 1.959963984540054
    p, zz = k/n, z*z
    center = (p+zz/(2*n))/(1+zz/n)
    half = z*math.sqrt(p*(1-p)/n+zz/(4*n*n))/(1+zz/n)
    return [max(0., center-half), min(1., center+half)]


def load_frozen_wide_threshold(path):
    data = json.loads(Path(path).read_text())["thresholds"]["wide"]
    threshold = CaSearchThreshold(CaSearch(**data["search"]), CaDetectionThreshold(**data["calibration"]))
    if threshold.search != CaSearch(FS, WIDE_GRID, prn=1):
        raise ValueError("expected the previously specified five-bin PRN-1 search")
    return threshold


def score_trace(trace, outage, doppler):
    summary = summarize(trace, outage, doppler)
    initial = next((row for row in trace[:outage[0]] if row.status == "acquired"), None)
    error = abs(initial.carrier_hz-doppler) if initial is not None else None
    summary.update(first_acquisition_carrier_error_hz=error,
                   acquired_before_outage_with_carrier_error_le_100hz=error is not None and error <= CARRIER_CRITERION_HZ)
    return summary


def aggregate(rows):
    result = []
    for cn0 in CN0_LEVELS:
        for doppler in DOPPLERS:
            selected = [row for row in rows if row["nominal_primary_cn0_db_hz"] == cn0 and row["primary_doppler_hz"] == doppler]
            if not selected:
                continue
            n = len(selected)
            acquired = sum(row["acquired_before_outage"] for row in selected)
            plausible = sum(row["acquired_before_outage_with_carrier_error_le_100hz"] for row in selected)
            latencies = [row["initial_acquisition_before_outage_ms"] for row in selected if row["initial_acquisition_before_outage_ms"] is not None]
            result.append(dict(nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler, replicas=n,
                acquired_before_outage=acquired, acquisition_fraction=acquired/n,
                acquisition_wilson_95=wilson_interval(acquired, n),
                acquired_with_carrier_criterion=plausible, carrier_criterion_fraction=plausible/n,
                carrier_criterion_wilson_95=wilson_interval(plausible, n),
                mean_fresh_fraction=float(np.mean([row["used_present_blocks"]/row["present_blocks"] for row in selected])),
                conditional_median_initial_acquisition_ms=float(statistics.median(latencies)) if latencies else None))
    return result


def run(output, threshold_report, *, replicas=12):
    replicas = positive_int("replicas", replicas)
    threshold_report, output = Path(threshold_report), Path(output)
    threshold = load_frozen_wide_threshold(threshold_report)
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    null_rows = []
    for index in range(199):
        simulator.noise_seed = seed_for(NULL_DOMAIN, 0, index)
        null_rows.append(simulator.generate_epoch([], COUNT))
    null_iq = np.stack(null_rows)
    null_trace = receive(null_iq, threshold, [index*COUNT for index in range(len(null_rows))])
    output.mkdir(parents=True, exist_ok=True)
    (output/"null_trace.json").write_text(json.dumps([asdict(row) for row in null_trace], default=json_default)+"\n")
    summaries, hashes = [], {}
    with (output/"traces.jsonl").open("w") as handle:
        for cn0 in CN0_LEVELS:
            for doppler in DOPPLERS:
                for index in range(replicas):
                    replica = REPLICA_OFFSET+index
                    rows, starts, seeds, outage = make_scene(cn0, doppler, OUTAGE_MS, replica, BLOCKS)
                    iq = render("batch", rows, starts, seeds, NOISE_DB)
                    trace = receive(iq, threshold, starts)
                    key = f"{cn0:g}/{doppler:g}/{replica}"
                    hashes[key] = hashlib.sha256(iq.tobytes()).hexdigest()
                    identity = dict(nominal_primary_cn0_db_hz=cn0, primary_doppler_hz=doppler, replica=replica)
                    summaries.append(dict(**identity, **score_trace(trace, outage, doppler)))
                    handle.write(json.dumps(dict(**identity, outage_blocks=outage,
                                            trace=[asdict(row) for row in trace]), default=json_default)+"\n")
    grouped = aggregate(summaries)
    (output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    (output/"aggregate.json").write_text(json.dumps(grouped, indent=2)+"\n")
    with (output/"summary.csv").open("w", newline="") as handle:
        fields = [key for key in summaries[0] if key != "states"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"experiments/run_cpu_receiver_stress.py", root/"experiments/run_cpu_batch_receiver.py"]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in
              ("signal_sim_cpu", "acquisition_cpu", "code_delay_cpu", "code_detection_cpu", "tracking_cpu")]
    metadata = dict(replicas=replicas, replica_offset=REPLICA_OFFSET, scenarios=len(summaries), blocks=BLOCKS,
        sampling_freq_hz=FS, noise_floor_db=NOISE_DB, outage_ms=OUTAGE_MS, pre_outage_deadline_ms=BLOCKS//3,
        threshold=asdict(threshold), threshold_report_sha256=hashlib.sha256(threshold_report.read_bytes()).hexdigest(),
        nominal_cn0_db_hz=CN0_LEVELS, doppler_hz=DOPPLERS, carrier_criterion_hz=CARRIER_CRITERION_HZ,
        iq_sha256=hashes, null_domain=NULL_DOMAIN, numpy=np.__version__,
        held_out_null=dict(blocks=len(null_trace), accepted_blocks=sum(row.detection_accepted for row in null_trace),
                           used_blocks=sum(row.measurement_used for row in null_trace),
                           iq_sha256=hashlib.sha256(null_iq.tobytes()).hexdigest()),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        limitations=["Exploratory fixed-gate 12-replica matrix, not a precise sensitivity threshold or RF validation.",
          "Wilson intervals apply to independent replica events within each condition, not correlated block updates.",
          "Conditions share draws within replicas; intervals are pointwise, not simultaneous across conditions.",
          "The 100-Hz criterion is a declared carrier-only evaluation rule, not code accuracy or integrity.",
          "Acquisition is censored at 21 ms; conditional median latency excludes failures.",
          "Nominal C/N0 and point-sampled waveform/aperture-model limitations persist; no rain or PVT claim."])
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata, grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cpu-receiver-sensitivity"))
    parser.add_argument("--threshold-report", type=Path, default=Path("output/cpu-carrier-grid-comparison/report.json"))
    parser.add_argument("--replicas", type=int, default=12)
    args = parser.parse_args()
    metadata, _ = run(args.output, args.threshold_report, replicas=args.replicas)
    print(json.dumps({key: metadata[key] for key in ("scenarios", "held_out_null")}, indent=2))
