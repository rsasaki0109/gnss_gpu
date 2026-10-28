"""Fixed three-bin versus independently calibrated five-bin receiver search.

Reuses the stress matrix's saved narrow traces and checksum-matched IQ. The
wide grid is fixed for every scenario, not selected using truth or diagnostics.
"""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from experiments.audit_cpu_carrier_support import decode_complex
from experiments.run_cpu_batch_receiver import FS, COUNT, render, receive, json_default
from experiments.run_cpu_receiver_stress import (
    CALIBRATION, EVALUATION, NOISE_DB, make_scene, seed_for, summarize,
)
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, CaSearchThreshold, acquire_ca_signal, calibrate_ca_search
from gnss_gpu.carrier_support_cpu import assess_ca_carrier_support
from gnss_gpu.code_detection_cpu import CaDetectionThreshold
from gnss_gpu.input_validation import positive_int

WIDE_GRID = (-1000., -500., 0., 500., 1000.)


def calibrate_grids(noise, searches):
    scores = [[acquire_ca_signal(row, search).statistic for row in noise] for search in searches]
    return [(calibrate_ca_search(values, search), values) for search, values in zip(searches, scores)]


def paired_carrier_errors(before, after, outage, truth):
    begin, end = outage
    if len(before) != len(after):
        raise ValueError("trace lengths must match")
    indices = [i for i, (a, b) in enumerate(zip(before, after))
               if not begin <= i < end and a["measurement_used"] and b["measurement_used"]]
    return dict(common_used_indices=indices, common_used_blocks=len(indices),
                common_carrier_mae_before_hz=float(np.mean([abs(before[i]["carrier_hz"]-truth) for i in indices])) if indices else None,
                common_carrier_mae_after_hz=float(np.mean([abs(after[i]["carrier_hz"]-truth) for i in indices])) if indices else None)


def run(recording_dir, output, *, timing_repeats=7):
    timing_repeats = positive_int("timing_repeats", timing_repeats)
    recording_dir, output = Path(recording_dir), Path(output)
    metadata = json.loads((recording_dir/"metadata.json").read_text())
    if metadata["sampling_freq_hz"] != FS or metadata["noise_floor_db"] != NOISE_DB:
        raise ValueError("recording sampling/noise settings do not match regeneration")
    narrow = CaSearch(**metadata["threshold"]["search"])
    wide = CaSearch(FS, WIDE_GRID, prn=narrow.prn)
    if not set(narrow.carrier_grid_hz) < set(wide.carrier_grid_hz):
        raise ValueError("wide grid must strictly contain original bins")
    simulator = CpuSignalSimulator(FS, noise_floor_db=NOISE_DB)
    def noise_blocks(domain, count):
        rows = []
        for index in range(count):
            simulator.noise_seed = seed_for(domain, 0, index)
            rows.append(simulator.generate_epoch([], COUNT))
        return np.stack(rows)
    calibration = noise_blocks(CALIBRATION, len(metadata["calibration_scores"]))
    (narrow_threshold, narrow_scores), (wide_threshold, wide_scores) = calibrate_grids(calibration, (narrow, wide))
    np.testing.assert_array_equal(narrow_scores, metadata["calibration_scores"])
    assert narrow_threshold == CaSearchThreshold(narrow, CaDetectionThreshold(**metadata["threshold"]["calibration"]))
    assert np.all(np.asarray(wide_scores) >= np.asarray(narrow_scores))
    null = noise_blocks(EVALUATION, metadata["held_out_null"]["blocks"])
    null_traces = {name: receive(null, threshold, [i*COUNT for i in range(len(null))])
                   for name, threshold in (("narrow", narrow_threshold), ("wide", wide_threshold))}
    for name in ("accepted_blocks", "used_blocks"):
        field = "detection_accepted" if name == "accepted_blocks" else "measurement_used"
        assert sum(getattr(row, field) for row in null_traces["narrow"]) == metadata["held_out_null"][name]
    output.mkdir(parents=True, exist_ok=True)
    summaries, representative, seen = [], None, set()
    with (output/"wide_traces.jsonl").open("w") as handle:
        for line in (recording_dir/"traces.jsonl").read_text().splitlines():
            old = json.loads(line, object_hook=decode_complex)
            key = str(old["condition"])
            if key in seen:
                raise ValueError("duplicate scenario")
            seen.add(key)
            cn0, doppler = old["nominal_primary_cn0_db_hz"], old["primary_doppler_hz"]
            rows, starts, seeds, outage = make_scene(cn0, doppler, old["outage_ms"], old["replica"], metadata["blocks"])
            assert list(outage) == old["outage_blocks"]
            iq = render("batch", rows, starts, seeds, NOISE_DB)
            assert hashlib.sha256(iq.tobytes()).hexdigest() == metadata["iq_sha256"][key]
            trace = receive(iq, wide_threshold, starts)
            identity = {key: old[key] for key in ("condition", "nominal_primary_cn0_db_hz", "primary_doppler_hz", "outage_ms", "replica")}
            paired = paired_carrier_errors(old["trace"], [asdict(row) for row in trace], outage, doppler)
            summaries.append(dict(**identity, **summarize(trace, outage, doppler), **paired,
                boundary_used_blocks=sum(row.measurement_used and assess_ca_carrier_support(row.refined_candidate, wide).status == "boundary" for row in trace)))
            handle.write(json.dumps(dict(**identity, trace=[asdict(row) for row in trace]), default=json_default)+"\n")
            if (cn0, doppler, old["outage_ms"], old["replica"]) == (55., 700., 8, 0):
                representative = iq, starts, old["trace"], [asdict(row) for row in trace]
    if seen != set(metadata["iq_sha256"]):
        raise ValueError("missing scenarios")
    if representative is None:
        raise ValueError("missing fixed timing scenario")
    # Receiver-only timings: warm, alternate order, compare results outside timers.
    iq, starts, reference_narrow, reference_wide = representative
    settings = {"narrow": (narrow_threshold, reference_narrow), "wide": (wide_threshold, reference_wide)}
    for threshold, expected in settings.values():
        assert [asdict(row) for row in receive(iq, threshold, starts)] == expected
    timings = []
    for repeat in range(timing_repeats):
        for name in (("narrow", "wide") if repeat % 2 == 0 else ("wide", "narrow")):
            threshold, expected = settings[name]
            begin = time.perf_counter()
            trace = receive(iq, threshold, starts)
            seconds = time.perf_counter()-begin
            assert [asdict(row) for row in trace] == expected
            timings.append(dict(repeat=repeat, grid=name, receiver_s=seconds))
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"experiments/run_cpu_receiver_stress.py", root/"experiments/run_cpu_batch_receiver.py"]
    paths += [root/"python/gnss_gpu"/(name+".py") for name in ("signal_sim_cpu", "acquisition_cpu", "code_delay_cpu", "code_detection_cpu", "tracking_cpu", "carrier_support_cpu")]
    report = dict(scenarios=len(summaries), regenerated_iq_matches_recording=True,
        narrow_calibration_and_null_counts_match=True, calibration_trials=len(calibration),
        thresholds={"narrow": asdict(narrow_threshold), "wide": asdict(wide_threshold)},
        calibration_scores={"narrow": narrow_scores, "wide": wide_scores},
        held_out_null={name: dict(blocks=len(null), accepted_blocks=sum(row.detection_accepted for row in trace),
                                  used_blocks=sum(row.measurement_used for row in trace)) for name, trace in null_traces.items()},
        timings=timings, median_receiver_s={name: statistics.median(row["receiver_s"] for row in timings if row["grid"] == name) for name in settings},
        timing_scenario=dict(cn0=55., doppler=700., outage_ms=8, replica=0, blocks=len(iq)),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        recording_sha256={name: hashlib.sha256((recording_dir/name).read_bytes()).hexdigest() for name in ("metadata.json", "traces.jsonl")},
        limitations=["Fixed five-bin grid calibrated independently; no adaptive grid selection or truth aiding.",
          "Same small paired synthetic matrix, not independent population or rare-event validation.",
          "Conditional and common-used errors exclude unavailable measurements; availability is reported separately.",
          "Receiver-only timing on one frozen 64-ms scenario, not a universal speed or real-time claim.",
          "Point-sampled synthesis/aperture fitter mismatch persists; no PVT, rain, RF or integrity claim."])
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    with (output/"summary.csv").open("w", newline="") as handle:
        fields = [key for key in summaries[0] if key not in ("states", "common_used_indices")]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    return report, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, default=Path("output/cpu-receiver-stress"))
    parser.add_argument("--output", type=Path, default=Path("output/cpu-carrier-grid-comparison"))
    parser.add_argument("--timing-repeats", type=int, default=7)
    args = parser.parse_args()
    report, _ = run(args.recording_dir, args.output, timing_repeats=args.timing_repeats)
    print(json.dumps({key: report[key] for key in ("scenarios", "held_out_null", "median_receiver_s")}, indent=2))
