"""Compare global-descriptor coefficient caching and complete signal/urban calls.

Use two persistent production workers with sequential GPU work, alternating
20-call batches by stage/round. Hash every timed IQ/metadata outside wall timing.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from bench_signal_kernel_compare import (
    ROOT, SOURCE_PATHS, _Profile as _BaseProfile, _Worker as _BaseWorker,
    _mixed_channels, _sha, _tree_hash, _validate as _base_validate,
)
from bench_raytrace import _make_building_mesh, _make_satellites
from bench_signal_sim_compare import _channels, _input_digest


def _native_cases():
    for samples in (2600, 260000, 2600000):
        for count in (33, 65, 128, 256, 257):
            yield f"native/GPS/samples{samples}/channels{count}", (
                2.6e6, 0., _channels(count), samples, -20., 42)
    for samples in (260000, 2600000):
        for count in (0, 32):
            yield f"native/control/samples{samples}/channels{count}", (
                2.6e6, 0., _channels(count), samples, -20., 42)
    yield "native/control/samples260000/channels12", (2.6e6, 0., _channels(12), 260000, -20., 42)
    for count in (65, 256):
        yield f"native/mixed_nav/samples260000/channels{count}", (
            2.6e6, 0., _mixed_channels(count, True), 260000, -20., 42)
        yield f"native/no_noise/samples260000/channels{count}", (
            2.6e6, 0., _channels(count), 260000, -10000., 42)
    for system in (1, 2, 3):
        channels = _channels(65)
        for channel in channels:
            channel.update(system=system, prn=(channel["prn"] - 1) % (24, 36, 37)[system - 1] + 1)
        yield f"native/system{system}/samples260000/channels65", (
            2.6e6, 0., channels, 260000, -20., 42)


def _stages():
    yield from (name for name, _ in _native_cases())
    for samples in (260000, 2600000):
        for count in (33, 65, 128, 256, 257):
            yield f"public/samples{samples}/channels{count}"
    for count in (33, 65, 257):
        yield f"public/samples2600/channels{count}"
    for samples in (2600, 260000):
        for count in (65, 257):
            yield f"cold/samples{samples}/channels{count}"
    for count in (1200, 12000):
        for samples in (2600, 260000):
            yield f"urban/triangles{count}/rays32/samples{samples}"


def _extra_cases():
    for count in (33, 64, 65, 127, 128, 255, 256, 257, 513):
        for samples in (1, 257, 4097):
            for nav in (False, True):
                yield f"global_boundary/channels{count}/samples{samples}/nav{nav}", (
                    4.092e6, -1.1e6, _mixed_channels(count, nav), samples, -40., 2**64 - 1)
    for system in range(5):
        for phase in (-2147483648., 2147483647.):
            channels = _channels(256)
            for channel in channels:
                channel.update(system=system, prn=1, code_phase=phase)
            yield f"global_integer_edge/system{system}/phase{phase}", (
                2.6e6, 1.25e6, channels, 513, -20., 37)
    for count in (33, 256, 257):
        for intermediate in (-1e16, 1e16):
            channels = _mixed_channels(count)
            for index, channel in enumerate(channels):
                channel["doppler_hz"] = -intermediate + index * .125
            yield f"global_cancellation/channels{count}/IF{intermediate}", (
                2.6e6, intermediate, channels, 257, -10000., 42)
    for count in (256, 257):
        channels = _mixed_channels(count)
        channels += [dict(prn=1, system=99, code_phase=0., carrier_phase=0.,
                          doppler_hz=0., amplitude=1., nav_bit=1)] * 17
        yield f"filtered_global_boundary/valid{count}", (2.6e6, 0., channels, 257, -20., 42)


def _validate(native, workspace):
    records = _base_validate(native, workspace)
    retained = []
    for name, arguments in _extra_cases():
        input_hash = _input_digest(arguments)
        cold, warm = native.generate_signal(*arguments), workspace.generate_signal(*arguments)
        np.testing.assert_array_equal(warm.view(np.uint32), cold.view(np.uint32), err_msg=name)
        assert _input_digest(arguments) == input_hash
        records[name] = dict(input_sha256=input_hash, output_sha256=_tree_hash(warm),
                             shape=list(warm.shape), dtype=str(warm.dtype))
        retained.append((warm, _tree_hash(warm)))
    for index, (samples, count) in enumerate(((1, 256), (8193, 257), (257, 65),
                                             (1, 32), (4097, 513), (2, 255), (513, 0), (256, 33))):
        arguments = (4.092e6, 1.25e6, _mixed_channels(count, True), samples, -20., 37)
        warm = workspace.generate_signal(*arguments)
        np.testing.assert_array_equal(warm.view(np.uint32), native.generate_signal(*arguments).view(np.uint32))
        records[f"global_reuse/{index}"] = dict(input_sha256=_input_digest(arguments),
            output_sha256=_tree_hash(warm), shape=list(warm.shape), dtype=str(warm.dtype))
        for output, digest in retained:
            assert _tree_hash(output) == digest
    return records


class _Profile(_BaseProfile):
    def __init__(self, extension):
        super().__init__(extension)
        self.native_cases.update(dict(_native_cases()))

    def stage(self, name):
        if not name.startswith("urban/"):
            return super().stage(name)
        from gnss_gpu.bvh import BVHAccelerator
        from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

        _, count, rays, samples = name.split("/")
        count, rays, samples = int(count[9:]), int(rays[4:]), int(samples[7:])
        key = count, rays
        if key not in self.urban:
            rx = np.array([6378137., 0., 0.])
            sat = _make_satellites(rays)[:, [2, 0, 1]] + rx
            mesh = _make_building_mesh(count // 12)[:, :, [2, 0, 1]] + rx
            sim = UrbanSignalSimulator(BVHAccelerator(mesh))
            sim.sim.noise_seed = 37
            self.urban[key] = sim, rx, sat, mesh
        sim, rx, sat, mesh = self.urban[key]
        return lambda: sim.compute_epoch(rx, sat, n_samples=samples), _input_digest((rx, sat, mesh))

    def measure(self, name, repeats):
        call, input_hash = self.stage(name)
        reference = call()
        digest = _tree_hash(reference)
        channels = len(reference["channels"]) if isinstance(reference, dict) else None
        for _ in range(3):
            assert _tree_hash(call()) == digest
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = call()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            assert _tree_hash(output) == digest, name
        return dict(median_ms=float(np.median(samples)), samples_ms=samples,
                    input_sha256=input_hash, output_sha256=digest,
                    generated_channels=channels, all_timed_outputs_identical=True)


def _serve(extension):
    profile = _Profile(extension)
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "close":
                break
            try:
                result = (_validate(profile.native, profile.workspace) if request["command"] == "validate"
                          else profile.measure(request["stage"], request["repeats"]))
                print(json.dumps(dict(result=result)), flush=True)
            except Exception:
                print(json.dumps(dict(error=traceback.format_exc())), flush=True)
    finally:
        profile.close()


class _Worker(_BaseWorker):
    def __init__(self, extension):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--extension", str(extension.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=ROOT)
        assert self._read()["ready"]


def _compare(args):
    reference = json.loads((ROOT / "benchmarks/signal_frequency_gpu_20261012.json").read_text(encoding="utf-8"))
    hashes = dict(baseline={path: _sha(args.baseline_dir / Path(path).name) for path in SOURCE_PATHS},
                  candidate={path: _sha(ROOT / path) for path in SOURCE_PATHS})
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert hashes["baseline"] == reference["source_sha256"]["candidate"], "Wrong preceding sources"
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"], "Wrong preceding binary"
    assert all(hashes["baseline"][path] == hashes["candidate"][path] for path in SOURCE_PATHS[1:])
    workers, validation, measurements = {}, {}, {}
    try:
        for label, path in dict(baseline=args.baseline, candidate=args.candidate).items():
            workers[label] = _Worker(path)
        for label, worker in workers.items():
            validation[label] = worker.request(command="validate")
        assert validation["baseline"] == validation["candidate"]
        print(f"{len(validation['baseline'])} IQ cases have identical bit hashes", flush=True)
        if not args.validation_only:
            names = [name for name in _stages() if not args.stages or any(part in name for part in args.stages)]
            for name in names:
                runs = {label: [] for label in workers}
                for index in range(args.rounds):
                    for label in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")):
                        runs[label].append(workers[label].request(command="measure", stage=name, repeats=args.repeats))
                outputs = {run["output_sha256"] for values in runs.values() for run in values}
                inputs = {run["input_sha256"] for values in runs.values() for run in values}
                channels = {run["generated_channels"] for values in runs.values() for run in values}
                assert len(outputs) == len(inputs) == len(channels) == 1, name
                medians = {label: [run["median_ms"] for run in values] for label, values in runs.items()}
                old, new = [float(np.median(medians[label])) for label in workers]
                measurements[name] = dict(baseline_ms=old, candidate_ms=new, speedup=old / new,
                    round_medians_ms=medians, runs=runs, output_sha256=outputs.pop(), input_sha256=inputs.pop(),
                    generated_channels=channels.pop(), all_timed_outputs_identical=True)
                print(f"{name}: {old:.4f} -> {new:.4f} ms ({old / new:.3f}x) "
                      f"channels={measurements[name]['generated_channels']}", flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    dependencies = ("benchmarks/bench_signal_global_frequency_compare.py", "benchmarks/bench_signal_kernel_compare.py",
        "benchmarks/bench_signal_sim_compare.py", "benchmarks/bench_bvh_combined_compare.py",
        "benchmarks/bench_bvh_resident_compare.py", "benchmarks/bench_bvh_multipath_compare.py", "benchmarks/bench_raytrace.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
        baseline="preceding inline coefficient cache; original global descriptor kernel above 32 channels",
        method="persistent actual production workers; sequential GPU work; alternating per-stage build order by round; nonexclusive host",
        scope="native includes conversion/transfers/kernel/synchronization/host IQ; public adds validation; "
              "urban includes LOS/reflection/channel metadata/actual CUDA IQ; cold includes workspace lifetime; "
              "CPU mesh/BVH construction, CUDA/code-table initialization, IPC and hashes excluded",
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_stage=3, reference_calls_per_stage=1,
        source_sha256=hashes, extension_sha256=extensions,
        bvh_extension_sha256={str(path.relative_to(ROOT)): _sha(path) for path in (ROOT / "python/gnss_gpu").glob("_bvh*.pyd")},
        benchmark_sha256={path: _sha(ROOT / path) for path in dependencies},
        validation=dict(cases=len(validation["baseline"]), all_iq_bits_identical=True,
            baseline_by_case=validation["baseline"], candidate_by_case=validation["candidate"]), measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extension:
        _serve(args.extension)
        return
    if args.rounds < 1 or args.repeats < 1 or not (args.baseline and args.candidate and args.baseline_dir and args.output):
        parser.error("positive rounds/repeats and baseline/candidate/baseline-dir/output required")
    report = _compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
