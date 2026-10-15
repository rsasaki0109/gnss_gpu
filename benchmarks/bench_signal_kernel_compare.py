"""Exact IQ validation and alternating actual-build signal/urban timing.

Persistent workers isolate pybind's global workspace type registration. Only
one worker runs GPU work at a time. Each stage alternates 20-call batches by
round, with synchronized wall timing and output hashes computed afterwards.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from bench_bvh_combined_compare import _tree_hash
from bench_raytrace import _make_building_mesh, _make_satellites
from bench_signal_sim_compare import (
    _channel, _channels, _input_digest, _load_extension, _validation_cases,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/signal_sim/signal_sim.cu", "include/gnss_gpu/signal_sim.h",
    "python/gnss_gpu/_signal_sim_bindings.cpp", "python/gnss_gpu/signal_sim.py",
    "python/gnss_gpu/urban_signal_sim.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases():
    yield from _validation_cases()
    for system, length in enumerate((1023, 511, 4092, 2046, 1023)):
        for phase in (-2147483648., -2147483647., -2. * length, -length - 1.,
                      -1. - 1e-12, -0., length - 1e-12, 2147480000., 2147483647.):
            for count in (1, 33):
                channels = [_channel(system=system, code_phase=phase, doppler_hz=-999.5,
                                     carrier_phase=.13, amplitude=.7, nav_bit=-1)] * count
                yield f"extreme_phase/system{system}/phase{phase}/channels{count}", (
                    2.6e6, -1.1e6, channels, 513, -20., 37)
    for count in (31, 32, 33, 65):
        for nav in (False, True):
            for seed in (1, 2**64 - 1):
                channels = _mixed_channels(count, nav)
                yield f"mixed_systems/channels{count}/nav{nav}/seed{seed}", (
                    4.092e6, 1.1e6, channels, 4097, -40., seed)


def _mixed_channels(count, nav=False):
    channels = _channels(count, nav)
    for index, ch in enumerate(channels):
        system = index % 5
        ch.update(system=system, prn=(ch["prn"] - 1) % (32, 24, 36, 37, 32)[system] + 1)
    return channels


def _native_cases():
    for samples in (2600, 260000, 2600000):
        for count in (0, 1, 12, 32, 33, 65):
            yield f"native/GPS/samples{samples}/channels{count}", (
                2.6e6, 0., _channels(count), samples, -20., 42)
    for count in (12, 32, 65):
        yield f"native/no_noise/samples260000/channels{count}", (
            2.6e6, 0., _channels(count), 260000, -10000., 42)
        for system in (1, 2, 3):
            channels = _channels(count)
            for ch in channels:
                ch.update(system=system, prn=(ch["prn"] - 1) % (24, 36, 37)[system - 1] + 1)
            yield f"native/system{system}/samples260000/channels{count}", (
                2.6e6, 0., channels, 260000, -20., 42)
        yield f"native/mixed_nav/samples260000/channels{count}", (
            2.6e6, 0., _mixed_channels(count, True), 260000, -20., 42)


def _stage_names():
    yield from (name for name, _ in _native_cases())
    for samples in (2600, 260000, 2600000):
        for count in (0, 12, 33):
            yield f"public/samples{samples}/channels{count}"
    for samples in (2600, 260000):
        for count in (0, 12, 65):
            yield f"cold/samples{samples}/channels{count}"
    for count in (12, 1200, 12000):
        for samples in (2600, 260000):
            yield f"urban/triangles{count}/rays12/samples{samples}"


def _validate(native, workspace):
    records, retained = {}, []
    for index, (name, arguments) in enumerate(_cases()):
        input_hash = _input_digest(arguments)
        cold, warm = native.generate_signal(*arguments), workspace.generate_signal(*arguments)
        np.testing.assert_array_equal(warm.view(np.uint32), cold.view(np.uint32), err_msg=name)
        assert _input_digest(arguments) == input_hash
        records[name] = dict(input_sha256=input_hash, output_sha256=_tree_hash(warm),
                             shape=list(warm.shape), dtype=str(warm.dtype))
        if index % 41 == 0:
            retained.append((warm, _tree_hash(warm)))
        for output, digest in retained:
            assert _tree_hash(output) == digest, "Later epoch overwrote returned array"
    for index, (samples, count, nav) in enumerate((
            (1, 1, False), (8193, 65, True), (2, 32, False), (260000, 33, True),
            (257, 0, False), (2600, 12, True), (1, 33, True), (26000, 65, False))):
        arguments = (4.092e6, -1e6, _mixed_channels(count, nav), samples, -20., 37)
        input_hash = _input_digest(arguments)
        output = workspace.generate_signal(*arguments)
        np.testing.assert_array_equal(output.view(np.uint32),
                                      native.generate_signal(*arguments).view(np.uint32))
        records[f"reuse/{index}"] = dict(input_sha256=input_hash, output_sha256=_tree_hash(output),
                                          shape=list(output.shape), dtype=str(output.dtype))
        retained.append((output, _tree_hash(output)))
        for channel in arguments[2]:
            if "nav_bits" in channel:
                channel["nav_bits"].fill(0)
        for array, digest in retained:
            assert _tree_hash(array) == digest
    return records


class _Profile:
    def __init__(self, extension):
        sys.path.insert(0, str(ROOT / "python"))
        self.native = _load_extension(extension.resolve(), "gnss_gpu._gnss_gpu_signal_sim")
        sys.modules["gnss_gpu._gnss_gpu_signal_sim"] = self.native
        import gnss_gpu
        from gnss_gpu.signal_sim import SignalSimulator

        gnss_gpu._gnss_gpu_signal_sim = self.native
        self.workspace = self.native.DeviceSignalWorkspace()
        self.simulator = SignalSimulator(noise_seed=42)
        self.native_cases = dict(_native_cases())
        self.urban = {}

    def stage(self, name):
        if name.startswith("native/"):
            arguments = self.native_cases[name]
            return lambda: self.workspace.generate_signal(*arguments), _input_digest(arguments)
        if name.startswith(("public/", "cold/")):
            kind, samples, count = name.split("/")
            samples, count = int(samples[7:]), int(count[8:])
            channels = _channels(count)
            arguments = (2.6e6, 0., channels, samples, -20., 42)
            if kind == "cold":
                return lambda: self.native.generate_signal(*arguments), _input_digest(arguments)
            return lambda: self.simulator.generate_epoch(channels, samples), _input_digest(arguments)
        from gnss_gpu.bvh import BVHAccelerator
        from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

        _, count, _, samples = name.split("/")
        count, samples = int(count[9:]), int(samples[7:])
        if count not in self.urban:
            rx = np.array([6378137., 0., 0.])
            sat = _make_satellites(12)[:, [2, 0, 1]] + rx
            mesh = _make_building_mesh(count // 12)[:, :, [2, 0, 1]] + rx
            sim = UrbanSignalSimulator(BVHAccelerator(mesh))
            sim.sim.noise_seed = 37
            self.urban[count] = sim, rx, sat, mesh
        sim, rx, sat, mesh = self.urban[count]
        return lambda: sim.compute_epoch(rx, sat, n_samples=samples), _input_digest((rx, sat, mesh))

    def measure(self, name, repeats):
        call, input_hash = self.stage(name)
        digest = _tree_hash(call())
        for _ in range(3):
            assert _tree_hash(call()) == digest
        timings = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = call()
            timings.append((time.perf_counter_ns() - start) / 1e6)
            assert _tree_hash(output) == digest, f"Timed output changed: {name}"
        return dict(median_ms=float(np.median(timings)), samples_ms=timings,
                    input_sha256=input_hash, output_sha256=digest,
                    all_timed_outputs_identical=True)

    def close(self):
        self.workspace.close()
        self.simulator.release_gpu()
        for sim, *_ in self.urban.values():
            sim.sim.release_gpu()
            sim.building_model.release_gpu()
        assert self.workspace.allocated_bytes == 0


def _serve(extension):
    profile = _Profile(extension)
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "close":
                break
            try:
                if request["command"] == "validate":
                    result = _validate(profile.native, profile.workspace)
                else:
                    result = profile.measure(request["stage"], request["repeats"])
                print(json.dumps(dict(result=result)), flush=True)
            except Exception:
                print(json.dumps(dict(error=traceback.format_exc())), flush=True)
    finally:
        profile.close()


class _Worker:
    def __init__(self, extension):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--extension", str(extension.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=ROOT)
        assert self._read()["ready"]

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(self.process.stderr.read() or "Worker exited without output")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def request(self, **command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        return self._read()["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(command="close")) + "\n")
            self.process.stdin.flush()
        self.process.wait(timeout=30)
        assert self.process.returncode == 0, self.process.stderr.read()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


def _compare(args):
    reference = json.loads((ROOT / "benchmarks/signal_workspace_gpu_20261012.json").read_text(encoding="utf-8"))
    hashes = dict(baseline={path: _sha(args.baseline_dir / Path(path).name) for path in SOURCE_PATHS},
                  candidate={path: _sha(ROOT / path) for path in SOURCE_PATHS})
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert hashes["baseline"] == reference["source_sha256"]["candidate"], "Wrong baseline sources"
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"], "Wrong baseline binary"
    assert all(hashes["baseline"][path] == hashes["candidate"][path] for path in SOURCE_PATHS[1:])
    workers, measurements = {}, {}
    validation = {}
    try:
        # Startup and every command are sequential; never overlap GPU work.
        for label, path in dict(baseline=args.baseline, candidate=args.candidate).items():
            workers[label] = _Worker(path)
        for label, worker in workers.items():
            validation[label] = worker.request(command="validate")
        assert validation["baseline"] == validation["candidate"], "IQ/input/shape hash mismatch"
        print(f"{len(validation['baseline'])} IQ cases have identical bit hashes", flush=True)
        if not args.validation_only:
            names = list(_stage_names())
            if args.stages:
                names = [name for name in names if any(part in name for part in args.stages)]
            for name in names:
                runs = {label: [] for label in workers}
                for round_index in range(args.rounds):
                    order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
                    for label in order:
                        runs[label].append(workers[label].request(
                            command="measure", stage=name, repeats=args.repeats))
                outputs = {run["output_sha256"] for group in runs.values() for run in group}
                inputs = {run["input_sha256"] for group in runs.values() for run in group}
                assert len(outputs) == len(inputs) == 1, name
                medians = {label: [run["median_ms"] for run in group] for label, group in runs.items()}
                old, new = [float(np.median(medians[label])) for label in workers]
                measurements[name] = dict(
                    baseline_ms=old, candidate_ms=new, speedup=old / new,
                    round_medians_ms=medians, runs=runs, output_sha256=outputs.pop(),
                    input_sha256=inputs.pop(), all_timed_outputs_identical=True)
                print(f"{name}: {old:.4f} -> {new:.4f} ms ({old / new:.3f}x)", flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    dependencies = (
        "benchmarks/bench_signal_kernel_compare.py", "benchmarks/bench_signal_sim_compare.py",
        "benchmarks/bench_bvh_combined_compare.py", "benchmarks/bench_bvh_resident_compare.py",
        "benchmarks/bench_bvh_multipath_compare.py", "benchmarks/bench_raytrace.py",
    )
    return dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
        baseline="preceding reusable signal workspace implementation",
        method="persistent actual production workers; sequential GPU work; alternating per-stage build order by round; nonexclusive host",
        scope="native includes conversion/transfers/kernel/synchronization/host IQ; public adds validation; "
              "urban adds LOS/reflection/channel metadata/actual CUDA IQ; cold includes workspace lifetime; "
              "CPU mesh/BVH construction, CUDA/code-table initialization, IPC and output hashes excluded",
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_stage=3,
        source_sha256=hashes, extension_sha256=extensions,
        bvh_extension_sha256={str(path.relative_to(ROOT)): _sha(path) for path in
                              (ROOT / "python/gnss_gpu").glob("_bvh*.pyd")},
        benchmark_sha256={path: _sha(ROOT / path) for path in dependencies},
        validation=dict(cases=len(validation["baseline"]), all_iq_bits_identical=True,
                        baseline_by_case=validation["baseline"], candidate_by_case=validation["candidate"]),
        measurements=measurements)


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
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    if args.extension:
        _serve(args.extension)
        return
    if not (args.baseline and args.candidate and args.baseline_dir and args.output):
        parser.error("baseline/candidate/baseline-dir/output required")
    report = _compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
