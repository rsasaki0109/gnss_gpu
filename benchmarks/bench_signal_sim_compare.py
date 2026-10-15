"""Compare two signal-simulation builds with exact IQ and native-call timing.

Archive a baseline extension/source, then rebuild with the same configuration.
Timed calls include Python/native conversion, allocation, transfers, kernels,
synchronization, output copies, and frees; one-time code-table setup is excluded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np


def _load_extension(path, name):
    handles = []
    if os.name == "nt":
        directories = [path.parent]
        for variable in ("CUDA_PATH", "CUDA_HOME"):
            if root := os.environ.get(variable):
                directories.append(Path(root) / "bin")
        handles = [os.add_dll_directory(str(directory)) for directory in directories
                   if directory.is_dir()]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._dll_directory_handles = handles
    return module


def _channel(prn=1, **overrides):
    result = dict(prn=prn, system=0, code_phase=0., carrier_phase=0.,
                  doppler_hz=0., amplitude=1., nav_bit=1)
    result.update(overrides)
    return result


def _channels(count, nav=False):
    result = []
    for index in range(count):
        channel = _channel(index % 32 + 1, code_phase=index - .25,
                           carrier_phase=index * .37, doppler_hz=index * 100. - 550.,
                           amplitude=1. / (index + 1), nav_bit=(-1) ** index)
        if nav and index % 3 != 0:
            channel.update(nav_bits=np.array([1, -1, 0, 1, -1], dtype=np.int32),
                           nav_bit_rate=26000. * (index + 1))
        result.append(channel)
    return result


def _input_digest(arguments):
    serialized = json.dumps(arguments, sort_keys=True,
                            default=lambda value: value.tolist()).encode()
    return hashlib.sha256(serialized).hexdigest()


def _digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validation_cases():
    for samples in (1, 2, 255, 256, 257, 2600, 26000, 260000):
        for count in (0, 1, 11, 31, 32, 33, 65):
            for noise in (-10000., -20., 0.):
                yield f"samples{samples}/channels{count}/noise{noise}", (
                    2.6e6, 0., _channels(count), samples, noise, 42)
    for system, max_prn in enumerate((32, 24, 36, 37, 32)):
        for prn in (1, max_prn):
            for seed in (1, 2**64 - 1):
                for phase in (-4092.75, 1023. - 1e-12):
                    channels = [_channel(prn, system=system, code_phase=phase,
                                         carrier_phase=-1e5, doppler_hz=-12345.67,
                                         amplitude=-.731, nav_bit=-1)]
                    yield f"system{system}/prn{prn}/seed{seed}/phase{phase}", (
                        4.092e6, 1.25e6, channels, 4097, -40., seed)
    for samples in (1, 257, 8193):
        for length in (0, 1, 2, 7):
            for rate in (50., 2.6e6 / 3, 2.6e6):
                channels = _channels(11, nav=True)
                channels[0].update(nav_bits=np.resize(
                    np.array([-3, 0, 2, 1, -1], dtype=np.int32), length),
                    nav_bit_rate=rate, nav_bit=-2)
                channels[-1].update(system=3, prn=37)
                yield f"nav/samples{samples}/length{length}/rate{rate}", (
                    2.6e6, -1e6, channels, samples, -20., 2**64 - 1)
    for count in (1, 11, 32, 65):
        channels = _channels(count)
        channels += [_channel(33), _channel(25, system=1), _channel(37, system=2),
                     _channel(38, system=3), _channel(33, system=4), _channel(system=99)]
        yield f"invalid_mixed/channels{count}", (2.6e6, 0., channels, 1025, -20., 1)
    yield "all_invalid", (2.6e6, 0., [_channel(33), _channel(system=99)], 257, -20., 42)
    for amplitudes in ((1e8, 1., -1e8), (1e8, -1e8, 1.), (-0., 0., -0.)):
        channels = [_channel(amplitude=amplitude, carrier_phase=.23,
                             doppler_hz=700.) for amplitude in amplitudes]
        yield f"sum_order/{amplitudes}", (2.6e6, 0., channels, 8193, -10000., 42)
    for seed in (0, 1, 42, 2**64 - 1):
        yield f"seed{seed}/noise_disabled", (2.6e6, 0., _channels(11), 4097, -10000., seed)
    for frequency in (1.023e6, 2.046e6, 2.6e6, 16.368e6):
        channels = [_channel(code_phase=phase, carrier_phase=phase / 13.,
                             doppler_hz=-999.5, amplitude=.125)
                    for phase in (-1023., -1e-12, 0., 1. - 1e-12, 1023.)]
        yield f"chip_boundaries/fs{frequency}", (frequency, -1.1e6, channels, 65537, -40., 1)
    yield "nav_none", (2.6e6, 0., [_channel(nav_bits=None)], 257, -20., 1)
    yield "nav_static_nonunit", (2.6e6, 0., [_channel(nav_bit=bit) for bit in (-3, 0, 2)],
                                 4097, -20., 42)
    yield "underflow", (2.6e6, 0., [_channel(amplitude=1e-40), _channel(amplitude=-1e-40)],
                        4097, -10000., 42)


def _validate(before, after):
    records = {}
    for name, arguments in _validation_cases():
        input_hash = _input_digest(arguments)
        old = before.generate_signal(*arguments)
        new = after.generate_signal(*arguments)
        np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32), err_msg=name)
        if _input_digest(arguments) != input_hash:
            raise RuntimeError(f"Input mutated: {name}")
        records[name] = {"input_sha256": input_hash, "baseline_sha256": _digest(old),
                         "candidate_sha256": _digest(new), "iq_floats": int(old.size)}
    return {"cases": len(records), "bit_identical": True, "records": records}


def _timing_cases():
    for samples in (2600, 26000, 260000, 2600000):
        for count in (0, 1, 11, 32):
            yield f"samples{samples}/channels{count}", (
                2.6e6, 0., _channels(count), samples, -20., 42)
    for samples in (2600, 26000, 260000):
        for count in (11, 32, 65):
            yield f"nav/samples{samples}/channels{count}", (
                2.6e6, 0., _channels(count, nav=True), samples, -20., 42)


def _measure(before, after, arguments, rounds, repeats):
    modules = {"baseline": before, "candidate": after}
    for module in modules.values():
        for _ in range(3):
            module.generate_signal(*arguments)
    samples = {label: [] for label in modules}
    for round_index in range(rounds):
        timings = {label: [] for label in modules}
        for repetition in range(repeats):
            order = ("baseline", "candidate") if (round_index + repetition) % 2 == 0 else (
                "candidate", "baseline")
            for label in order:
                start = time.perf_counter_ns()
                modules[label].generate_signal(*arguments)
                timings[label].append((time.perf_counter_ns() - start) / 1e6)
        for label in samples:
            samples[label].append(timings[label])
    medians = {label: [float(np.median(sample)) for sample in data]
               for label, data in samples.items()}
    old, new = [float(np.median(medians[label])) for label in modules]
    return {"baseline_ms": old, "candidate_ms": new, "speedup": old / new,
            "round_medians_ms": medians, "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    before = _load_extension(args.baseline.resolve(), "baseline._gnss_gpu_signal_sim")
    after = _load_extension(args.candidate.resolve(), "candidate._gnss_gpu_signal_sim")
    validation = _validate(before, after)
    print(f"{validation['cases']} validation cases are bit-identical", flush=True)
    measurements = {}
    if not args.validation_only:
        for name, arguments in _timing_cases():
            measurements[name] = _measure(before, after, arguments, args.rounds, args.repeats)
            record = measurements[name]
            print(f"{name}: {record['baseline_ms']:.3f} -> {record['candidate_ms']:.3f} ms "
                  f"({record['speedup']:.2f}x)", flush=True)
    source = Path(__file__).resolve().parents[1] / "src/signal_sim/signal_sim.cu"
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "method": "same process, alternating build order per call, synchronized native wall time",
        "scope": "native validation/conversion/allocation/upload/kernels/output/free included; "
                 "one-time code-table initialization and Python wrapper validation excluded",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_build": 3,
        "source_sha256": {"baseline": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                          "candidate": hashlib.sha256(source.read_bytes()).hexdigest()},
        "extension_sha256": {
            "baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        },
        "validation": validation, "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
