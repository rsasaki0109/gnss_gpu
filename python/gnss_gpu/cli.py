"""GPU-first command line onboarding for :mod:`gnss_gpu`."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if name == "nvcc":
        for variable in ("CUDA_PATH", "CUDA_HOME"):
            root = os.environ.get(variable)
            if root:
                candidate = Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
                if candidate.is_file():
                    return str(candidate)
    return None


def _probe_command(name: str, arguments: Sequence[str], label: str) -> Check:
    executable = _executable(name)
    if not executable:
        return Check(label, "FAIL", f"{name} was not found on PATH", f"Install {name} and reopen the shell.")
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(label, "FAIL", str(exc), f"Verify that {executable} can run in this shell.")
    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0].strip() if output else f"exit status {result.returncode}"
    if result.returncode:
        return Check(label, "FAIL", detail, f"Verify the {name} installation and PATH.")
    return Check(label, "PASS", detail)


def _probe_nvidia() -> Check:
    executable = _executable("nvidia-smi")
    if not executable:
        return Check(
            "NVIDIA GPU/driver",
            "FAIL",
            "nvidia-smi was not found on PATH",
            "Install a current NVIDIA driver and reopen the shell.",
        )
    queries = (
        "name,driver_version,memory.total,compute_cap",
        "name,driver_version,memory.total",
    )
    for query in queries:
        try:
            result = subprocess.run(
                [executable, f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Check("NVIDIA GPU/driver", "FAIL", str(exc), "Verify the NVIDIA driver installation.")
        if result.returncode == 0 and result.stdout.strip():
            return Check("NVIDIA GPU/driver", "PASS", result.stdout.strip().splitlines()[0])
    detail = (result.stderr or result.stdout).strip().splitlines()
    return Check(
        "NVIDIA GPU/driver",
        "FAIL",
        detail[0] if detail else f"nvidia-smi exited with status {result.returncode}",
        "Update or repair the NVIDIA driver.",
    )


def _probe_module(module_name: str, label: str) -> Check:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return Check(label, "WARN", "not installed", "Run `python python/gnss_gpu/cli.py build` from the repository root.")
        return Check(label, "FAIL", f"dependency missing: {exc.name}", "Repair the Python/CUDA runtime installation.")
    except (ImportError, OSError) as exc:
        return Check(label, "FAIL", str(exc), "Check CUDA runtime DLL/shared-library paths, then rebuild.")
    origin = getattr(module, "__file__", None)
    return Check(label, "PASS", f"imported from {origin}" if origin else "imported successfully")


def _gpu_roundtrip() -> dict[str, object]:
    from gnss_gpu.acquisition import Acquisition
    from gnss_gpu.signal_sim import SignalSimulator

    started = time.perf_counter()
    simulator = SignalSimulator(noise_seed=1)
    sample_count = int(simulator.sampling_freq * 1e-3)
    channels = [{
        "prn": 1,
        "code_phase": 0.0,
        "carrier_phase": 0.0,
        "doppler_hz": 750.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }]
    iq = simulator.generate_epoch(channels, n_samples=sample_count)
    acquisition = Acquisition(
        sampling_freq=simulator.sampling_freq,
        intermediate_freq=simulator.intermediate_freq,
    )
    results = acquisition.acquire(iq[0::2].copy(), prn_list=[1])
    if len(results) != 1 or not results[0].get("acquired"):
        raise RuntimeError(f"PRN 1 was not acquired: {results!r}")
    return {
        "preset": "signal-acquisition",
        "backend": "CUDA",
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "sample_count": sample_count,
        "prn": int(results[0]["prn"]),
        "doppler_hz": float(results[0]["doppler_hz"]),
        "acquired": True,
    }


def collect_diagnostics(*, runtime_test: bool = True) -> list[Check]:
    python_ok = sys.version_info >= (3, 9)
    checks = [
        Check(
            "Python",
            "PASS" if python_ok else "FAIL",
            f"{platform.python_implementation()} {platform.python_version()} ({sys.executable})",
            "Install Python 3.9 or newer." if not python_ok else "",
        ),
        Check("Platform", "PASS", f"{platform.system()} {platform.release()} / {platform.machine()}"),
        _probe_nvidia(),
        _probe_command("nvcc", ["--version"], "CUDA compiler"),
        _probe_command("cmake", ["--version"], "CMake"),
    ]
    core = _probe_module("gnss_gpu._gnss_gpu", "Native core")
    signal_sim = _probe_module("gnss_gpu._gnss_gpu_signal_sim", "Signal simulator binding")
    acquisition = _probe_module("gnss_gpu._gnss_gpu_acq", "Acquisition binding")
    checks.extend((core, signal_sim, acquisition))
    bindings_ready = signal_sim.status == "PASS" and acquisition.status == "PASS"
    if runtime_test and bindings_ready:
        try:
            result = _gpu_roundtrip()
        except Exception as exc:  # CUDA errors vary by driver/runtime version.
            checks.append(Check("CUDA runtime round-trip", "FAIL", str(exc), "Run `gnss-gpu doctor` after checking the driver/runtime match."))
        else:
            checks.append(Check("CUDA runtime round-trip", "PASS", f"PRN {result['prn']} acquired in {result['elapsed_ms']} ms"))
    elif runtime_test:
        checks.append(Check("CUDA runtime round-trip", "WARN", "not run because bindings are missing", "Run `python python/gnss_gpu/cli.py build`, then rerun doctor."))
    return checks


def readiness(checks: Sequence[Check]) -> str:
    required_build = {"Python", "NVIDIA GPU/driver", "CUDA compiler", "CMake"}
    if any(check.status == "FAIL" and check.name in required_build for check in checks):
        return "NOT READY"
    runtime = next((check for check in checks if check.name == "CUDA runtime round-trip"), None)
    if runtime and runtime.status == "PASS":
        return "READY TO RUN"
    if any(check.status == "FAIL" for check in checks):
        return "NOT READY"
    return "READY TO BUILD"


def _print_diagnostics(checks: Sequence[Check]) -> str:
    state = readiness(checks)
    print("gnss_gpu GPU doctor")
    print("=" * 72)
    for check in checks:
        print(f"[{check.status:<4}] {check.name:<27} {check.detail}")
        if check.remedy and check.status != "PASS":
            print(f"       fix: {check.remedy}")
    print("-" * 72)
    print(f"State: {state}")
    if state == "READY TO BUILD":
        print("Next:  python python/gnss_gpu/cli.py build")
    elif state == "READY TO RUN":
        print("Next:  gnss-gpu run --preset signal-acquisition")
    else:
        print("Fix the FAIL items above, then rerun `gnss-gpu doctor`.")
    return state


def _project_root(path: Path) -> Path:
    root = path.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "CMakeLists.txt").is_file():
        raise ValueError(f"{root} is not a gnss_gpu source checkout (pyproject.toml/CMakeLists.txt missing)")
    return root


def build_command(root: Path, architecture: str, no_build_isolation: bool) -> list[str]:
    command = [sys.executable, "-m", "pip", "install", "--verbose", str(root), "--config-settings", f"cmake.define.CMAKE_CUDA_ARCHITECTURES={architecture}"]
    if no_build_isolation:
        command.append("--no-build-isolation")
    return command


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = collect_diagnostics(runtime_test=not args.skip_runtime)
    state = _print_diagnostics(checks)
    if args.json:
        Path(args.json).write_text(json.dumps({"state": state, "checks": [asdict(c) for c in checks]}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")
    return 1 if state == "NOT READY" else 0


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        root = _project_root(args.project_root)
    except ValueError as exc:
        print(f"Build error: {exc}", file=sys.stderr)
        return 2
    command = build_command(root, args.architecture, args.no_build_isolation)
    print("Running:", subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        print("Build failed. Run `gnss-gpu doctor` for environment-specific fixes.", file=sys.stderr)
        return result.returncode
    print("Build installed successfully.")
    print("Next: gnss-gpu doctor")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.preset != "signal-acquisition":
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        return 2
    try:
        result = _gpu_roundtrip()
    except Exception as exc:
        print(f"GPU preset failed: {exc}", file=sys.stderr)
        print("Run `gnss-gpu doctor` to diagnose the environment.", file=sys.stderr)
        return 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **result,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("GPU signal simulation -> acquisition: PASS")
    print(f"Backend:     {result['backend']}")
    print(f"PRN:         {result['prn']} acquired")
    print(f"Doppler:     {result['doppler_hz']:.1f} Hz")
    print(f"Runtime:     {result['elapsed_ms']:.2f} ms")
    print(f"Manifest:    {manifest_path}")
    print("Next:        change a preset parameter or run a GPU benchmark")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnss-gpu", description="GPU-first setup and demo runner for gnss_gpu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="diagnose the NVIDIA/CUDA build and runtime environment")
    doctor.add_argument("--skip-runtime", action="store_true", help="check imports without launching a CUDA kernel")
    doctor.add_argument("--json", metavar="PATH", help="also write a machine-readable diagnostic report")
    doctor.set_defaults(handler=_cmd_doctor)

    build = subparsers.add_parser("build", help="build and install CUDA extensions from this checkout")
    build.add_argument("--project-root", type=Path, default=Path.cwd(), help="gnss_gpu checkout (default: current directory)")
    build.add_argument("--architecture", default="native", help="CMake CUDA architecture value (default: native)")
    build.add_argument("--no-build-isolation", action="store_true", help="reuse build packages from the active environment")
    build.add_argument("--dry-run", action="store_true", help="print the pip build command without running it")
    build.set_defaults(handler=_cmd_build)

    run = subparsers.add_parser("run", help="run a checked GPU onboarding preset")
    run.add_argument("--preset", default="signal-acquisition", choices=("signal-acquisition",))
    run.add_argument("--output-dir", type=Path, help="directory for manifest.json (default: runs/<UTC timestamp>)")
    run.set_defaults(handler=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
