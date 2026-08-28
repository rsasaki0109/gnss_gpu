"""GPU-first command line onboarding for :mod:`gnss_gpu`."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

RUN_MANIFEST_SCHEMA = "gnss_gpu_run_manifest_v1"
RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_COMPARISON_SCHEMA = "gnss_gpu_run_comparison_v1"
DEFAULT_PLATEAU_GML = Path("data") / "sample_plateau.gml"


class UrbanNavRunError(RuntimeError):
    """User-facing UrbanNav error without importing the optional data module.

    ``urbannav_cli`` pulls in NumPy and the package I/O stack.  The historical
    source-checkout entry point (``python python/gnss_gpu/cli.py doctor``) must
    remain usable before those runtime dependencies are installed, so the
    UrbanNav implementation is loaded only when one of its commands is used.
    The implementation's error also derives from :class:`RuntimeError`, which
    keeps this compatibility alias suitable for the CLI's error handling.
    """


def _load_urbannav_cli():
    """Load the optional UrbanNav command implementation on demand.

    Keeping this import behind a command boundary is important for a fresh
    checkout: ``doctor``, ``build`` and ``--help`` intentionally use only the
    standard library.  A source checkout can opt into the data commands with
    ``PYTHONPATH=python python -m gnss_gpu.cli ...`` after installing runtime
    dependencies.
    """

    try:
        return importlib.import_module("gnss_gpu.urbannav_cli")
    except (ImportError, OSError) as exc:
        missing = getattr(exc, "name", None)
        detail = f" (missing {missing})" if missing else ""
        raise RuntimeError(
            "UrbanNav data commands require the installed gnss_gpu package and "
            f"runtime dependencies{detail}. From a source checkout, install the "
            "package and run `PYTHONPATH=python python -m gnss_gpu.cli data inspect PATH` "
            "or use the installed `gnss-gpu` command."
        ) from exc


def inspect_input(path: str | Path):
    """Compatibility wrapper that lazily dispatches to UrbanNav inspection."""

    return _load_urbannav_cli().inspect_input(path)


def format_inspection(result: object) -> str:
    """Compatibility wrapper that lazily formats an UrbanNav inspection."""

    return _load_urbannav_cli().format_inspection(result)


def run_urbannav_pf(*args: object, **kwargs: object):
    """Compatibility wrapper that lazily dispatches the UrbanNav PF run."""

    return _load_urbannav_cli().run_urbannav_pf(*args, **kwargs)


class ManifestError(ValueError):
    """Raised when a run manifest cannot be safely loaded or compared."""


class PlateauPresetError(RuntimeError):
    """User-facing error for missing PLATEAU data or GPU prerequisites."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _package_version() -> str:
    """Return the installed/source package version without requiring a build."""

    try:
        from gnss_gpu._version import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _safe_platform_component(getter: object, fallback: str = "unknown") -> str:
    """Return a platform metadata component without allowing probe failures out.

    Python's platform helpers have historically used Windows WMI for some of
    their uname fallbacks. Metadata is diagnostic only, so a missing/broken provider must
    never prevent a run manifest from being written.
    """

    try:
        value = getter()  # type: ignore[operator]
    except BaseException:
        return fallback
    text = str(value).strip()
    return text or fallback


def _safe_platform_info() -> str:
    """Build a stable human-readable platform label without Windows WMI.

    On Windows use the process/runtime APIs and environment variables directly;
    high-level platform helpers can invoke WMI. Other systems use the
    individual platform getters, each independently guarded.
    """

    if os.name == "nt" or sys.platform.startswith("win"):
        release = "unknown"
        get_windows_version = getattr(sys, "getwindowsversion", None)
        if callable(get_windows_version):
            try:
                version = get_windows_version()
                release = ".".join(
                    str(part)
                    for part in (
                        getattr(version, "major", ""),
                        getattr(version, "minor", ""),
                        getattr(version, "build", ""),
                    )
                    if str(part) != ""
                ) or release
            except BaseException:
                pass
        machine = (
            os.environ.get("PROCESSOR_ARCHITEW6432")
            or os.environ.get("PROCESSOR_ARCHITECTURE")
            or "unknown"
        )
        return f"Windows-{release}-{machine}"

    return "-".join(
        (
            _safe_platform_component(getattr(platform, "system", None)),
            _safe_platform_component(getattr(platform, "release", None)),
            _safe_platform_component(getattr(platform, "machine", None)),
        )
    )


def _git_value(root: Path, *arguments: str) -> str:
    """Read a git value, returning ``unknown`` when the checkout is unavailable."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    value = completed.stdout.strip()
    return value or "unknown"


def _repo_root(start: Path | None = None) -> Path:
    """Find the source checkout root for paths recorded in a manifest."""

    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        for root in (candidate, *candidate.parents):
            if (root / "pyproject.toml").is_file() and (root / "CMakeLists.txt").is_file():
                return root
    return Path.cwd().resolve()


def _path_label(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_gpu_info() -> dict[str, object]:
    """Collect stable GPU metadata while remaining usable on GPU-less CI."""

    info: dict[str, object] = {
        "available": False,
        "name": None,
        "driver_version": None,
        "memory_total_mb": None,
        "compute_capability": None,
    }
    executable = _executable("nvidia-smi")
    if not executable:
        return info

    queries = (
        "name,driver_version,memory.total,compute_cap",
        "name,driver_version,memory.total",
        "name,driver_version",
    )
    for query in queries:
        try:
            result = subprocess.run(
                [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode or not result.stdout.strip():
            continue
        fields = [field.strip() for field in result.stdout.strip().splitlines()[0].split(",")]
        if fields:
            info["name"] = fields[0]
            info["available"] = True
        if len(fields) >= 2:
            info["driver_version"] = fields[1]
        if len(fields) >= 3:
            try:
                info["memory_total_mb"] = float(fields[2])
            except ValueError:
                info["memory_total_mb"] = fields[2]
        if len(fields) >= 4:
            info["compute_capability"] = fields[3]
        break
    return info


def _input_record(path: Path, root: Path) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    return {
        "path": _path_label(path, root),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _artifact_record(path: Path, root: Path) -> dict[str, object] | None:
    path = Path(path).resolve()
    if not path.is_file():
        return None
    return {
        "path": _path_label(path, root),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _json_scalar(value: object) -> object:
    """Convert NumPy-like scalar values without importing NumPy in the CLI."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except Exception:
            return str(value)
        if isinstance(converted, (str, int, float, bool)) or converted is None:
            return converted
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return str(value)


def _metrics_from_result(result: Mapping[str, object]) -> dict[str, object]:
    """Normalize preset output into scalar, directly comparable metrics."""

    raw_metrics = result.get("metrics")
    metrics: dict[str, object] = (
        {str(key): _json_scalar(value) for key, value in raw_metrics.items()}
        if isinstance(raw_metrics, Mapping)
        else {}
    )
    if "runtime_ms" not in metrics and result.get("elapsed_ms") is not None:
        metrics["runtime_ms"] = _json_scalar(result["elapsed_ms"])

    suite = result.get("suite")
    if isinstance(suite, Mapping):
        rows = suite.get("rows")
        if isinstance(rows, list):
            for raw_row in rows:
                if not isinstance(raw_row, Mapping) or not raw_row.get("estimator"):
                    continue
                prefix = str(raw_row["estimator"]).lower()
                for source, target in (
                    ("baseline_rms_m", f"{prefix}_baseline_rms_m"),
                    ("mask_soft_rms_m", f"{prefix}_mask_soft_rms_m"),
                    ("robust_rms_m", f"{prefix}_robust_rms_m"),
                    ("baseline_p50_m", f"{prefix}_baseline_p50_m"),
                    ("mask_soft_p50_m", f"{prefix}_mask_soft_p50_m"),
                    ("rms_gain_pct", f"{prefix}_rms_gain_pct"),
                    ("mask_soft_wins", f"{prefix}_mask_soft_wins"),
                    ("n_solved_epochs", f"{prefix}_n_solved_epochs"),
                ):
                    if source in raw_row and raw_row[source] != "":
                        metrics[target] = _json_scalar(raw_row[source])
        for source, target in (
            ("best_mask_soft_rms_m", "best_mask_soft_rms_m"),
            ("min_rms_gain_pct", "rms_gain_pct"),
        ):
            if source in suite:
                metrics[target] = _json_scalar(suite[source])

    # Keep the compact names useful for existing scripts and operators.
    if "best_mask_soft_rms_m" in metrics:
        metrics.setdefault("rms_m", metrics["best_mask_soft_rms_m"])
    if "spp_baseline_rms_m" in metrics:
        metrics.setdefault("baseline_rms_m", metrics["spp_baseline_rms_m"])
    if "spp_mask_soft_p50_m" in metrics:
        metrics.setdefault("p50_m", metrics["spp_mask_soft_p50_m"])
    for key in ("n_epochs", "n_requested_epochs", "n_satellites", "n_triangles", "nlos_fraction"):
        if key in result:
            metrics.setdefault(key, _json_scalar(result[key]))
    return metrics


def build_run_manifest(
    *,
    preset: str,
    result: Mapping[str, object],
    parameters: Mapping[str, object],
    input_paths: Sequence[Path] = (),
    artifact_paths: Mapping[str, Path] | None = None,
    repo_root: Path | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build the shared v1 schema used by every ``gnss-gpu run`` preset."""

    root = _repo_root(repo_root)
    inputs = [_input_record(Path(path), root) for path in input_paths]
    input_hashes = {str(item["path"]): str(item["sha256"]) for item in inputs}
    artifacts: dict[str, object] = {}
    for name, raw_path in (artifact_paths or {}).items():
        record = _artifact_record(Path(raw_path), root)
        if record is not None:
            artifacts[str(name)] = record

    git_sha = _git_value(root, "rev-parse", "HEAD")
    gpu = _query_gpu_info()
    result_gpu = result.get("gpu")
    if isinstance(result_gpu, Mapping):
        gpu.update({str(key): _json_scalar(value) for key, value in result_gpu.items()})
    backend = str(result.get("backend", "CUDA"))
    metrics = _metrics_from_result(result)
    manifest: dict[str, object] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "manifest_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": _package_version(),
        "git_sha": git_sha,
        "commit_sha": git_sha,
        "git": {"sha": git_sha},
        "backend": backend,
        "gpu": gpu,
        "gpu_info": gpu,
        "inputs": inputs,
        "input_hashes": input_hashes,
        "input_sha256": input_hashes,
        "parameters": {str(key): _json_scalar(value) for key, value in parameters.items()},
        "metrics": metrics,
        "artifacts": artifacts,
        "artifact_hashes": {
            name: record["sha256"]
            for name, record in artifacts.items()
            if isinstance(record, Mapping) and "sha256" in record
        },
        "preset": str(preset),
        "command": list(command or ("gnss-gpu", "run", "--preset", str(preset))),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": _safe_platform_info(),
        },
        # Preserve the full preset result for post-run inspection.  Consumers
        # should use ``metrics`` and the hashed artifact records for comparison.
        "result": _json_scalar(dict(result)),
    }
    # Flat aliases keep manifests created by the first GPU-first CLI readable
    # by existing scripts while all new consumers use the common schema above.
    for key in ("elapsed_ms", "sample_count", "prn", "doppler_hz", "acquired", "ray_source", "gml_path"):
        if key in result:
            manifest[key] = _json_scalar(result[key])
    manifest["python"] = platform.python_version()
    manifest["platform"] = _safe_platform_info()
    return manifest


def _default_plateau_gml(root: Path | None = None) -> Path:
    return (_repo_root(root) / DEFAULT_PLATEAU_GML).resolve()


def _validate_plateau_gml(gml_path: Path) -> Path:
    path = Path(gml_path).expanduser().resolve()
    try:
        exists = path.exists()
        is_file = path.is_file()
        size_bytes = path.stat().st_size if is_file else 0
    except OSError as exc:
        raise PlateauPresetError(
            f"PLATEAU mesh could not be read: {path} ({exc})\n"
            "Fix: check file permissions and rerun with `--gml PATH`."
        ) from exc
    if not exists:
        raise PlateauPresetError(
            f"PLATEAU mesh was not found: {path}\n"
            "Fix: restore data/sample_plateau.gml from the checkout, or fetch a local PLATEAU subset with "
            "`python experiments/fetch_plateau_subset.py --help`, then rerun the preset."
        )
    if not is_file or size_bytes == 0:
        raise PlateauPresetError(
            f"PLATEAU mesh is not a non-empty file: {path}\n"
            "Fix: provide a readable CityGML file with `gnss-gpu run --preset plateau-nlos --gml PATH`."
        )
    return path


def _load_experiment_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PlateauPresetError(f"could not load PLATEAU suite module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_plateau_nlos(
    *,
    gml_path: Path,
    output_dir: Path,
    pf_particles: int = 3000,
    allow_cpu_fallback: bool = False,
) -> dict[str, object]:
    """Run the checked-in PLATEAU mask + SPP/PF/FGO suite as one preset."""

    gml_path = _validate_plateau_gml(gml_path)
    if pf_particles <= 0:
        raise PlateauPresetError("--pf-particles must be positive")
    if not allow_cpu_fallback:
        try:
            importlib.import_module("gnss_gpu._bvh")
        except (ImportError, OSError) as exc:
            raise PlateauPresetError(
                "the CUDA BVH extension is not available for the GPU PLATEAU preset "
                f"({exc})\n"
                "Fix: run `gnss-gpu doctor`, then `gnss-gpu build`, and rerun. "
                "Use `--allow-cpu-fallback` only for a CPU smoke test."
            ) from exc
    suite_path = _repo_root(gml_path) / "experiments" / "run_plateau_nlos_demo_suite.py"
    if not suite_path.is_file():
        suite_path = Path(__file__).resolve().parents[2] / "experiments" / "run_plateau_nlos_demo_suite.py"
    suite_module = _load_experiment_module("gnss_gpu_cli_plateau_suite", suite_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mask_csv": output_dir / "plateau_nlos_demo_mask.csv",
        "mask_summary": output_dir / "plateau_nlos_demo_mask_summary.json",
        "spp_summary": output_dir / "plateau_nlos_demo_spp_replay_summary.json",
        "pf_summary": output_dir / "plateau_nlos_demo_pf_replay_summary.json",
        "fgo_summary": output_dir / "plateau_nlos_demo_fgo_replay_summary.json",
        "suite_json": output_dir / "plateau_nlos_demo_suite_summary.json",
        "suite_markdown": output_dir / "plateau_nlos_demo_suite_summary.md",
        "suite_csv": output_dir / "plateau_nlos_demo_suite_summary.csv",
    }
    started = time.perf_counter()
    try:
        suite_result = suite_module.run_suite(
            mask_csv=paths["mask_csv"],
            mask_summary_json=paths["mask_summary"],
            spp_summary_json=paths["spp_summary"],
            pf_summary_json=paths["pf_summary"],
            fgo_summary_json=paths["fgo_summary"],
            suite_json=paths["suite_json"],
            suite_md=paths["suite_markdown"],
            suite_csv=paths["suite_csv"],
            pf_particles=pf_particles,
            gml_path=gml_path,
        )
    except FileNotFoundError as exc:
        raise PlateauPresetError(
            f"PLATEAU suite input is missing: {exc}\n"
            "Fix: check the CityGML path and rerun with `--gml PATH`; the checked-in sample is "
            "`data/sample_plateau.gml`."
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise PlateauPresetError(
            f"PLATEAU NLOS suite failed: {exc}\n"
            "Fix: run `gnss-gpu doctor`, confirm the CUDA BVH extension is built, and verify the CityGML file."
        ) from exc
    except Exception as exc:
        # XML parsers and optional native bindings use several exception
        # classes across supported Python/CUDA versions.  Keep malformed data
        # and broken runtime dependencies user-facing instead of leaking a
        # traceback from the CLI.
        raise PlateauPresetError(
            f"PLATEAU NLOS suite failed: {exc}\n"
            "Fix: verify the CityGML file is readable, then run `gnss-gpu doctor` and rebuild the CUDA extension."
        ) from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    suite = suite_result.get("suite") if isinstance(suite_result, Mapping) else None
    mask = suite_result.get("mask") if isinstance(suite_result, Mapping) else None
    ray_source = str(mask.get("ray_source", "")) if isinstance(mask, Mapping) else ""
    gpu_backend = ray_source.lower() == "native bvh"
    if not gpu_backend and not allow_cpu_fallback:
        raise PlateauPresetError(
            f"PLATEAU preset used `{ray_source or 'unknown ray source'}`, not the CUDA BVH backend.\n"
            "Fix: run `gnss-gpu doctor`, then `gnss-gpu build` and rerun. "
            "Use `--allow-cpu-fallback` only for a CPU smoke test."
        )
    result: dict[str, object] = {
        "preset": "plateau-nlos",
        "backend": "CUDA" if gpu_backend else "CPU",
        "ray_source": ray_source,
        "gml_path": str(gml_path),
        "elapsed_ms": elapsed_ms,
        "suite": suite_result.get("suite", {}) if isinstance(suite_result, Mapping) else {},
        "mask": mask if isinstance(mask, Mapping) else {},
        "metrics": {
            "runtime_ms": elapsed_ms,
            "gpu_raytrace": gpu_backend,
        },
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    if isinstance(suite, Mapping):
        result["n_epochs"] = suite.get("mask", {}).get("epochs") if isinstance(suite.get("mask"), Mapping) else None
    return result


def _schema_version_supported(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == float(RUN_MANIFEST_SCHEMA_VERSION)
    normalized = str(value).strip().lower()
    return normalized in {
        str(RUN_MANIFEST_SCHEMA_VERSION),
        f"{RUN_MANIFEST_SCHEMA_VERSION}.0",
        RUN_MANIFEST_SCHEMA,
    }


def _normalise_input_hashes(raw: Mapping[str, object]) -> dict[str, str]:
    hashes = raw.get("input_hashes", raw.get("input_sha256"))
    if (
        ("input_hashes" in raw or "input_sha256" in raw)
        and hashes is not None
        and not isinstance(hashes, Mapping)
    ):
        raise ManifestError("input_hashes must be an object")
    if isinstance(hashes, Mapping):
        result: dict[str, str] = {}
        for key, value in hashes.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value:
                raise ManifestError("input_hashes must map string paths to non-empty hash strings")
            result[key] = value
        return result
    inputs = raw.get("inputs")
    if "inputs" in raw and inputs is not None and not isinstance(inputs, list):
        raise ManifestError("inputs must be a list of objects")
    if isinstance(inputs, list):
        result = {}
        for item in inputs:
            if not isinstance(item, Mapping):
                raise ManifestError("each inputs entry must be an object")
            path = item.get("path")
            digest = item.get("sha256")
            if isinstance(path, str) and isinstance(digest, str) and digest:
                result[path] = digest
        return result
    return {}


def _normalise_metrics(raw: Mapping[str, object]) -> dict[str, float | int | bool]:
    metrics = raw.get("metrics")
    if metrics is None:
        metrics = {}
        # Manifests emitted by the first GPU-first CLI were flat.  Promote
        # their useful fields into the common metrics namespace.
        for source, target in (
            ("elapsed_ms", "runtime_ms"),
            ("runtime_ms", "runtime_ms"),
            ("rms_m", "rms_m"),
            ("baseline_rms_m", "baseline_rms_m"),
            ("acquired", "acquired"),
        ):
            if source in raw:
                metrics[target] = raw[source]
    if not isinstance(metrics, Mapping):
        raise ManifestError("manifest metrics must be an object of scalar values")
    normalized: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise ManifestError("manifest metric names must be strings")
        if isinstance(value, bool):
            normalized[key] = value
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ManifestError(f"manifest metric {key!r} must be a finite number or boolean")
        normalized[key] = value
    return normalized


def load_run_manifest(path: Path | str) -> dict[str, object]:
    """Load and validate a run manifest, accepting the pre-v1 flat format."""

    requested = Path(path).expanduser()
    manifest_path = requested / "manifest.json" if requested.is_dir() else requested
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest file does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read valid JSON manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(f"manifest root must be a JSON object: {manifest_path}")
    raw = dict(payload)

    schema = raw.get("schema")
    schema_version = raw.get("schema_version")
    legacy = False
    if schema is None:
        if not _schema_version_supported(schema_version):
            raise ManifestError(f"unsupported manifest schema version {schema_version!r}")
        # No schema plus the old flat fields is the format written by v0.3.0.
        if "backend" not in raw and "preset" not in raw:
            raise ManifestError("manifest is missing schema and preset/backend fields")
        legacy = True
    elif schema == RUN_MANIFEST_SCHEMA and _schema_version_supported(schema_version):
        pass
    else:
        raise ManifestError(
            f"unsupported manifest schema {schema!r} (expected {RUN_MANIFEST_SCHEMA!r})"
        )

    backend = raw.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ManifestError("manifest backend must be a non-empty string")
    metrics = _normalise_metrics(raw)
    if not metrics:
        raise ManifestError("manifest contains no comparable metrics")
    parameters = raw.get("parameters")
    if parameters is None:
        parameters = {"preset": raw.get("preset", "unknown")}
        legacy = True
    if not isinstance(parameters, Mapping):
        raise ManifestError("manifest parameters must be an object")
    artifacts = raw.get("artifacts", {})
    if not isinstance(artifacts, (Mapping, list)):
        raise ManifestError("manifest artifacts must be an object or list")
    preset = raw.get("preset")
    if not isinstance(preset, str):
        preset = parameters.get("preset", "unknown") if isinstance(parameters, Mapping) else "unknown"
    if not isinstance(preset, str):
        preset = str(preset)
    gpu = raw.get("gpu", raw.get("gpu_info", {}))
    if gpu is None:
        gpu = {}
    if not isinstance(gpu, Mapping):
        raise ManifestError("manifest gpu information must be an object")

    normalized = dict(raw)
    normalized.update(
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "preset": preset,
            "backend": backend,
            "gpu": dict(gpu),
            "gpu_info": dict(gpu),
            "input_hashes": _normalise_input_hashes(raw),
            "parameters": dict(parameters),
            "metrics": metrics,
            "artifacts": artifacts,
            "_manifest_path": str(manifest_path),
            "_legacy": legacy,
        }
    )
    return normalized


def _numeric_metrics(manifest: Mapping[str, object]) -> dict[str, float]:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ManifestError("manifest metrics must be an object")
    values: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            values[str(key)] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[str(key)] = float(value)
    if not values:
        raise ManifestError("manifest contains no finite numeric metrics")
    return values


def _metric_direction(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("gain", "win", "acquired")):
        return "higher"
    if any(token in lowered for token in ("rms", "p50", "mean", "max", "error", "runtime", "elapsed", "latency", "time")):
        return "lower"
    return "unknown"


def compare_manifests(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Compare two normalized manifests and return a JSON-safe report."""

    baseline_schema = baseline.get("schema")
    candidate_schema = candidate.get("schema")
    if baseline_schema != RUN_MANIFEST_SCHEMA or candidate_schema != RUN_MANIFEST_SCHEMA:
        raise ManifestError("baseline and candidate manifest schemas are incompatible")
    baseline_preset = str(baseline.get("preset", "unknown"))
    candidate_preset = str(candidate.get("preset", "unknown"))
    if baseline_preset != candidate_preset and baseline_preset != "unknown" and candidate_preset != "unknown":
        raise ManifestError(
            f"cannot compare different presets: baseline={baseline_preset!r}, candidate={candidate_preset!r}"
        )
    baseline_metrics = _numeric_metrics(baseline)
    candidate_metrics = _numeric_metrics(candidate)
    warnings: list[str] = []
    if baseline.get("backend") != candidate.get("backend"):
        warnings.append(
            f"backend differs ({baseline.get('backend')} -> {candidate.get('backend')}); speed deltas may not be comparable"
        )
    if baseline.get("input_hashes", {}) != candidate.get("input_hashes", {}):
        warnings.append("input hashes differ; accuracy deltas use different input data")
    if baseline.get("_legacy") or candidate.get("_legacy"):
        warnings.append("one or both manifests use the legacy flat format")

    metric_report: dict[str, object] = {}
    for name in sorted(set(baseline_metrics) | set(candidate_metrics)):
        base = baseline_metrics.get(name)
        cand = candidate_metrics.get(name)
        if base is None or cand is None:
            metric_report[name] = {
                "baseline": base,
                "candidate": cand,
                "delta": None,
                "percent_change": None,
                "improvement_pct": None,
                "direction": _metric_direction(name),
                "status": "missing",
            }
            continue
        delta = cand - base
        percent_change = None if base == 0.0 else 100.0 * delta / abs(base)
        direction = _metric_direction(name)
        if direction == "lower":
            improvement = None if base == 0.0 else 100.0 * (base - cand) / abs(base)
        elif direction == "higher":
            improvement = None if base == 0.0 else 100.0 * (cand - base) / abs(base)
        else:
            improvement = None
        if delta == 0.0:
            status = "same"
        elif direction == "lower":
            status = "improved" if delta < 0.0 else "regressed"
        elif direction == "higher":
            status = "improved" if delta > 0.0 else "regressed"
        else:
            status = "changed"
        metric_report[name] = {
            "baseline": base,
            "candidate": cand,
            "delta": delta,
            "percent_change": percent_change,
            "improvement_pct": improvement,
            "direction": direction,
            "status": status,
        }
    return {
        "schema": RUN_COMPARISON_SCHEMA,
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "compatible": True,
        "baseline": {
            "path": baseline.get("_manifest_path"),
            "preset": baseline_preset,
            "backend": baseline.get("backend"),
            "git_sha": baseline.get("git_sha", baseline.get("git", {}).get("sha") if isinstance(baseline.get("git"), Mapping) else None),
        },
        "candidate": {
            "path": candidate.get("_manifest_path"),
            "preset": candidate_preset,
            "backend": candidate.get("backend"),
            "git_sha": candidate.get("git_sha", candidate.get("git", {}).get("sha") if isinstance(candidate.get("git"), Mapping) else None),
        },
        "input_hashes_match": baseline.get("input_hashes", {}) == candidate.get("input_hashes", {}),
        "warnings": warnings,
        "metrics": metric_report,
        "metric_deltas": {
            name: item.get("delta")
            for name, item in metric_report.items()
            if isinstance(item, Mapping)
        },
        "metric_percent_changes": {
            name: item.get("percent_change")
            for name, item in metric_report.items()
            if isinstance(item, Mapping)
        },
    }


def _format_compare_console(report: Mapping[str, object]) -> str:
    lines = [
        "gnss_gpu run comparison",
        "=" * 72,
        f"Preset: {report['baseline']['preset']}",  # type: ignore[index]
        f"Backend: {report['baseline']['backend']} -> {report['candidate']['backend']}",  # type: ignore[index]
        f"Input hashes: {'MATCH' if report['input_hashes_match'] else 'DIFFER'}",
        "",
        f"{'metric':<32}{'baseline':>14}{'candidate':>14}{'delta':>14}{'status':>12}",
        "-" * 86,
    ]
    metrics = report.get("metrics", {})
    if isinstance(metrics, Mapping):
        for name, raw in metrics.items():
            item = raw if isinstance(raw, Mapping) else {}
            def _display(value: object) -> str:
                if value is None:
                    return "-"
                if isinstance(value, float):
                    return f"{value:.4g}"
                return str(value)
            lines.append(
                f"{str(name):<32}{_display(item.get('baseline')):>14}"
                f"{_display(item.get('candidate')):>14}{_display(item.get('delta')):>14}"
                f"{str(item.get('status', '')):>12}"
            )
    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _format_compare_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# gnss_gpu run comparison",
        "",
        f"- Preset: `{report['baseline']['preset']}`",  # type: ignore[index]
        f"- Backend: `{report['baseline']['backend']}` → `{report['candidate']['backend']}`",  # type: ignore[index]
        f"- Input hashes: **{'match' if report['input_hashes_match'] else 'differ'}**",
        "",
        "| Metric | Baseline | Candidate | Delta | Improvement | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    metrics = report.get("metrics", {})
    if isinstance(metrics, Mapping):
        for name, raw in metrics.items():
            item = raw if isinstance(raw, Mapping) else {}
            def _display(value: object) -> str:
                if value is None:
                    return "-"
                if isinstance(value, float):
                    return f"{value:.4g}"
                return str(value)
            improvement = item.get("improvement_pct")
            improvement_text = "-" if improvement is None else f"{float(improvement):+.2f}%"
            lines.append(
                f"| `{name}` | {_display(item.get('baseline'))} | {_display(item.get('candidate'))} | "
                f"{_display(item.get('delta'))} | {improvement_text} | {item.get('status', '')} |"
            )
    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


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
        Check("Platform", "PASS", _safe_platform_info()),
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


def _cmd_data_inspect(args: argparse.Namespace) -> int:
    try:
        result = inspect_input(args.input)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Data inspect error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        try:
            output = Path(args.json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            print(f"Data inspect error: could not write {args.json}: {exc}", file=sys.stderr)
            return 2
    print(format_inspection(result))
    if args.json:
        print(f"Report:    {Path(args.json).expanduser().resolve()}")
    return 0 if result.ready else 1


def _cmd_run(args: argparse.Namespace) -> int:
    root = _repo_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or Path("runs") / timestamp).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    input_paths: list[Path] = []
    artifact_paths: dict[str, Path] = {}
    if args.preset == "signal-acquisition":
        try:
            result = _gpu_roundtrip()
        except Exception as exc:
            print(f"GPU preset failed: {exc}", file=sys.stderr)
            print("Run `gnss-gpu doctor` to diagnose the environment.", file=sys.stderr)
            return 1
        parameters: dict[str, object] = {"preset": args.preset}
    elif args.preset == "plateau-nlos":
        try:
            requested_gml = getattr(args, "gml", None)
            gml_path = _validate_plateau_gml(requested_gml or _default_plateau_gml(root))
            result = _run_plateau_nlos(
                gml_path=gml_path,
                output_dir=output_dir,
                pf_particles=getattr(args, "pf_particles", 3000),
                allow_cpu_fallback=getattr(args, "allow_cpu_fallback", False),
            )
        except PlateauPresetError as exc:
            print(f"GPU preset failed: {exc}", file=sys.stderr)
            return 1
        input_paths.append(gml_path)
        raw_artifacts = result.get("artifact_paths") if isinstance(result, Mapping) else None
        if isinstance(raw_artifacts, Mapping):
            artifact_paths = {str(key): Path(str(value)) for key, value in raw_artifacts.items()}
        parameters = {
            "preset": args.preset,
            "gml_path": _path_label(gml_path, root),
            "pf_particles": getattr(args, "pf_particles", 3000),
            "allow_cpu_fallback": bool(getattr(args, "allow_cpu_fallback", False)),
        }
    elif args.preset == "urbannav-pf":
        input_path = getattr(args, "input", None)
        if input_path is None:
            print(
                "UrbanNav PF preset requires --input PATH. "
                "Run gnss-gpu data inspect PATH first.",
                file=sys.stderr,
            )
            return 2
        raw_systems = getattr(args, "systems", "G")
        systems = tuple(
            part.strip().upper()
            for part in str(raw_systems).split(",")
            if part.strip()
        )
        if not systems:
            print("UrbanNav GPU preset requires at least one GNSS system in --systems.", file=sys.stderr)
            return 2
        try:
            result = run_urbannav_pf(
                input_path,
                output_dir,
                particles=getattr(args, "particles", 10000),
                max_epochs=getattr(args, "max_epochs", 300),
                start_epoch=getattr(args, "start_epoch", 0),
                systems=systems,
                rover_source=getattr(args, "urban_rover", "ublox"),
                seed=getattr(args, "seed", 42),
                no_plots=bool(getattr(args, "no_plots", False)),
            )
        except UrbanNavRunError as exc:
            print(f"UrbanNav GPU preset failed: {exc}", file=sys.stderr)
            print(
                "No CPU fallback was used. Run gnss-gpu data inspect PATH, "
                "then gnss-gpu doctor if the CUDA runtime is unavailable.",
                file=sys.stderr,
            )
            return 1
        input_paths.extend(Path(path) for path in result.get("input_paths", ()))
        raw_artifacts = result.get("artifact_paths")
        if isinstance(raw_artifacts, Mapping):
            artifact_paths = {
                str(key): Path(str(value))
                for key, value in raw_artifacts.items()
            }
        parameters = dict(result.get("parameters", {}))
        parameters.setdefault("preset", args.preset)
    else:
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        return 2
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if "elapsed_ms" not in result:
        result["elapsed_ms"] = elapsed_ms
    command = ["gnss-gpu", "run", "--preset", args.preset]
    if args.preset == "plateau-nlos":
        command.extend(["--gml", str(gml_path), "--pf-particles", str(getattr(args, "pf_particles", 3000))])
        if getattr(args, "allow_cpu_fallback", False):
            command.append("--allow-cpu-fallback")
    elif args.preset == "urbannav-pf":
        command.extend(
            [
                "--input",
                str(args.input),
                "--particles",
                str(getattr(args, "particles", 10000)),
                "--max-epochs",
                str(getattr(args, "max_epochs", 300)),
                "--start-epoch",
                str(getattr(args, "start_epoch", 0)),
                "--systems",
                str(getattr(args, "systems", "G")),
                "--urban-rover",
                str(getattr(args, "urban_rover", "ublox")),
                "--seed",
                str(getattr(args, "seed", 42)),
            ]
        )
        if getattr(args, "no_plots", False):
            command.append("--no-plots")
    manifest = build_run_manifest(
        preset=args.preset,
        result=result,
        parameters=parameters,
        input_paths=input_paths,
        artifact_paths=artifact_paths,
        repo_root=root,
        command=command,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.preset == "signal-acquisition":
        print("GPU signal simulation -> acquisition: PASS")
        print(f"Backend:     {result['backend']}")
        print(f"PRN:         {result['prn']} acquired")
        print(f"Doppler:     {result['doppler_hz']:.1f} Hz")
        print(f"Runtime:     {result['elapsed_ms']:.2f} ms")
        print(f"Manifest:    {manifest_path}")
        print("Next:        gnss-gpu run --preset plateau-nlos")
    elif args.preset == "plateau-nlos":
        suite = result.get("suite", {})
        print("PLATEAU NLOS GPU demo suite: PASS")
        print(f"Backend:     {result.get('backend', 'CUDA')}")
        print(f"Ray source:  {result.get('ray_source', 'unknown')}")
        if isinstance(suite, Mapping):
            print(
                f"NLOS:        {float(suite.get('mask', {}).get('nlos_frac', 0.0)) * 100.0:.1f}%"
                if isinstance(suite.get("mask"), Mapping)
                else "NLOS:        unavailable"
            )
            if suite.get("best_mask_soft_estimator") is not None:
                print(
                    f"Best RMS:    {suite['best_mask_soft_estimator']} "
                    f"{float(suite['best_mask_soft_rms_m']):.2f} m"
                )
        print(f"Runtime:     {float(result.get('elapsed_ms', elapsed_ms)):.2f} ms")
        print(f"Manifest:    {manifest_path}")
        print("Next:        gnss-gpu run --preset plateau-nlos --output-dir runs/plateau-nlos-candidate")
    elif args.preset == "urbannav-pf":
        metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
        print("UrbanNav GPU particle-filter run: PASS")
        print(f"Backend:     {result.get('backend', 'CUDA')}")
        print(f"Dataset:     {result.get('dataset_name', 'unknown')}")
        print(f"Epochs:      {result.get('n_epochs', 0)}")
        if isinstance(metrics, Mapping):
            if metrics.get("pf_rms_2d_m") is not None:
                print(f"PF RMS 2D:   {float(metrics['pf_rms_2d_m']):.3f} m")
            if metrics.get("wls_rms_2d_m") is not None:
                print(f"WLS RMS 2D:  {float(metrics['wls_rms_2d_m']):.3f} m")
        print(f"Runtime:     {float(result.get('elapsed_ms', elapsed_ms)):.2f} ms")
        print(f"Manifest:    {manifest_path}")
        print(
            "Next:        gnss-gpu run --preset urbannav-pf --input "
            f"{args.input} --output-dir runs/urbannav-pf-candidate"
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        baseline = load_run_manifest(args.baseline)
        candidate = load_run_manifest(args.candidate)
        report = compare_manifests(baseline, candidate)
    except (ManifestError, OSError, TypeError, ValueError) as exc:
        print(f"Compare error: {exc}", file=sys.stderr)
        return 2

    output_path: Path | None = Path(args.output).expanduser().resolve() if args.output else None
    output_format = args.format
    if args.json_output and args.markdown_output:
        print("Compare error: use only one of --json and --markdown", file=sys.stderr)
        return 2
    if args.json_output:
        if output_path is not None:
            print("Compare error: use only one of --output and --json", file=sys.stderr)
            return 2
        output_format = "json"
        output_path = Path(args.json_output).expanduser().resolve()
    if args.markdown_output:
        if output_path is not None:
            print("Compare error: use only one of --output and --markdown", file=sys.stderr)
            return 2
        output_format = "markdown"
        output_path = Path(args.markdown_output).expanduser().resolve()
    if output_path is None:
        candidate_manifest = Path(str(candidate["_manifest_path"]))
        output_path = candidate_manifest.parent / ("comparison.json" if output_format == "json" else "comparison.md")

    baseline_manifest_path = Path(str(baseline["_manifest_path"])).resolve()
    candidate_manifest_path = Path(str(candidate["_manifest_path"])).resolve()
    if output_path.resolve() in (baseline_manifest_path, candidate_manifest_path):
        print("Compare error: report output must not overwrite a manifest", file=sys.stderr)
        return 2

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        elif output_format == "markdown":
            output_path.write_text(_format_compare_markdown(report), encoding="utf-8")
        else:
            raise ManifestError(f"unsupported comparison output format: {output_format}")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Compare error: could not write report {output_path}: {exc}", file=sys.stderr)
        return 2

    print(_format_compare_console(report))
    print(f"Report: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnss-gpu", description="GPU-first setup and demo runner for gnss_gpu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="inspect local GNSS/UrbanNav inputs")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    inspect = data_subparsers.add_parser(
        "inspect",
        help="detect a local RINEX/UrbanNav/PPC bundle and validate its data contract",
    )
    inspect.add_argument("input", type=Path, help="run directory, dataset root, or RINEX file")
    inspect.add_argument("--json", metavar="PATH", help="also write a machine-readable inspection report")
    inspect.set_defaults(handler=_cmd_data_inspect)

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
    run.add_argument(
        "--preset",
        default="signal-acquisition",
        choices=("signal-acquisition", "plateau-nlos", "urbannav-pf"),
    )
    run.add_argument("--output-dir", type=Path, help="directory for manifest.json (default: runs/<UTC timestamp>)")
    run.add_argument(
        "--input",
        type=Path,
        help="UrbanNav/PPC run directory or dataset root for the urbannav-pf preset",
    )
    run.add_argument(
        "--gml",
        type=Path,
        help="PLATEAU CityGML input for plateau-nlos (default: data/sample_plateau.gml)",
    )
    run.add_argument(
        "--pf-particles",
        type=int,
        default=3000,
        help="particle count for the plateau-nlos replay (default: 3000)",
    )
    run.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="allow CPU triangle ray-cast for plateau-nlos smoke tests",
    )
    run.add_argument(
        "--particles",
        "--urbannav-particles",
        dest="particles",
        type=int,
        default=10000,
        help="particle count for the urbannav-pf preset (default: 10000)",
    )
    run.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="maximum usable epochs for the urbannav-pf preset (default: 300)",
    )
    run.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="skip this many usable epochs before the urbannav-pf run",
    )
    run.add_argument(
        "--systems",
        type=str,
        default="G",
        help="comma-separated GNSS systems for urbannav-pf, e.g. G or G,E,J",
    )
    run.add_argument(
        "--urban-rover",
        type=str,
        default="ublox",
        help="UrbanNav rover source: ublox or trimble (default: ublox)",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="reproducible PF seed for urbannav-pf (default: 42)",
    )
    run.add_argument(
        "--no-plots",
        action="store_true",
        help="write a placeholder visualization instead of the error timeline",
    )
    run.set_defaults(handler=_cmd_run)

    compare = subparsers.add_parser(
        "compare",
        help="compare two run manifests and write a Markdown or JSON report",
    )
    compare.add_argument("baseline", help="baseline run directory or manifest.json")
    compare.add_argument("candidate", help="candidate run directory or manifest.json")
    compare.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="report format (default: markdown)",
    )
    compare.add_argument("--output", "-o", type=Path, help="report output path")
    compare.add_argument("--json", dest="json_output", type=Path, help="write a JSON report")
    compare.add_argument("--markdown", dest="markdown_output", type=Path, help="write a Markdown report")
    compare.set_defaults(handler=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
