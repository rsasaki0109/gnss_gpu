"""User-facing UrbanNav/PPC input inspection and GPU PF runner.

The command-line onboarding path deliberately lives outside the experiment
scripts.  The experiment scripts remain useful for research sweeps, while
this module provides a small, deterministic contract for:

* checking a local UrbanNav/PPC/RINEX bundle before starting a long run;
* running the existing UrbanNavLoader data contract through the CUDA
  particle filter; and
* writing artifacts that are consumable by the shared run-manifest/compare
  implementation in gnss_gpu.cli.

No function in this module downloads data.  Missing data is reported with a
concrete command for the repository's existing fetch helper.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gnss_gpu.io.nav_rinex import read_nav_rinex
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.io.urbannav import UrbanNavLoader


URBANNAV_PRESET = "urbannav-pf"
URBANNAV_REQUIRED_ROLES = (
    "rover observation",
    "base observation",
    "navigation",
    "ground truth",
)
_OBS_SUFFIXES = {".obs", ".rnx", ".o"}
_NAV_SUFFIXES = {".nav", ".rnx", ".n", ".g", ".p", ".l"}
_TIME_ALIASES = {
    "gps t ow (s)",
    "gps tow (s)",
    "gps tow",
    "gps_time",
    "gps time",
    "gps_tow",
    "time",
    "timestamp",
}
_ECEF_ALIASES = (
    {"ecef x (m)", "ecef x", "ecef_x", "x", "pos_x"},
    {"ecef y (m)", "ecef y", "ecef_y", "y", "pos_y"},
    {"ecef z (m)", "ecef z", "ecef_z", "z", "pos_z"},
)
_LLH_ALIASES = (
    {"latitude (deg)", "latitude", "lat"},
    {"longitude (deg)", "longitude", "lon"},
)


def _is_rinex_obs_path(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".rnx" and "nav" in path.name.lower() and "obs" not in path.name.lower():
        return False
    return (
        suffix in _OBS_SUFFIXES
        or (len(suffix) == 4 and suffix[1:3].isdigit() and suffix.endswith("o"))
    )


def _is_rinex_nav_path(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".rnx" and "obs" in path.name.lower() and "nav" not in path.name.lower():
        return False
    return (
        suffix in _NAV_SUFFIXES
        or (len(suffix) == 4 and suffix[1:3].isdigit() and suffix.endswith("n"))
    )


class InputInspectionError(ValueError):
    """Raised when an input cannot be inspected safely."""


class UrbanNavRunError(RuntimeError):
    """Raised for a user-actionable UrbanNav GPU run failure."""


@dataclass
class InputInspection:
    """Structured result returned by inspect_input.

    files maps the stable role names used in the user-facing output to
    discovered paths.  Paths are absolute so callers can safely pass the
    result to a runner; as_dict additionally provides JSON-safe strings.
    """

    input_path: Path
    resolved_path: Path
    detected_format: str
    status: str
    run_dir: Path | None = None
    files: dict[str, Path | None] = field(default_factory=dict)
    required_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    optional_files: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_presets: list[str] = field(default_factory=list)
    repair_commands: list[str] = field(default_factory=list)
    candidates: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    @property
    def format(self) -> str:
        """Short alias useful to callers that treat this as a data report."""

        return self.detected_format

    def as_dict(self) -> dict[str, Any]:
        def path_value(value: Path | None) -> str | None:
            return str(value) if value is not None else None

        return {
            "input": str(self.input_path),
            "resolved": str(self.resolved_path),
            "detected_format": self.detected_format,
            "status": self.status,
            "run_dir": path_value(self.run_dir),
            "files": {key: path_value(value) for key, value in self.files.items()},
            "required_files": list(self.required_files),
            "missing_files": list(self.missing_files),
            "optional_files": list(self.optional_files),
            "missing_fields": list(self.missing_fields),
            "findings": list(self.findings),
            "warnings": list(self.warnings),
            "suggested_presets": list(self.suggested_presets),
            "repair_commands": list(self.repair_commands),
            "candidates": [str(path) for path in self.candidates],
            "metadata": _json_safe(self.metadata),
        }


def _json_safe(value: Any) -> Any:
    """Convert NumPy/path values to JSON-safe values without importing CLI."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalise_header(header: Sequence[str]) -> set[str]:
    return {str(item).strip().lower().replace("  ", " ") for item in header if item is not None}


def _find_first(directory: Path, patterns: Sequence[str]) -> Path | None:
    """Find the first case-insensitive matching file in a directory."""

    files = [path for path in directory.iterdir() if path.is_file()]
    for pattern in patterns:
        pattern_lower = pattern.lower()
        exact = sorted(path for path in files if fnmatch.fnmatchcase(path.name.lower(), pattern_lower))
        if exact:
            return exact[0]
    return None


def _discover_files(directory: Path) -> dict[str, Path | None]:
    """Discover the exact file roles understood by the existing loaders."""

    rover = _find_first(
        directory,
        (
            "rover_ublox.obs",
            "rover_trimble.obs",
            "rover.obs",
            "*ublox*.obs",
            "*trimble*.obs",
            "rover*.obs",
            "*rover*.??o",
            "*ublox*.??o",
            "*trimble*.??o",
            "rover*.??o",
            "*.obs",
            "*.OBS",
            "*.??o",
        ),
    )
    base = _find_first(
        directory,
        (
            "base.obs",
            "base_trimble.obs",
            "base_ublox.obs",
            "base*.obs",
            "base*.??o",
            "*.base.obs",
            "*.base.??o",
        ),
    )
    # A broad *.obs fallback can select a rover file as base.  Keep the role
    # missing in that case so inspect can explain the ambiguity.
    if base is not None and rover is not None and base.resolve() == rover.resolve():
        base = None
    nav = _find_first(
        directory,
        (
            "base.nav",
            "*.nav",
            "*.NAV",
            "*_nav.rnx",
            "base*.??n",
            "*.??n",
            "*.n",
            "*.N",
        ),
    )
    if nav is not None and any(
        nav.resolve() == candidate.resolve()
        for candidate in (rover, base)
        if candidate is not None
    ):
        nav = None
    reference = _find_first(
        directory,
        (
            "reference.csv",
            "*groundtruth*.csv",
            "*ground_truth*.csv",
            "*gt*.csv",
        ),
    )
    imu = _find_first(directory, ("imu.csv", "*imu*.csv"))
    return {
        "rover observation": rover,
        "base observation": base,
        "navigation": nav,
        "ground truth": reference,
        "imu": imu,
    }


def _looks_like_ppc(directory: Path, files: Mapping[str, Path | None]) -> bool:
    return all((directory / name).is_file() for name in PPCDatasetLoader.REQUIRED_FILES)


def _looks_like_urbannav(directory: Path, files: Mapping[str, Path | None]) -> bool:
    return all(files.get(role) is not None for role in URBANNAV_REQUIRED_ROLES)


def _candidate_run_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    candidates: list[Path] = []
    try:
        if _looks_like_urbannav(path, _discover_files(path)):
            candidates.append(path)
        for child in sorted(path.rglob("*")):
            if not child.is_dir() or child == path:
                continue
            if _looks_like_urbannav(child, _discover_files(child)):
                candidates.append(child)
    except OSError:
        return candidates
    # Preserve deterministic order and avoid nested duplicate discoveries.
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _resolve_input_path(path: str | Path) -> tuple[Path, Path, list[Path]]:
    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve()
    except OSError as exc:
        raise InputInspectionError(f"input path could not be resolved: {requested} ({exc})") from exc
    if resolved.is_dir():
        candidates = _candidate_run_dirs(resolved)
        if candidates:
            # A direct run directory is unambiguous.  For a dataset root with
            # multiple runs, inspect reports all candidates and the runner
            # requires an explicit run path.
            direct = candidates[0] if candidates[0] == resolved else None
            return resolved, direct or candidates[0], candidates
        return resolved, resolved, []
    if resolved.is_file():
        return resolved, resolved.parent, []
    return resolved, resolved.parent, []


def _csv_contract(path: Path) -> tuple[list[str], int, list[str], list[str]]:
    """Validate the reference CSV contract used by both loaders."""

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], 0, [f"reference.csv could not be read: {exc}"], []

    normalised = _normalise_header(header)
    missing: list[str] = []
    findings: list[str] = []
    if not normalised.intersection(_TIME_ALIASES):
        missing.append("reference time (GPS TOW/time/timestamp)")
    ecef_ok = all(normalised.intersection(aliases) for aliases in _ECEF_ALIASES)
    llh_ok = all(normalised.intersection(aliases) for aliases in _LLH_ALIASES)
    if ecef_ok:
        findings.append("reference position: ECEF x/y/z")
    elif llh_ok:
        findings.append("reference position: latitude/longitude (converted by loader)")
    else:
        missing.append("reference position (ECEF x/y/z or latitude/longitude)")
    if not rows:
        missing.append("at least one reference row")
    elif not missing:
        # The loaders convert every row, so a header-only contract check is
        # insufficient: report the first malformed value before a long run.
        time_column = next(
            (column for column in header if column.strip().lower() in _TIME_ALIASES),
            None,
        )
        position_columns = None
        for aliases in _ECEF_ALIASES:
            column = next(
                (candidate for candidate in header if candidate.strip().lower() in aliases),
                None,
            )
            if column is None:
                position_columns = None
                break
            if position_columns is None:
                position_columns = []
            position_columns.append(column)
        if position_columns is None:
            position_columns = [
                next(
                    candidate
                    for candidate in header
                    if candidate.strip().lower() in aliases
                )
                for aliases in _LLH_ALIASES
            ]
        check_columns = ([time_column] if time_column else []) + list(position_columns)
        for row_index, row in enumerate(rows, start=2):
            for column in check_columns:
                try:
                    value = float(row.get(column, "nan"))
                except (TypeError, ValueError):
                    value = float("nan")
                if not np.isfinite(value):
                    missing.append(f"reference.csv row {row_index}: numeric {column}")
                    break
            if missing:
                break
    return header, len(rows), missing, findings


def _obs_contract(
    path: Path,
    *,
    require_approx_position: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Parse a RINEX observation header and report usable fields."""

    try:
        obs = read_rinex_obs(path)
    except Exception as exc:  # parser errors differ between RINEX variants
        return {"path": str(path)}, [f"RINEX observation parse failed: {exc}"], []

    systems = sorted(system for system in obs.header.obs_types if system)
    codes = sorted({code for values in obs.header.obs_types.values() for code in values})
    pseudo_codes = [code for code in codes if code.startswith(("C", "P"))]
    finite_epochs = sum(1 for epoch in obs.epochs if epoch.satellites)
    usable_pseudorange_epochs = sum(
        1
        for epoch in obs.epochs
        if any(
            np.isfinite(float(value)) and float(value) >= 1.0e6
            for observations in epoch.observations.values()
            for code, value in observations.items()
            if code.startswith(("C", "P"))
        )
    )
    findings = [
        f"RINEX {obs.header.version:g} observation file",
        f"epochs: {finite_epochs}",
        f"epochs with pseudorange: {usable_pseudorange_epochs}",
        f"constellations: {','.join(systems) if systems else 'unknown'}",
    ]
    missing: list[str] = []
    if obs.header.version <= 0.0:
        missing.append("RINEX VERSION / TYPE header")
    if not pseudo_codes:
        missing.append("pseudorange observation code (C* or P*)")
    if finite_epochs == 0:
        missing.append("at least one observation epoch")
    if usable_pseudorange_epochs == 0:
        missing.append("at least one usable pseudorange epoch")
    approx = np.asarray(obs.header.approx_position, dtype=np.float64)
    if require_approx_position and (
        not np.all(np.isfinite(approx)) or float(np.linalg.norm(approx)) < 1.0e6
    ):
        missing.append("base APPROX POSITION XYZ (base observation only)")
    return {
        "version": float(obs.header.version),
        "systems": systems,
        "observation_codes": codes,
        "pseudorange_codes": pseudo_codes,
        "epochs": finite_epochs,
        "approx_position": np.asarray(obs.header.approx_position, dtype=float).tolist(),
    }, missing, findings


def _nav_contract(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    try:
        messages = read_nav_rinex(path, systems=("G", "R", "E", "C", "J"))
    except Exception as exc:
        return {"path": str(path)}, [f"RINEX navigation parse failed: {exc}"], []
    count = sum(len(values) for values in messages.values())
    systems = sorted({message.system for values in messages.values() for message in values})
    findings = [
        "RINEX navigation file",
        f"navigation messages: {count}",
        f"constellations: {','.join(systems) if systems else 'unknown'}",
    ]
    missing = [] if count else ["at least one broadcast navigation message"]
    return {"messages": count, "systems": systems}, missing, findings


def _repair_commands(run_dir: Path, missing: Sequence[str], *, format_name: str) -> list[str]:
    commands: list[str] = []
    run_name = run_dir.name
    if run_name not in {"Odaiba", "Shinjuku"}:
        run_name = "Odaiba"
    if any(role in missing for role in ("rover observation", "base observation", "navigation", "ground truth")):
        commands.append(
            f"python experiments/fetch_urbannav_subset.py --run {run_name} --output-dir {run_dir.parent}"
        )
    if "ground truth" in missing:
        commands.append(
            f"copy <groundtruth.csv> {run_dir / 'reference.csv'}  # Windows; use cp on Linux/macOS"
        )
    if "navigation" in missing:
        commands.append(
            f"copy <navigation.nav> {run_dir / 'base.nav'}  # Windows; use cp on Linux/macOS"
        )
    if format_name == "RINEX observation" and not commands:
        commands.append(
            "Place a matching base*.obs, base.nav, and reference.csv beside the observation file"
        )
    return list(dict.fromkeys(commands))


def inspect_input(path: str | Path) -> InputInspection:
    """Inspect a local UrbanNav/PPC/RINEX input without downloading anything."""

    input_path, resolved, candidates = _resolve_input_path(path)
    if not input_path.exists():
        run_dir = resolved if resolved.is_dir() else resolved.parent
        repair = _repair_commands(run_dir, URBANNAV_REQUIRED_ROLES, format_name="unknown")
        return InputInspection(
            input_path=input_path,
            resolved_path=resolved,
            detected_format="missing path",
            status="INVALID",
            run_dir=None,
            required_files=list(URBANNAV_REQUIRED_ROLES),
            missing_files=list(URBANNAV_REQUIRED_ROLES),
            findings=["input path does not exist"],
            repair_commands=repair,
        )

    if input_path.is_file() and not (
        _is_rinex_obs_path(input_path)
        or _is_rinex_nav_path(input_path)
        or input_path.suffix.lower() == ".csv"
    ):
        return InputInspection(
            input_path=input_path,
            resolved_path=resolved,
            detected_format="unknown file",
            status="UNKNOWN",
            findings=[f"unsupported input extension: {input_path.suffix or '(none)'}"],
            repair_commands=[
                "Pass an UrbanNav/PPC run directory or a RINEX .obs/.nav file to gnss-gpu data inspect"
            ],
        )

    run_dir = resolved if resolved.is_dir() else resolved.parent
    files = _discover_files(run_dir) if run_dir.is_dir() else {}
    # When the user points directly at one RINEX file, retain that explicit
    # file if discovery selected another file with the same extension.
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if _is_rinex_obs_path(input_path) and not (
            suffix == ".rnx" and "nav" in input_path.name.lower()
        ):
            name = input_path.name.lower()
            if "base" in name:
                files["base observation"] = input_path
            else:
                files["rover observation"] = input_path
        elif _is_rinex_nav_path(input_path):
            files["navigation"] = input_path
        elif suffix == ".csv" and "reference" in input_path.name.lower():
            files["ground truth"] = input_path

    is_ppc = run_dir.is_dir() and _looks_like_ppc(run_dir, files)
    is_urban = run_dir.is_dir() and _looks_like_urbannav(run_dir, files)
    if is_ppc:
        detected_format = "PPC run (RINEX + reference)"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    elif is_urban:
        detected_format = "UrbanNav run (RINEX + reference)"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    elif run_dir.is_dir() and any(value is not None for value in files.values()):
        detected_format = "RINEX bundle"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    elif input_path.is_file() and _is_rinex_obs_path(input_path):
        detected_format = "RINEX observation"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    elif input_path.is_file() and _is_rinex_nav_path(input_path):
        detected_format = "RINEX navigation"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    elif input_path.is_file() and input_path.suffix.lower() == ".csv":
        detected_format = "reference CSV"
        required = ["rover observation", "base observation", "navigation", "ground truth"]
    else:
        detected_format = "unknown input"
        required = ["rover observation", "base observation", "navigation", "ground truth"]

    missing_files = [role for role in required if files.get(role) is None]
    findings: list[str] = []
    missing_fields: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {"candidate_count": len(candidates)}
    for role in required:
        file_path = files.get(role)
        if file_path is None:
            continue
        try:
            if file_path.stat().st_size <= 0:
                missing_fields.append(f"{role}: non-empty file")
                continue
        except OSError as exc:
            missing_fields.append(f"{role}: readable file ({exc})")
            continue
        if role in {"rover observation", "base observation"}:
            contract, missing, detail = _obs_contract(
                file_path,
                require_approx_position=role == "base observation",
            )
            metadata[role.replace(" ", "_")] = contract
            missing_fields.extend(f"{role}: {item}" for item in missing)
            findings.extend(f"{role}: {item}" for item in detail)
        elif role == "navigation":
            contract, missing, detail = _nav_contract(file_path)
            metadata[role] = contract
            missing_fields.extend(f"navigation: {item}" for item in missing)
            findings.extend(detail)
        elif role == "ground truth":
            _header, count, missing, detail = _csv_contract(file_path)
            metadata["ground_truth"] = {"rows": count}
            missing_fields.extend(missing)
            findings.extend(detail)

    # Optional IMU is useful to the research smoother but is not required by
    # the current undifferenced PF path.  Report it explicitly rather than
    # treating its absence as a broken input.
    if files.get("imu") is None:
        warnings.append("imu.csv not present (optional for urbannav-pf)")
    else:
        metadata["imu"] = {"size_bytes": files["imu"].stat().st_size}

    if candidates and len(candidates) > 1:
        warnings.append("multiple UrbanNav run directories found; pass one run directory explicitly")
    if detected_format == "unknown input":
        status = "UNKNOWN"
    elif missing_files or missing_fields:
        status = "INCOMPLETE" if input_path.exists() else "INVALID"
    else:
        status = "READY"

    suggestions: list[str] = []
    if (
        is_urban
        or is_ppc
        or detected_format in {"RINEX bundle", "RINEX observation", "RINEX navigation", "reference CSV"}
    ):
        suggestions.append(URBANNAV_PRESET)
    repair = _repair_commands(run_dir, missing_files, format_name=detected_format)
    if missing_fields and not repair:
        repair.append(
            "Re-extract the RINEX/reference files from a complete local dataset and rerun this inspection"
        )
    return InputInspection(
        input_path=input_path,
        resolved_path=resolved,
        detected_format=detected_format,
        status=status,
        run_dir=run_dir if run_dir.is_dir() else None,
        files=files,
        required_files=required,
        missing_files=missing_files,
        optional_files=["imu.csv"],
        missing_fields=missing_fields,
        findings=findings,
        warnings=warnings,
        suggested_presets=suggestions,
        repair_commands=repair,
        candidates=candidates,
        metadata=metadata,
    )


def format_inspection(result: InputInspection) -> str:
    """Format an inspection as concise terminal output."""

    expected_names = {
        "rover observation": "rover*.obs",
        "base observation": "base*.obs",
        "navigation": "base.nav / *.nav",
        "ground truth": "reference.csv",
    }
    lines = [
        "gnss_gpu data inspect",
        "=" * 72,
        f"Input:     {result.input_path}",
        f"Detected:  {result.detected_format}",
        f"Status:    {result.status}",
    ]
    if result.run_dir is not None:
        lines.append(f"Run dir:   {result.run_dir}")
    lines.extend(("", "Required files:"))
    for role in result.required_files:
        path = result.files.get(role)
        display_role = f"{role} ({expected_names.get(role, role)})"
        lines.append(
            f"  [{'PASS' if path is not None else 'MISS'}] "
            f"{display_role:<42} {path or '-'}"
        )
    if result.optional_files:
        lines.append("\nOptional files:")
        for role in result.optional_files:
            path = result.files.get(role[:-4] if role.endswith(".csv") else role)
            lines.append(f"  [{'PASS' if path is not None else 'INFO'}] {role:<20} {path or 'not present'}")
    if result.findings:
        lines.append("\nFindings:")
        lines.extend(f"  - {finding}" for finding in result.findings)
    if result.missing_fields:
        lines.append("\nMissing/invalid fields:")
        lines.extend(f"  - {item}" for item in result.missing_fields)
    if result.warnings:
        lines.append("\nNotes:")
        lines.extend(f"  - {item}" for item in result.warnings)
    if result.suggested_presets:
        lines.append(
            "\nSuggested run:" if result.ready else "\nExpected preset after repair:"
        )
        lines.append(
            f"  gnss-gpu run --preset {result.suggested_presets[0]} --input {result.run_dir or result.resolved_path}"
        )
    if result.repair_commands:
        lines.append("\nRepair suggestions (not executed):")
        lines.extend(f"  {command}" for command in result.repair_commands)
    lines.append("\nNo external data was downloaded.")
    return "\n".join(lines)


def _resolve_ready_run(result: InputInspection) -> tuple[Path, str]:
    if not result.ready or result.run_dir is None:
        details = [result.status.lower()]
        if result.missing_files:
            details.append("missing files: " + ", ".join(result.missing_files))
        if result.missing_fields:
            details.append("invalid fields: " + ", ".join(result.missing_fields))
        raise UrbanNavRunError(
            "input is not ready (" + "; ".join(details) + "). "
            "Run gnss-gpu data inspect PATH for repair suggestions."
        )
    if result.candidates and len(result.candidates) > 1 and result.run_dir == result.candidates[0]:
        # A root containing multiple runs is not safe to guess from.  A direct
        # run path has candidates=[itself] and remains accepted.
        if result.resolved_path != result.run_dir:
            listed = ", ".join(str(path) for path in result.candidates[:5])
            raise UrbanNavRunError(
                f"input contains multiple UrbanNav runs ({listed}); pass one run directory with --input"
            )
    format_name = "ppc" if result.detected_format.startswith("PPC") else "urbannav"
    return result.run_dir, format_name


def _finite_wls_solution(solver: Callable[..., Any], sat: Any, pr: Any, weights: Any) -> np.ndarray:
    try:
        raw = solver(sat, pr, weights, 10, 1e-4)
        solution = raw[0] if isinstance(raw, tuple) else raw
        solution = np.asarray(solution, dtype=np.float64).ravel()
    except Exception as exc:
        raise UrbanNavRunError(
            f"CUDA WLS initialization failed: {exc}. "
            "Fix: run gnss-gpu doctor, rebuild the native CUDA extensions, and rerun."
        ) from exc
    if solution.size < 4 or not np.all(np.isfinite(solution[:4])) or np.all(solution[:3] == 0.0):
        raise UrbanNavRunError(
            "CUDA WLS initialization returned no finite position. "
            "Check that the RINEX has at least four usable satellites and matching navigation data."
        )
    return solution[:4]


def _ecef_to_lla(reference: np.ndarray) -> tuple[float, float]:
    x, y, z = reference
    a = 6_378_137.0
    e2 = 6.694379990141316e-3
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2)) if p else math.copysign(math.pi / 2.0, z)
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = p / max(math.cos(lat), 1e-12) - n if abs(math.cos(lat)) > 1e-12 else z / max(sin_lat, 1e-12) - n * (1.0 - e2)
        lat = math.atan2(z, p * (1.0 - e2 * n / max(n + alt, 1e-12))) if p else lat
    return lat, lon


def _enu_errors(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Compute horizontal E/N error using a local frame at each truth point."""

    est = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(truth, dtype=np.float64).reshape(-1, 3)
    if len(est) != len(ref):
        raise ValueError("estimated and truth lengths differ")
    errors = np.empty(len(est), dtype=np.float64)
    for index, (position, reference) in enumerate(zip(est, ref)):
        lat, lon = _ecef_to_lla(reference)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        diff = position - reference
        east = -sin_lon * diff[0] + cos_lon * diff[1]
        north = -sin_lat * cos_lon * diff[0] - sin_lat * sin_lon * diff[1] + cos_lat * diff[2]
        errors[index] = math.hypot(east, north)
    return errors


def _scalar_metrics(errors: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(errors, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"n_epochs": 0, "rms_2d_m": float("nan")}
    return {
        "n_epochs": int(len(finite)),
        "mean_2d_m": float(np.mean(finite)),
        "rms_2d_m": float(np.sqrt(np.mean(finite * finite))),
        "p50_2d_m": float(np.percentile(finite, 50)),
        "p95_2d_m": float(np.percentile(finite, 95)),
        "max_2d_m": float(np.max(finite)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in keys})


def _write_svg(path: Path, pf_errors: np.ndarray, wls_errors: np.ndarray) -> None:
    """Write a dependency-free error timeline visualization."""

    width, height = 960, 440
    left, right, top, bottom = 72, 24, 48, 56
    plot_width = width - left - right
    plot_height = height - top - bottom
    arrays = [
        np.asarray(values, dtype=np.float64)[np.isfinite(values)]
        for values in (pf_errors, wls_errors)
    ]
    max_error = max([float(np.max(values)) for values in arrays if len(values)] + [1.0])
    max_error = max(1.0, min(max_error * 1.1, 1.0e6))

    def polyline(values: np.ndarray) -> str:
        points: list[str] = []
        values = np.asarray(values, dtype=np.float64)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = left + (index / max(len(values) - 1, 1)) * plot_width
            y = top + plot_height * (1.0 - min(float(value), max_error) / max_error)
            points.append(f"{x:.2f},{y:.2f}")
        return " ".join(points)

    grid = []
    for tick in range(5):
        y = top + plot_height * tick / 4.0
        value = max_error * (1.0 - tick / 4.0)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d8dee9"/>'
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#4c566a">{value:.3g}</text>'
        )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="72" y="26" font-size="18" font-family="sans-serif" fill="#1f2937">UrbanNav GPU PF error timeline</text>',
        *grid,
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#4c566a"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#4c566a"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{polyline(pf_errors)}"/>',
        f'<polyline fill="none" stroke="#dc2626" stroke-width="1.5" points="{polyline(wls_errors)}"/>',
        f'<text x="{left + 10}" y="{height-18}" font-size="12" font-family="sans-serif" fill="#4c566a">epoch</text>',
        f'<text x="16" y="{top + plot_height/2}" transform="rotate(-90 16 {top + plot_height/2})" font-size="12" font-family="sans-serif" fill="#4c566a">horizontal error [m]</text>',
        '<rect x="690" y="18" width="14" height="3" fill="#2563eb"/><text x="712" y="23" font-size="12" font-family="sans-serif">GPU PF</text>',
        '<rect x="790" y="18" width="14" height="3" fill="#dc2626"/><text x="812" y="23" font-size="12" font-family="sans-serif">WLS</text>',
        '</svg>',
    ]
    path.write_text("".join(svg), encoding="utf-8")


def _write_summary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), Mapping) else {}
    lines = [
        "# UrbanNav GPU particle-filter run",
        "",
        f"- Dataset: {summary.get('dataset_name', 'unknown')}",
        f"- Backend: {summary.get('backend', 'unknown')}",
        f"- Epochs: {summary.get('n_epochs', 0)}",
        "",
        "| Method | RMS 2D [m] | P50 [m] | P95 [m] | Max [m] |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("GPU PF", "pf"), ("GPU WLS", "wls")):
        row = metrics.get(key, {}) if isinstance(metrics.get(key), Mapping) else {}
        lines.append(
            f"| {label} | {float(row.get('rms_2d_m', float('nan'))):.4g} | "
            f"{float(row.get('p50_2d_m', float('nan'))):.4g} | "
            f"{float(row.get('p95_2d_m', float('nan'))):.4g} | "
            f"{float(row.get('max_2d_m', float('nan'))):.4g} |"
        )
    lines.extend(("", "The run was executed locally; no data was downloaded by the CLI.", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def run_urbannav_pf(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    particles: int = 10_000,
    max_epochs: int | None = 300,
    start_epoch: int = 0,
    systems: tuple[str, ...] = ("G",),
    rover_source: str = "ublox",
    seed: int = 42,
    no_plots: bool = False,
    loader_factory: Callable[[str | Path], Any] | None = None,
    wls_solver: Callable[..., Any] | None = None,
    pf_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the existing UrbanNav/PPC loader through the CUDA PF.

    The injectable factories are intended for GPU-less unit tests. Production
    defaults always resolve to the repository's real loaders and native
    CUDA-backed classes; this onboarding preset has no CPU fallback.
    """

    if particles <= 0:
        raise UrbanNavRunError("--particles must be positive")
    if max_epochs is not None and max_epochs <= 0:
        raise UrbanNavRunError("--max-epochs must be positive when provided")
    if start_epoch < 0:
        raise UrbanNavRunError("--start-epoch must be non-negative")
    inspection = inspect_input(input_path)
    run_dir, format_name = _resolve_ready_run(inspection)
    loader_cls = loader_factory or (PPCDatasetLoader if format_name == "ppc" else UrbanNavLoader)
    loader = loader_cls(run_dir)
    try:
        if format_name == "ppc":
            data = loader.load_experiment_data(
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                systems=systems,
            )
        else:
            data = loader.load_experiment_data(
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                systems=systems,
                rover_source=rover_source,
            )
    except Exception as exc:
        raise UrbanNavRunError(
            f"could not load {inspection.detected_format}: {exc}. "
            "Run gnss-gpu data inspect PATH to see the exact input contract."
        ) from exc

    n_epochs = int(data.get("n_epochs", 0))
    if n_epochs <= 0:
        raise UrbanNavRunError("the input produced no usable epochs; check RINEX/navigation time overlap")
    times = np.asarray(data.get("times", []), dtype=np.float64)
    truth = np.asarray(data.get("ground_truth", []), dtype=np.float64).reshape(-1, 3)
    sat_ecef = data.get("sat_ecef")
    pseudoranges = data.get("pseudoranges")
    weights = data.get("weights")
    if len(truth) != n_epochs or not isinstance(sat_ecef, Sequence):
        raise UrbanNavRunError("loader returned an incomplete experiment data contract")

    if wls_solver is None:
        try:
            from gnss_gpu import wls_position as wls_solver  # type: ignore[assignment]
        except Exception as exc:
            raise UrbanNavRunError(
                f"CUDA WLS binding is unavailable: {exc}. "
                "Run gnss-gpu doctor and gnss-gpu build."
            ) from exc
    wls_positions = np.zeros((n_epochs, 4), dtype=np.float64)
    for index in range(n_epochs):
        wls_positions[index] = _finite_wls_solution(
            wls_solver,
            sat_ecef[index],
            pseudoranges[index],
            weights[index],
        )

    if pf_factory is None:
        try:
            from gnss_gpu import ParticleFilter as pf_factory  # type: ignore[assignment]
        except Exception as exc:
            raise UrbanNavRunError(
                f"CUDA ParticleFilter binding is unavailable: {exc}. "
                "Run gnss-gpu doctor and gnss-gpu build; this preset does not use CPU fallback."
            ) from exc

    try:
        pf = pf_factory(
            n_particles=particles,
            sigma_pos=2.0,
            sigma_cb=300.0,
            sigma_pr=8.0,
            resampling="megopolis",
            seed=seed,
        )
        pf.initialize(
            wls_positions[0, :3],
            clock_bias=float(wls_positions[0, 3]),
            spread_pos=50.0,
            spread_cb=500.0,
        )
    except Exception as exc:
        raise UrbanNavRunError(
            f"CUDA ParticleFilter initialization failed: {exc}. "
            "Run gnss-gpu doctor, then rebuild the CUDA PF extension; this preset does not use CPU fallback."
        ) from exc

    pf_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    started = time.perf_counter()
    for index in range(n_epochs):
        if index == 0:
            dt = float(data.get("dt", 1.0))
        elif len(times) > index and times[index] > times[index - 1]:
            dt = float(times[index] - times[index - 1])
        else:
            dt = float(data.get("dt", 1.0))
        try:
            pf.predict(dt=dt)
            pf.update(sat_ecef[index], pseudoranges[index], weights=weights[index])
            estimate = np.asarray(pf.estimate(), dtype=np.float64).ravel()
        except Exception as exc:
            raise UrbanNavRunError(
                f"CUDA ParticleFilter failed at epoch {index}: {exc}. "
                "Check the CUDA runtime with gnss-gpu doctor."
            ) from exc
        if estimate.size < 3 or not np.all(np.isfinite(estimate[:3])):
            raise UrbanNavRunError(f"CUDA ParticleFilter returned an invalid estimate at epoch {index}")
        pf_positions[index] = estimate[:3]
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    pf_errors = _enu_errors(pf_positions, truth)
    wls_errors = _enu_errors(wls_positions[:, :3], truth)
    if not np.all(np.isfinite(pf_errors)) or not np.all(np.isfinite(wls_errors)):
        raise UrbanNavRunError(
            "the run produced a non-finite positioning error; "
            "check the reference coordinates and RINEX/navigation data"
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory_csv = output / "urbannav_pf_trajectory.csv"
    rows = []
    for index in range(n_epochs):
        rows.append(
            {
                "epoch": index,
                "gps_tow": float(times[index]) if len(times) > index else index,
                "pf_x": float(pf_positions[index, 0]),
                "pf_y": float(pf_positions[index, 1]),
                "pf_z": float(pf_positions[index, 2]),
                "pf_error_2d_m": float(pf_errors[index]),
                "wls_x": float(wls_positions[index, 0]),
                "wls_y": float(wls_positions[index, 1]),
                "wls_z": float(wls_positions[index, 2]),
                "wls_error_2d_m": float(wls_errors[index]),
                "satellite_count": int(len(sat_ecef[index])),
            }
        )
    _write_csv(trajectory_csv, rows)
    summary_json = output / "urbannav_pf_summary.json"
    summary_md = output / "urbannav_pf_summary.md"
    visualization = output / "urbannav_pf_error_timeline.svg"
    summary: dict[str, Any] = {
        "schema": "gnss_gpu_urbannav_pf_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": str(data.get("dataset_name", run_dir.name)),
        "input": str(run_dir),
        "format": format_name,
        "backend": "CUDA",
        "n_epochs": n_epochs,
        "n_satellites_median": int(data.get("n_satellites", 0)),
        "constellations": list(data.get("constellations", ())),
        "rover_source": rover_source,
        "elapsed_ms": elapsed_ms,
        "metrics": {
            "pf": _scalar_metrics(pf_errors),
            "wls": _scalar_metrics(wls_errors),
            "runtime_ms": elapsed_ms,
        },
        "parameters": {
            "preset": URBANNAV_PRESET,
            "particles": particles,
            "max_epochs": max_epochs,
            "start_epoch": start_epoch,
            "systems": list(systems),
            "rover_source": rover_source,
            "seed": seed,
        },
        "artifacts": {
            "trajectory_csv": str(trajectory_csv),
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "error_timeline_svg": str(visualization),
        },
    }
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_summary_markdown(summary_md, summary)
    # Keep the visualization artifact stable for compare/report consumers.
    if not no_plots:
        _write_svg(visualization, pf_errors, wls_errors)
    else:
        visualization.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120">'
            '<text x="16" y="48" font-family="sans-serif">Visualization disabled with --no-plots</text></svg>',
            encoding="utf-8",
        )
    return {
        "preset": URBANNAV_PRESET,
        "backend": "CUDA",
        "elapsed_ms": elapsed_ms,
        "dataset_name": str(data.get("dataset_name", run_dir.name)),
        "n_epochs": n_epochs,
        "n_satellites": int(data.get("n_satellites", 0)),
        "constellations": list(data.get("constellations", ())),
        "metrics": {
            "pf_rms_2d_m": float(summary["metrics"]["pf"]["rms_2d_m"]),
            "pf_p50_2d_m": float(summary["metrics"]["pf"]["p50_2d_m"]),
            "pf_p95_2d_m": float(summary["metrics"]["pf"]["p95_2d_m"]),
            "wls_rms_2d_m": float(summary["metrics"]["wls"]["rms_2d_m"]),
            "wls_p50_2d_m": float(summary["metrics"]["wls"]["p50_2d_m"]),
            "runtime_ms": elapsed_ms,
        },
        "artifact_paths": {
            "trajectory_csv": trajectory_csv,
            "summary_json": summary_json,
            "summary_markdown": summary_md,
            "error_timeline_svg": visualization,
        },
        "input_paths": [path for path in inspection.files.values() if path is not None],
        "parameters": summary["parameters"],
        "summary": summary,
    }


__all__ = [
    "InputInspection",
    "InputInspectionError",
    "UrbanNavRunError",
    "URBANNAV_PRESET",
    "format_inspection",
    "inspect_input",
    "run_urbannav_pf",
]
