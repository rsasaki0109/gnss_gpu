#!/usr/bin/env python3
"""Validate source-to-product PPC FIX-rate inference bundles.

The product scorer starts from prepared epoch/window/base CSVs, but those CSVs
must correspond to real PPC source runs.  This helper validates that contract:

- raw PPC run directories exist and have rover/base/nav/reference files
- derived epoch/window/base CSVs are present and keyed to the raw runs
- every product window row has a matching base prediction row
- optional product output paths do not collide
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


EPOCH_REQUIRED_COLUMNS = {"city", "run", "gps_tow"}
WINDOW_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
    "window_start_tow",
    "window_end_tow",
    "sim_matched_epochs",
}
BASE_REQUIRED_COLUMNS = {"city", "run", "window_index"}
BASE_VALUE_COLUMNS = {"corrected_pred_fix_rate_pct", "pred_fix_rate_pct"}


@dataclass(frozen=True)
class SourceRun:
    city: str
    run: str
    run_dir: Path
    demo5_pos: Path | None = None
    sim_sat_csv: Path | None = None


@dataclass(frozen=True)
class SourceBundle:
    manifest_path: Path
    runs: tuple[SourceRun, ...]
    epochs_csv: Path
    window_csv: Path
    base_prediction_csv: Path
    prepare_prefix: str | None = None
    prepared_window_csv: Path | None = None
    inference_output_prefix: Path | None = None
    raw_source_prepare: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedSourceBundle:
    manifest_path: Path
    run_count: int
    run_keys: tuple[tuple[str, str], ...]
    epochs_csv: Path
    window_csv: Path
    base_prediction_csv: Path
    prepare_prefix: str | None
    prepared_window_csv: Path | None
    inference_output_prefix: Path | None


def _die(message: str) -> None:
    sys.stderr.write(f"\nERROR: {message}\n")
    raise SystemExit(1)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _die(f"{label} must be an object")
    return value


def _resolve_path(value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _die(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _optional_path(value: Any, base_dir: Path, label: str) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value, base_dir, label)


def _read_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return set(reader.fieldnames or [])


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        _die(f"missing {label}:\n  {path}")
    if not path.is_file():
        _die(f"{label} is not a file:\n  {path}")


def _require_columns(path: Path, label: str, required: set[str]) -> set[str]:
    _require_file(path, label)
    columns = _read_columns(path)
    missing = sorted(required - columns)
    if missing:
        _die(f"{label} is missing columns: {', '.join(missing)}\n  {path}")
    return columns


def _normalize_window_key(row: dict[str, str], path: Path) -> tuple[str, str, int]:
    try:
        window_index = int(float(row["window_index"]))
    except (KeyError, TypeError, ValueError):
        _die(f"invalid window_index in {path}")
    return str(row["city"]), str(row["run"]), window_index


def _route_keys(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {(str(row["city"]), str(row["run"])) for row in reader}


def _route_counts(path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (str(row["city"]), str(row["run"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _window_keys(path: Path) -> set[tuple[str, str, int]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {_normalize_window_key(row, path) for row in reader}


def _window_route_counts(path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (str(row["city"]), str(row["run"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return sum(1 for _row in reader)


def _preview(items: set[tuple[Any, ...]], limit: int = 5) -> str:
    ordered = sorted(items)
    text = ", ".join("/".join(str(part) for part in item) for item in ordered[:limit])
    if len(ordered) > limit:
        text += f" ... (+{len(ordered) - limit} more)"
    return text


def _discover_runs(data_root: Path) -> tuple[SourceRun, ...]:
    if PPCDatasetLoader.is_run_directory(data_root):
        run_dirs = [data_root]
    else:
        run_dirs = [
            path
            for path in sorted(data_root.rglob("run*"))
            if PPCDatasetLoader.is_run_directory(path)
        ]
    return tuple(
        SourceRun(city=path.parent.name, run=path.name, run_dir=path)
        for path in run_dirs
    )


def _explicit_runs(rows: Any, base_dir: Path) -> tuple[SourceRun, ...]:
    if not isinstance(rows, list):
        _die("runs must be a list")
    out: list[SourceRun] = []
    for idx, value in enumerate(rows):
        row = _as_mapping(value, f"runs[{idx}]")
        run_dir = _resolve_path(row.get("run_dir"), base_dir, f"runs[{idx}].run_dir")
        city = str(row.get("city") or run_dir.parent.name)
        run = str(row.get("run") or run_dir.name)
        out.append(
            SourceRun(
                city=city,
                run=run,
                run_dir=run_dir,
                demo5_pos=_optional_path(row.get("demo5_pos"), base_dir, f"runs[{idx}].demo5_pos"),
                sim_sat_csv=_optional_path(row.get("sim_sat_csv"), base_dir, f"runs[{idx}].sim_sat_csv"),
            )
        )
    return tuple(out)


def load_source_bundle(manifest_path: Path) -> SourceBundle:
    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    data = _as_mapping(raw, "manifest")
    base_dir = manifest_path.parent

    if "runs" in data:
        runs = _explicit_runs(data["runs"], base_dir)
    elif "data_root" in data:
        data_root = _resolve_path(data["data_root"], base_dir, "data_root")
        runs = _discover_runs(data_root)
    else:
        _die("manifest needs either runs[] or data_root")
    if not runs:
        _die("source manifest found no PPC runs")

    derived = _as_mapping(data.get("derived_inputs"), "derived_inputs")
    outputs = _as_mapping(data.get("outputs", {}), "outputs")
    prepare_prefix = _optional_path(outputs.get("prepare_prefix"), base_dir, "outputs.prepare_prefix")

    return SourceBundle(
        manifest_path=manifest_path,
        runs=runs,
        epochs_csv=_resolve_path(derived.get("epochs_csv"), base_dir, "derived_inputs.epochs_csv"),
        window_csv=_resolve_path(derived.get("window_csv"), base_dir, "derived_inputs.window_csv"),
        base_prediction_csv=_resolve_path(
            derived.get("base_prediction_csv"),
            base_dir,
            "derived_inputs.base_prediction_csv",
        ),
        prepare_prefix=str(prepare_prefix) if prepare_prefix is not None else None,
        prepared_window_csv=_optional_path(outputs.get("prepared_window_csv"), base_dir, "outputs.prepared_window_csv"),
        inference_output_prefix=_optional_path(
            outputs.get("inference_output_prefix"),
            base_dir,
            "outputs.inference_output_prefix",
        ),
        raw_source_prepare=(
            _as_mapping(data.get("raw_source_prepare"), "raw_source_prepare")
            if "raw_source_prepare" in data
            else None
        ),
    )


def _validate_run(run: SourceRun) -> None:
    if not PPCDatasetLoader.is_run_directory(run.run_dir):
        missing = [
            name
            for name in PPCDatasetLoader.REQUIRED_FILES
            if not (run.run_dir / name).exists()
        ]
        detail = f" missing: {', '.join(missing)}" if missing else ""
        _die(f"invalid PPC run directory for {run.city}/{run}:{detail}\n  {run.run_dir}")
    if run.demo5_pos is not None:
        _require_file(run.demo5_pos, f"demo5 pos for {run.city}/{run}")
    if run.sim_sat_csv is not None:
        _require_file(run.sim_sat_csv, f"simulator satellite CSV for {run.city}/{run}")


def _validate_output_paths(bundle: SourceBundle) -> None:
    if bundle.prepared_window_csv is None or bundle.inference_output_prefix is None:
        return
    final_window = bundle.inference_output_prefix.with_name(
        bundle.inference_output_prefix.name + "_window_predictions.csv"
    )
    if bundle.prepared_window_csv.expanduser().resolve() == final_window.expanduser().resolve():
        _die(
            "prepared_window_csv must not equal final inference window output:\n"
            f"  prepared: {bundle.prepared_window_csv}\n"
            f"  final:    {final_window}"
        )


def _metadata_count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        _die(f"{label} must be an integer count")
    try:
        count = int(value)
    except (TypeError, ValueError):
        _die(f"{label} must be an integer count")
    if count != value and not (isinstance(value, float) and value.is_integer()):
        _die(f"{label} must be an integer count")
    if count < 0:
        _die(f"{label} must be non-negative")
    return count


def _validate_raw_source_prepare_metadata(bundle: SourceBundle, raw_keys: set[tuple[str, str]]) -> None:
    metadata = bundle.raw_source_prepare
    if metadata is None:
        return

    expected_totals = {
        "epoch_count": _count_rows(bundle.epochs_csv),
        "window_count": _count_rows(bundle.window_csv),
        "base_prediction_count": _count_rows(bundle.base_prediction_csv),
    }
    for field, actual in expected_totals.items():
        if field not in metadata:
            _die(f"raw_source_prepare.{field} is missing from source manifest metadata")
        recorded = _metadata_count(metadata[field], f"raw_source_prepare.{field}")
        if recorded != actual:
            _die(
                f"raw_source_prepare.{field} does not match derived CSV rows:\n"
                f"  recorded: {recorded}\n"
                f"  actual:   {actual}"
            )

    summaries = metadata.get("runs")
    if not isinstance(summaries, list):
        _die("raw_source_prepare.runs must be a list")
    epoch_counts = _route_counts(bundle.epochs_csv)
    window_counts = _window_route_counts(bundle.window_csv)
    base_counts = _window_route_counts(bundle.base_prediction_csv)
    summary_keys: set[tuple[str, str]] = set()
    for idx, value in enumerate(summaries):
        row = _as_mapping(value, f"raw_source_prepare.runs[{idx}]")
        key = (str(row.get("city", "")), str(row.get("run", "")))
        if not key[0] or not key[1]:
            _die(f"raw_source_prepare.runs[{idx}] needs city and run")
        if key in summary_keys:
            _die(f"raw_source_prepare.runs contains duplicate run: {key[0]}/{key[1]}")
        summary_keys.add(key)
        for field, counts in (
            ("epoch_count", epoch_counts),
            ("window_count", window_counts),
            ("base_prediction_count", base_counts),
        ):
            if field not in row:
                _die(f"raw_source_prepare.runs[{idx}].{field} is missing")
            recorded = _metadata_count(row[field], f"raw_source_prepare.runs[{idx}].{field}")
            actual = counts.get(key, 0)
            if recorded != actual:
                _die(
                    f"raw_source_prepare.runs[{idx}].{field} does not match derived CSV rows for {key[0]}/{key[1]}:\n"
                    f"  recorded: {recorded}\n"
                    f"  actual:   {actual}"
                )

    if summary_keys != raw_keys:
        missing = raw_keys - summary_keys
        extra = summary_keys - raw_keys
        details: list[str] = []
        if missing:
            details.append(f"missing: {_preview(missing)}")
        if extra:
            details.append(f"extra: {_preview(extra)}")
        _die(f"raw_source_prepare.runs does not match manifest runs: {'; '.join(details)}")


def validate_source_bundle(bundle: SourceBundle) -> ValidatedSourceBundle:
    for run in bundle.runs:
        _validate_run(run)

    epoch_columns = _require_columns(bundle.epochs_csv, "derived epoch CSV", EPOCH_REQUIRED_COLUMNS)
    window_columns = _require_columns(bundle.window_csv, "derived window CSV", WINDOW_REQUIRED_COLUMNS)
    base_columns = _require_columns(bundle.base_prediction_csv, "base prediction CSV", BASE_REQUIRED_COLUMNS)
    if not (BASE_VALUE_COLUMNS & base_columns):
        _die(
            "base prediction CSV needs one prediction column:\n"
            f"  {bundle.base_prediction_csv}\n"
            f"  expected one of: {', '.join(sorted(BASE_VALUE_COLUMNS))}"
        )
    if "actual_fixed" in epoch_columns or "actual_fix_rate_pct" in window_columns:
        print("note: source bundle contains labels; product inference will ignore labels where not needed")

    raw_key_list = [(run.city, run.run) for run in bundle.runs]
    raw_keys = set(raw_key_list)
    if len(raw_keys) != len(raw_key_list):
        duplicates = {
            key
            for key in raw_keys
            if raw_key_list.count(key) > 1
        }
        _die(f"source manifest contains duplicate runs: {_preview(duplicates)}")

    epoch_keys = _route_keys(bundle.epochs_csv)
    window_route_keys = {(city, run) for city, run, _idx in _window_keys(bundle.window_csv)}
    missing_epoch = raw_keys - epoch_keys
    missing_window = raw_keys - window_route_keys
    extra_epoch = epoch_keys - raw_keys
    extra_window = window_route_keys - raw_keys
    if missing_epoch:
        _die(f"derived epoch CSV is missing raw runs: {_preview(missing_epoch)}")
    if missing_window:
        _die(f"derived window CSV is missing raw runs: {_preview(missing_window)}")
    if extra_epoch:
        _die(f"derived epoch CSV contains runs not declared in the source manifest: {_preview(extra_epoch)}")
    if extra_window:
        _die(f"derived window CSV contains runs not declared in the source manifest: {_preview(extra_window)}")

    window_keys = _window_keys(bundle.window_csv)
    base_keys = _window_keys(bundle.base_prediction_csv)
    missing_base = window_keys - base_keys
    if missing_base:
        _die(
            "base prediction CSV is missing window rows required by the product window CSV:\n"
            f"  {_preview(missing_base)}"
        )

    _validate_raw_source_prepare_metadata(bundle, raw_keys)
    _validate_output_paths(bundle)

    return ValidatedSourceBundle(
        manifest_path=bundle.manifest_path,
        run_count=len(bundle.runs),
        run_keys=tuple(sorted(raw_keys)),
        epochs_csv=bundle.epochs_csv,
        window_csv=bundle.window_csv,
        base_prediction_csv=bundle.base_prediction_csv,
        prepare_prefix=bundle.prepare_prefix,
        prepared_window_csv=bundle.prepared_window_csv,
        inference_output_prefix=bundle.inference_output_prefix,
    )


def _template() -> dict[str, Any]:
    return {
        "runs": [
            {
                "city": "nagoya",
                "run": "run1",
                "run_dir": "/path/to/PPC-Dataset/nagoya/run1",
                "demo5_pos": "/path/to/demo5_pos/nagoya_run1/rtklib.pos",
                "sim_sat_csv": "/path/to/simulator/nagoya_run1_satellites.csv",
            }
        ],
        "derived_inputs": {
            "epochs_csv": "/path/to/preprocessed_epochs.csv",
            "window_csv": "/path/to/window_features.csv",
            "base_prediction_csv": "/path/to/refinedgrid_window_predictions.csv",
        },
        "outputs": {
            "prepare_prefix": "experiments/results/my_run_prepare",
            "prepared_window_csv": "experiments/results/my_run_prepared_window_predictions.csv",
            "inference_output_prefix": "experiments/results/my_run_product",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PPC product source bundle manifests")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="validate a source bundle manifest")
    check.add_argument("--manifest", type=Path, required=True)
    template = sub.add_parser("template", help="write an example manifest")
    template.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "template":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_template(), indent=2) + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
        return
    if args.cmd == "check":
        validated = validate_source_bundle(load_source_bundle(args.manifest))
        print(
            "source bundle ok: "
            f"{validated.run_count} run(s), epochs={validated.epochs_csv}, "
            f"windows={validated.window_csv}, base={validated.base_prediction_csv}"
        )
        return
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
