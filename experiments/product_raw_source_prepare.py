#!/usr/bin/env python3
"""Build bootstrap product CSV inputs from raw PPC source runs.

This is the first productized raw-source preparation step.  It reads PPC
RINEX/reference source directories, emits label-free epoch/window/base CSVs,
and writes a derived source manifest that can be passed to
``predict.py --source-bundle-inference``.

The generated features are model-schema compatible.  Features that require the
full research simulator/refinedgrid pipeline are filled with neutral zeros, so
this path is a bootstrap/degraded raw-source bridge rather than a replacement
for the calibrated upstream feature pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402

from product_inference_model import DEFAULT_MODEL_PATH  # noqa: E402
from product_source_bundle import SourceRun  # noqa: E402


RESULTS_DIR = _SCRIPT_DIR / "results"
STAT_NAMES = ("mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max")
KEY_COLUMNS = {"city", "run", "gps_tow", "epoch", "_window_index"}
DEFAULT_SYSTEMS = ("G", "E", "J")


@dataclass(frozen=True)
class RawPrepareOutputs:
    epochs_csv: Path
    window_csv: Path
    base_prediction_csv: Path
    source_manifest: Path


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


def _discover_runs(data_root: Path) -> tuple[SourceRun, ...]:
    if PPCDatasetLoader.is_run_directory(data_root):
        run_dirs = [data_root]
    else:
        run_dirs = [
            path
            for path in sorted(data_root.rglob("run*"))
            if PPCDatasetLoader.is_run_directory(path)
        ]
    return tuple(SourceRun(city=path.parent.name, run=path.name, run_dir=path) for path in run_dirs)


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


def load_raw_source_runs(manifest_path: Path) -> tuple[tuple[SourceRun, ...], dict[str, Any], Path]:
    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    data = _as_mapping(raw, "manifest")
    base_dir = manifest_path.parent
    if "runs" in data:
        runs = _explicit_runs(data["runs"], base_dir)
    elif "data_root" in data:
        runs = _discover_runs(_resolve_path(data["data_root"], base_dir, "data_root"))
    else:
        _die("manifest needs either runs[] or data_root")
    if not runs:
        _die("source manifest found no PPC runs")
    keys = [(run.city, run.run) for run in runs]
    if len(set(keys)) != len(keys):
        _die("source manifest contains duplicate city/run keys")
    return runs, data, base_dir


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _die(f"no rows to write for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path} ({len(rows)} rows)")


def _load_model_feature_names(model_path: Path) -> list[str]:
    with gzip.open(model_path, "rb") as fh:
        artifact = pickle.load(fh)
    names = artifact.get("raw_feature_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        _die(f"model artifact does not contain raw_feature_names:\n  {model_path}")
    return list(names)


def _first_obs(obs: dict[str, float], prefixes: tuple[str, ...]) -> float:
    for code, value in obs.items():
        if code.startswith(prefixes) and value:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {name: 0.0 for name in STAT_NAMES}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _count(values: list[float] | np.ndarray, predicate) -> int:
    arr = _finite(values)
    if arr.size == 0:
        return 0
    return int(np.count_nonzero(predicate(arr)))


def _phase_epoch_rows(run: SourceRun, systems: tuple[str, ...]) -> dict[float, dict[str, float]]:
    obs = read_rinex_obs(run.run_dir / "rover.obs")
    previous_phase: dict[str, float] = {}
    previous_tow: dict[str, float] = {}
    streak_s: dict[str, float] = {}
    rows: dict[float, dict[str, float]] = {}
    for epoch in obs.epochs:
        tow = float(_datetime_to_gps_seconds_of_week(epoch.time))
        phase_deltas: list[float] = []
        doppler_residuals: list[float] = []
        present_sats: set[str] = set()
        lli_count = 0
        for sat_id in epoch.satellites:
            if not sat_id or sat_id[0] not in systems:
                continue
            sat_obs = epoch.observations.get(sat_id, {})
            phase = _first_obs(sat_obs, ("L",))
            doppler = _first_obs(sat_obs, ("D",))
            if not math.isfinite(phase) or phase == 0.0:
                continue
            present_sats.add(sat_id)
            if sat_id in previous_phase and sat_id in previous_tow:
                dt = max(tow - previous_tow[sat_id], 0.0)
                delta = phase - previous_phase[sat_id]
                phase_deltas.append(abs(delta))
                if math.isfinite(doppler) and dt > 0.0:
                    doppler_residuals.append(abs(delta + doppler * dt))
                if abs(delta) <= 0.25:
                    streak_s[sat_id] = streak_s.get(sat_id, 0.0) + dt
                else:
                    streak_s[sat_id] = 0.0
            else:
                streak_s[sat_id] = 0.0
            previous_phase[sat_id] = phase
            previous_tow[sat_id] = tow

        disappeared = set(previous_phase) - present_sats
        break_count = len([sat for sat in disappeared if previous_tow.get(sat, tow) < tow])
        for sat_id in disappeared:
            previous_phase.pop(sat_id, None)
            previous_tow.pop(sat_id, None)
            streak_s.pop(sat_id, None)
        raw_stats = _stats(phase_deltas)
        dop_stats = _stats(doppler_residuals)
        rows[round(tow, 3)] = {
            "rinex_phase_present_count": float(len(present_sats)),
            "rinex_phase_fraction": float(len(present_sats) / max(len(epoch.satellites), 1)),
            "rinex_phase_streak_ge30p0s_count": float(sum(1 for sat in present_sats if streak_s.get(sat, 0.0) >= 30.0)),
            "rinex_phase_streak_ge60p0s_count": float(sum(1 for sat in present_sats if streak_s.get(sat, 0.0) >= 60.0)),
            "rinex_gf_streak_ge30p0s_count": 0.0,
            "rinex_gf_streak_ge60p0s_count": 0.0,
            "rinex_phase_jump_ge0p25cy_count": float(_count(phase_deltas, lambda arr: arr >= 0.25)),
            "rinex_phase_jump_ge0p5cy_count": float(_count(phase_deltas, lambda arr: arr >= 0.5)),
            "rinex_phase_jump_ge1p0cy_count": float(_count(phase_deltas, lambda arr: arr >= 1.0)),
            "rinex_gf_slip_ge0p2m_count": 0.0,
            "rinex_gf_slip_ge0p5m_count": 0.0,
            "rinex_phase_break_count": float(break_count),
            "rinex_phase_lost_count": 0.0,
            "rinex_phase_lli_count": float(lli_count),
            "rinex_phase_raw_delta_cycles_p50": raw_stats["p50"],
            "rinex_phase_raw_delta_cycles_p90": raw_stats["p90"],
            "rinex_phase_raw_delta_cycles_max": raw_stats["max"],
            "rinex_phase_doppler_min_residual_cycles_p50": dop_stats["p50"],
            "rinex_phase_doppler_min_residual_cycles_p90": dop_stats["p90"],
            "rinex_phase_doppler_min_residual_cycles_max": dop_stats["max"],
        }
    return rows


def _epoch_rows_for_run(
    run: SourceRun,
    *,
    systems: tuple[str, ...],
    max_epochs: int | None,
) -> list[dict[str, object]]:
    if not PPCDatasetLoader.is_run_directory(run.run_dir):
        missing = [name for name in PPCDatasetLoader.REQUIRED_FILES if not (run.run_dir / name).exists()]
        _die(f"invalid PPC run directory for {run.city}/{run.run}: missing {', '.join(missing)}\n  {run.run_dir}")
    data = PPCDatasetLoader(run.run_dir).load_experiment_data(max_epochs=max_epochs, systems=systems)
    phase_by_tow = _phase_epoch_rows(run, systems)
    sat_streak_s: dict[str, float] = {}
    prev_tow: float | None = None
    rows: list[dict[str, object]] = []
    for idx, tow in enumerate(np.asarray(data["times"], dtype=np.float64)):
        dt = 0.0 if prev_tow is None else max(float(tow - prev_tow), 0.0)
        sat_ids = [str(sat) for sat in data["used_prns"][idx]]
        sat_set = set(sat_ids)
        for sat_id in list(sat_streak_s):
            if sat_id not in sat_set:
                sat_streak_s[sat_id] = 0.0
        for sat_id in sat_ids:
            sat_streak_s[sat_id] = sat_streak_s.get(sat_id, 0.0) + dt
        sat_count = float(len(sat_ids))
        weights = np.asarray(data["weights"][idx], dtype=np.float64)
        snr_stats = _stats(weights)
        phase = phase_by_tow.get(round(float(tow), 3), {})
        los30 = float(sum(1 for sat_id in sat_ids if sat_streak_s.get(sat_id, 0.0) >= 30.0))
        los60 = float(sum(1 for sat_id in sat_ids if sat_streak_s.get(sat_id, 0.0) >= 60.0))
        # ADOP is not available in this bootstrap path.  Use a monotonic
        # geometry proxy so downstream validationhold state has a neutral,
        # deterministic signal instead of a missing column.
        log10_adop_proxy = float(math.log10(1.0 / max(sat_count - 3.0, 1.0)))
        row: dict[str, object] = {
            "city": run.city,
            "run": run.run,
            "epoch": idx,
            "gps_tow": float(tow),
            "sat": sat_count,
            "phase": float(phase.get("rinex_phase_present_count", 0.0)),
            "lli": float(phase.get("rinex_phase_lli_count", 0.0)),
            "los": los30,
            "nlos": max(sat_count - los30, 0.0),
            "residual": 0.0,
            "log_residual": 0.0,
            "phase_fraction": float(phase.get("rinex_phase_fraction", 0.0)),
            "los_fraction": los30 / max(sat_count, 1.0),
            "nlos_fraction": max(sat_count - los30, 0.0) / max(sat_count, 1.0),
            "phase_los_gap": float(phase.get("rinex_phase_present_count", 0.0)) - los30,
            "residual_per_los": 0.0,
            "snr_mean": snr_stats["mean"],
            "snr_p50": snr_stats["p50"],
            "snr_min": snr_stats["min"],
            "sim_los_cont_ge30p0s_count": los30,
            "sim_los_cont_ge60p0s_count": los60,
            "sim_los_system_g_cont_ge30p0s_count": los30,
            "sim_los_system_g_cont_ge60p0s_count": los60,
            "sim_adop_cont_ge30p0s_count": los30,
            "sim_adop_cont_ge60p0s_count": los60,
            "sim_adop_los_count": los30,
            "sim_adop_all_count": sat_count,
            "sim_adop_cont_ge30p0s_mean_el_deg": 0.0,
            "sim_adop_los_mean_el_deg": 0.0,
            "sim_adop_cont_ge30p0s_log10_adop": log10_adop_proxy,
            "sim_adop_cont_ge60p0s_log10_adop": log10_adop_proxy,
            "sim_adop_los_log10_adop": log10_adop_proxy,
            "sim_adop_all_log10_adop": log10_adop_proxy,
        }
        row.update(phase)
        rows.append(row)
        prev_tow = float(tow)
    return rows


def _numeric_epoch_columns(rows: list[dict[str, object]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in KEY_COLUMNS:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in names:
                names.append(key)
    return names


def _baseline_prediction(row: dict[str, object]) -> float:
    sat = float(row.get("sat_p50", row.get("sat_mean", 0.0)) or 0.0)
    phase_fraction = float(row.get("phase_fraction_mean", 0.0) or 0.0)
    los_fraction = float(row.get("los_fraction_mean", 0.0) or 0.0)
    jump = float(row.get("rinex_phase_jump_ge0p5cy_count_max", 0.0) or 0.0)
    snr = float(row.get("snr_p50_mean", 0.0) or 0.0)
    sat_score = max(0.0, min(1.0, (sat - 4.0) / 8.0))
    snr_score = max(0.0, min(1.0, (snr - 20.0) / 25.0)) if snr > 0.0 else 0.35
    clean = math.exp(-0.35 * max(jump, 0.0))
    pred = 100.0 * (0.04 + 0.58 * sat_score * max(phase_fraction, 0.1) * max(los_fraction, 0.1) * clean + 0.12 * snr_score)
    return float(max(0.0, min(95.0, pred)))


def build_window_rows(
    epoch_rows: list[dict[str, object]],
    *,
    model_feature_names: list[str],
    window_duration_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in epoch_rows:
        grouped.setdefault((str(row["city"]), str(row["run"])), []).append(row)

    window_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    for (city, run), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: float(row["gps_tow"]))
        start_tow = float(rows[0]["gps_tow"])
        window_indices = {
            idx: int((float(row["gps_tow"]) - start_tow) // window_duration_s)
            for idx, row in enumerate(rows)
        }
        numeric_cols = _numeric_epoch_columns(rows)
        for window_index in sorted(set(window_indices.values())):
            selected = [row for idx, row in enumerate(rows) if window_indices[idx] == window_index]
            w_start = start_tow + window_index * window_duration_s
            w_end = w_start + window_duration_s
            out: dict[str, object] = {name: 0.0 for name in model_feature_names}
            out.update(
                {
                    "city": city,
                    "run": run,
                    "window_index": window_index,
                    "window_start_tow": w_start,
                    "window_end_tow": w_end,
                    "sim_matched_epochs": len(selected),
                    "sim_epoch_count": len(selected),
                    "matched_epoch_count": len(selected),
                    "sim_coverage_fraction": min(1.0, len(selected) * 0.2 / max(window_duration_s, 0.2)),
                    "window_duration_s": window_duration_s,
                    "window_index_value": float(window_index),
                }
            )
            for col in numeric_cols:
                values = [float(row.get(col, 0.0) or 0.0) for row in selected]
                for stat, value in _stats(values).items():
                    name = f"{col}_{stat}"
                    if name in out:
                        out[name] = value
                    else:
                        out.setdefault(name, value)
                if len(values) > 1:
                    deltas = np.diff(np.asarray(values, dtype=np.float64))
                else:
                    deltas = np.asarray([0.0], dtype=np.float64)
                delta_stats = _stats(deltas)
                for stat in ("mean", "std", "min", "p10", "p90", "max"):
                    name = f"{col}_delta_{stat}"
                    if name in out:
                        out[name] = delta_stats[stat]
                if f"{col}_delta_neg_count" in out:
                    out[f"{col}_delta_neg_count"] = int(np.count_nonzero(deltas < 0.0))
                if f"{col}_delta_pos_count" in out:
                    out[f"{col}_delta_pos_count"] = int(np.count_nonzero(deltas > 0.0))
                if f"{col}_delta_sign_change_count" in out:
                    signs = np.sign(deltas)
                    sign_changes = int(np.count_nonzero(np.diff(signs) != 0)) if len(signs) > 1 else 0
                    out[f"{col}_delta_sign_change_count"] = sign_changes
            pred = _baseline_prediction(out)
            base_rows.append(
                {
                    "city": city,
                    "run": run,
                    "window_index": window_index,
                    "corrected_pred_fix_rate_pct": pred,
                    "pred_fix_rate_pct": pred,
                    "raw_source_bootstrap": 1,
                }
            )
            window_rows.append(out)
    return window_rows, base_rows


def _default_output_prefix(manifest_path: Path) -> Path:
    return RESULTS_DIR / manifest_path.with_suffix("").name


def _manifest_runs_json(runs: tuple[SourceRun, ...]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for run in runs:
        row = {"city": run.city, "run": run.run, "run_dir": str(run.run_dir)}
        if run.demo5_pos is not None:
            row["demo5_pos"] = str(run.demo5_pos)
        if run.sim_sat_csv is not None:
            row["sim_sat_csv"] = str(run.sim_sat_csv)
        out.append(row)
    return out


def prepare_raw_source_bundle(
    *,
    manifest_path: Path,
    output_prefix: Path | None,
    output_manifest: Path | None,
    model_path: Path,
    systems: tuple[str, ...],
    window_duration_s: float,
    max_epochs_per_run: int | None,
) -> RawPrepareOutputs:
    runs, raw_manifest, _base_dir = load_raw_source_runs(manifest_path)
    model_feature_names = _load_model_feature_names(model_path)
    prefix = output_prefix or _default_output_prefix(manifest_path)
    prefix = prefix.expanduser()
    if not prefix.is_absolute():
        prefix = Path.cwd() / prefix
    out_manifest = output_manifest or prefix.with_name(prefix.name + "_source_manifest.json")
    epochs_csv = prefix.with_name(prefix.name + "_epochs.csv")
    window_csv = prefix.with_name(prefix.name + "_window_features.csv")
    base_csv = prefix.with_name(prefix.name + "_base_window_predictions.csv")

    epoch_rows: list[dict[str, object]] = []
    for run in runs:
        print(f"[raw] {run.city}/{run.run}: {run.run_dir}", flush=True)
        epoch_rows.extend(
            _epoch_rows_for_run(
                run,
                systems=systems,
                max_epochs=max_epochs_per_run,
            )
        )
    if not epoch_rows:
        _die("raw source preparation produced no epochs")
    window_rows, base_rows = build_window_rows(
        epoch_rows,
        model_feature_names=model_feature_names,
        window_duration_s=window_duration_s,
    )
    _write_rows(epochs_csv, epoch_rows)
    _write_rows(window_csv, window_rows)
    _write_rows(base_csv, base_rows)

    outputs = _as_mapping(raw_manifest.get("outputs", {}), "outputs")
    prepare_prefix = str(outputs.get("prepare_prefix") or prefix.with_name(prefix.name + "_prepare"))
    prepared_window_csv = str(
        outputs.get("prepared_window_csv")
        or prefix.with_name(prefix.name + "_prepared_window_predictions.csv")
    )
    inference_output_prefix = str(outputs.get("inference_output_prefix") or prefix.with_name(prefix.name + "_product"))
    derived_manifest = {
        "runs": _manifest_runs_json(runs),
        "derived_inputs": {
            "epochs_csv": str(epochs_csv),
            "window_csv": str(window_csv),
            "base_prediction_csv": str(base_csv),
        },
        "outputs": {
            "prepare_prefix": prepare_prefix,
            "prepared_window_csv": prepared_window_csv,
            "inference_output_prefix": inference_output_prefix,
        },
        "raw_source_prepare": {
            "source_manifest": str(manifest_path.expanduser().resolve()),
            "model_feature_count": len(model_feature_names),
            "systems": list(systems),
            "window_duration_s": float(window_duration_s),
            "max_epochs_per_run": max_epochs_per_run,
            "mode": "bootstrap_neutral_fill",
        },
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(derived_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out_manifest}")
    return RawPrepareOutputs(
        epochs_csv=epochs_csv,
        window_csv=window_csv,
        base_prediction_csv=base_csv,
        source_manifest=out_manifest,
    )


def preflight_raw_source_bundle(
    *,
    manifest_path: Path,
    model_path: Path,
) -> tuple[int, int]:
    runs, _raw_manifest, _base_dir = load_raw_source_runs(manifest_path)
    model_feature_names = _load_model_feature_names(model_path)
    invalid: list[str] = []
    for run in runs:
        if PPCDatasetLoader.is_run_directory(run.run_dir):
            continue
        missing = [name for name in PPCDatasetLoader.REQUIRED_FILES if not (run.run_dir / name).exists()]
        invalid.append(f"{run.city}/{run.run}: missing {', '.join(missing)} ({run.run_dir})")
    if invalid:
        _die("invalid PPC run directories:\n  " + "\n  ".join(invalid))
    return len(runs), len(model_feature_names)


def _parse_systems(value: str) -> tuple[str, ...]:
    systems = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not systems:
        _die("--systems must include at least one constellation")
    return systems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bootstrap PPC product CSVs from raw source runs")
    parser.add_argument("--manifest", type=Path, required=True, help="raw source manifest with runs[] or data_root")
    parser.add_argument("--output-prefix", type=Path, help="prefix for generated epoch/window/base CSVs")
    parser.add_argument("--output-manifest", type=Path, help="derived manifest for predict.py --source-bundle-inference")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="product model artifact used for feature schema")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS), help="constellation list, e.g. G,E,J")
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--max-epochs-per-run", type=int, help="debug cap per run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_epochs = args.max_epochs_per_run if args.max_epochs_per_run and args.max_epochs_per_run > 0 else None
    outputs = prepare_raw_source_bundle(
        manifest_path=args.manifest,
        output_prefix=args.output_prefix,
        output_manifest=args.output_manifest,
        model_path=args.model,
        systems=_parse_systems(args.systems),
        window_duration_s=args.window_duration_s,
        max_epochs_per_run=max_epochs,
    )
    print("\nraw source preparation finished")
    print(f"epochs CSV: {outputs.epochs_csv}")
    print(f"window CSV: {outputs.window_csv}")
    print(f"base CSV: {outputs.base_prediction_csv}")
    print(f"derived source manifest: {outputs.source_manifest}")


if __name__ == "__main__":
    main()
