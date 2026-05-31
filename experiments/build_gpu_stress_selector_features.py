#!/usr/bin/env python3
"""Build selector-ready GPU stress features.

This converts Phase 3 `gpu_scenario_sweeper` output into a compact feature
catalog with stable `gpu_*` columns.  It can also augment an existing
candidate-level selector CSV by joining epoch-level GPU features on
`(run_id, tow)`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
DEFAULT_SWEEP_CSV = REPO / "experiments/results/gpu_scenario_sweeper/gpu_scenario_sweep_summary.csv"
DEFAULT_CATALOG_CSV = REPO / "experiments/results/gpu_stress_selector_features.csv"

GPU_RF_FEATURE_COLUMNS = [
    "gpu_rf_jammer_jnr_db",
    "gpu_rf_spoof_delay_samples",
    "gpu_rf_target_snr",
    "gpu_rf_code_phase_error_samples",
    "gpu_rf_code_phase_abs_error_samples",
    "gpu_rf_doppler_error_hz",
    "gpu_rf_false_lock",
    "gpu_rf_acquisition_miss",
    "gpu_rf_interference_detected",
    "gpu_rf_max_false_prn_snr",
]

GPU_URBAN_FEATURE_COLUMNS = [
    "gpu_urban_building_height_scale",
    "gpu_urban_particles_per_epoch",
    "gpu_urban_n_sat",
    "gpu_urban_n_los",
    "gpu_urban_n_nlos",
    "gpu_urban_mean_blocked_ratio",
    "gpu_urban_max_blocked_ratio",
    "gpu_urban_low_elev_blocked_ratio",
    "gpu_urban_mean_elevation_los_deg",
    "gpu_urban_mean_elevation_nlos_deg",
    "gpu_urban_expected_nlos_bias_m",
    "gpu_urban_route_weight_delta_log",
    "gpu_urban_particle_blocked_mean",
    "gpu_urban_particle_blocked_std",
    "gpu_urban_particle_shadow_contrast",
    "gpu_urban_shadow_risk_score",
    "gpu_urban_source_nav_rinex",
    "gpu_urban_source_satellite_json",
    "gpu_urban_route_backend_cuda",
    "gpu_urban_particle_backend_cuda",
]

GPU_FAILURE_FEATURE_COLUMNS = [
    "gpu_total_risk_score",
    "gpu_failure_nominal",
    "gpu_failure_degraded",
    "gpu_failure_high_risk",
    "gpu_failure_false_lock",
    "gpu_failure_acquisition_miss",
]

GPU_FEATURE_COLUMNS = [
    *GPU_RF_FEATURE_COLUMNS,
    *GPU_URBAN_FEATURE_COLUMNS,
    *GPU_FAILURE_FEATURE_COLUMNS,
]

CATALOG_COLUMNS = [
    "run_id",
    "tow",
    "label",
    "gpu_stress_scenario_id",
    "gpu_failure_label",
    "gpu_stress_target_bad",
    *GPU_FEATURE_COLUMNS,
]


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _first_present(row: dict, *names: str):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _feature_float(row: dict, output_col: str, *input_cols: str, default=0.0) -> float:
    value = _first_present(row, output_col, *input_cols)
    return _to_float(value, default)


def _backend_is_cuda(value) -> float:
    return float("cuda" in str(value or "").strip().lower())


def _shadow_risk_score(
    *,
    mean_blocked: float,
    max_blocked: float,
    low_elev_blocked: float,
    expected_bias_m: float,
    particle_contrast: float,
) -> float:
    score = (
        0.42 * max(0.0, min(1.0, mean_blocked))
        + 0.20 * max(0.0, min(1.0, max_blocked))
        + 0.18 * max(0.0, min(1.0, low_elev_blocked))
        + 0.14 * max(0.0, min(1.0, expected_bias_m / 45.0))
        + 0.06 * max(0.0, min(1.0, particle_contrast))
    )
    return max(0.0, min(1.0, float(score)))


def _failure_one_hot(failure_label: str) -> dict[str, float]:
    label = str(failure_label or "nominal").strip() or "nominal"
    return {
        "gpu_failure_nominal": float(label == "nominal"),
        "gpu_failure_degraded": float(label == "degraded"),
        "gpu_failure_high_risk": float(label == "high_risk"),
        "gpu_failure_false_lock": float(label == "false_lock"),
        "gpu_failure_acquisition_miss": float(label == "acquisition_miss"),
    }


def _scenario_id(row: dict, index: int) -> str:
    key = "|".join(
        [
            str(row.get("jammer_jnr_db", "")),
            str(row.get("spoof_delay_samples", "")),
            str(row.get("building_height_scale", "")),
            str(row.get("particles_per_epoch", "")),
            str(index),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _scenario_label(row: dict) -> str:
    return (
        f"gpu_jnr{_to_float(row.get('jammer_jnr_db')):g}"
        f"_d{_to_int(row.get('spoof_delay_samples'))}"
        f"_h{_to_float(row.get('building_height_scale')):g}"
        f"_p{_to_int(row.get('particles_per_epoch'))}"
    )


def _feature_row(row: dict, index: int, *, run_id: str, tow_step_s: float) -> dict:
    failure_label = str(row.get("failure_label", "nominal")).strip() or "nominal"
    target_bad = failure_label != "nominal"
    code_error = _to_float(row.get("code_phase_error_samples"))
    mean_blocked = _to_float(row.get("mean_blocked_ratio"))
    max_blocked = _to_float(row.get("max_blocked_ratio"))
    particle_contrast = _to_float(row.get("mean_particle_shadow_contrast"))
    urban_risk = _shadow_risk_score(
        mean_blocked=mean_blocked,
        max_blocked=max_blocked,
        low_elev_blocked=_to_float(row.get("low_elev_blocked_ratio")),
        expected_bias_m=_to_float(row.get("expected_nlos_bias_m")),
        particle_contrast=particle_contrast,
    )

    out = {
        "run_id": run_id,
        "tow": round(index * tow_step_s, 1),
        "label": _scenario_label(row),
        "gpu_stress_scenario_id": _scenario_id(row, index),
        "gpu_failure_label": failure_label,
        "gpu_stress_target_bad": float(target_bad),
        "gpu_rf_jammer_jnr_db": _to_float(row.get("jammer_jnr_db")),
        "gpu_rf_spoof_delay_samples": float(_to_int(row.get("spoof_delay_samples"))),
        "gpu_rf_target_snr": _to_float(row.get("target_snr")),
        "gpu_rf_code_phase_error_samples": code_error,
        "gpu_rf_code_phase_abs_error_samples": abs(code_error),
        "gpu_rf_doppler_error_hz": _to_float(row.get("doppler_error_hz")),
        "gpu_rf_false_lock": float(_to_bool(row.get("false_lock"))),
        "gpu_rf_acquisition_miss": float(_to_bool(row.get("acquisition_miss"))),
        "gpu_rf_interference_detected": float(_to_bool(row.get("interference_detected"))),
        "gpu_rf_max_false_prn_snr": _to_float(row.get("max_false_prn_snr")),
        "gpu_urban_building_height_scale": _to_float(row.get("building_height_scale"), 1.0),
        "gpu_urban_particles_per_epoch": float(_to_int(row.get("particles_per_epoch"))),
        "gpu_urban_n_sat": _to_float(row.get("n_sat")),
        "gpu_urban_n_los": _to_float(row.get("n_los")),
        "gpu_urban_n_nlos": _to_float(row.get("n_nlos")),
        "gpu_urban_mean_blocked_ratio": mean_blocked,
        "gpu_urban_max_blocked_ratio": max_blocked,
        "gpu_urban_low_elev_blocked_ratio": _to_float(row.get("low_elev_blocked_ratio")),
        "gpu_urban_mean_elevation_los_deg": _to_float(row.get("mean_elevation_los_deg")),
        "gpu_urban_mean_elevation_nlos_deg": _to_float(row.get("mean_elevation_nlos_deg")),
        "gpu_urban_expected_nlos_bias_m": _to_float(row.get("expected_nlos_bias_m")),
        "gpu_urban_route_weight_delta_log": _to_float(row.get("route_weight_delta_log")),
        "gpu_urban_particle_blocked_mean": _to_float(row.get("particle_blocked_mean")),
        "gpu_urban_particle_blocked_std": _to_float(row.get("particle_blocked_std")),
        "gpu_urban_particle_shadow_contrast": particle_contrast,
        "gpu_urban_shadow_risk_score": urban_risk,
        "gpu_urban_source_nav_rinex": 0.0,
        "gpu_urban_source_satellite_json": 0.0,
        "gpu_urban_route_backend_cuda": _backend_is_cuda(row.get("route_backend")),
        "gpu_urban_particle_backend_cuda": _backend_is_cuda(row.get("particle_backend")),
        "gpu_total_risk_score": _to_float(row.get("total_risk_score")),
        **_failure_one_hot(failure_label),
    }
    return out


def build_catalog(
    sweep_csv: Path,
    *,
    run_id: str = "gpu_sweep",
    tow_step_s: float = 1.0,
) -> list[dict]:
    with sweep_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        _feature_row(row, index, run_id=run_id, tow_step_s=tow_step_s)
        for index, row in enumerate(rows)
    ]


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _neutral_gpu_features() -> dict:
    return {
        "gpu_rf_jammer_jnr_db": 0.0,
        "gpu_rf_spoof_delay_samples": 0.0,
        "gpu_rf_target_snr": 0.0,
        "gpu_rf_code_phase_error_samples": 0.0,
        "gpu_rf_code_phase_abs_error_samples": 0.0,
        "gpu_rf_doppler_error_hz": 0.0,
        "gpu_rf_false_lock": 0.0,
        "gpu_rf_acquisition_miss": 0.0,
        "gpu_rf_interference_detected": 0.0,
        "gpu_rf_max_false_prn_snr": 0.0,
        "gpu_urban_building_height_scale": 1.0,
        "gpu_urban_particles_per_epoch": 0.0,
        "gpu_urban_n_sat": 0.0,
        "gpu_urban_n_los": 0.0,
        "gpu_urban_n_nlos": 0.0,
        "gpu_urban_mean_blocked_ratio": 0.0,
        "gpu_urban_max_blocked_ratio": 0.0,
        "gpu_urban_low_elev_blocked_ratio": 0.0,
        "gpu_urban_mean_elevation_los_deg": 0.0,
        "gpu_urban_mean_elevation_nlos_deg": 0.0,
        "gpu_urban_expected_nlos_bias_m": 0.0,
        "gpu_urban_route_weight_delta_log": 0.0,
        "gpu_urban_particle_blocked_mean": 0.0,
        "gpu_urban_particle_blocked_std": 0.0,
        "gpu_urban_particle_shadow_contrast": 0.0,
        "gpu_urban_shadow_risk_score": 0.0,
        "gpu_urban_source_nav_rinex": 0.0,
        "gpu_urban_source_satellite_json": 0.0,
        "gpu_urban_route_backend_cuda": 0.0,
        "gpu_urban_particle_backend_cuda": 0.0,
        "gpu_total_risk_score": 0.0,
        "gpu_failure_nominal": 1.0,
        "gpu_failure_degraded": 0.0,
        "gpu_failure_high_risk": 0.0,
        "gpu_failure_false_lock": 0.0,
        "gpu_failure_acquisition_miss": 0.0,
    }


def _normalize_gpu_feature_row(row: dict) -> dict:
    out = _neutral_gpu_features()

    for col in GPU_RF_FEATURE_COLUMNS:
        out[col] = _feature_float(row, col)

    out["gpu_urban_building_height_scale"] = _feature_float(
        row,
        "gpu_urban_building_height_scale",
        "building_height_scale",
        default=1.0,
    )
    out["gpu_urban_particles_per_epoch"] = _feature_float(
        row,
        "gpu_urban_particles_per_epoch",
        "particles_per_epoch",
    )
    out["gpu_urban_n_sat"] = _feature_float(row, "gpu_urban_n_sat", "n_sat")
    out["gpu_urban_n_los"] = _feature_float(row, "gpu_urban_n_los", "n_los")
    out["gpu_urban_n_nlos"] = _feature_float(row, "gpu_urban_n_nlos", "n_nlos")
    out["gpu_urban_mean_blocked_ratio"] = _feature_float(
        row,
        "gpu_urban_mean_blocked_ratio",
        "mean_blocked_ratio",
    )
    out["gpu_urban_max_blocked_ratio"] = _feature_float(
        row,
        "gpu_urban_max_blocked_ratio",
        "max_blocked_ratio",
    )
    out["gpu_urban_low_elev_blocked_ratio"] = _feature_float(
        row,
        "gpu_urban_low_elev_blocked_ratio",
        "low_elevation_blocked_ratio",
        "low_elev_blocked_ratio",
    )
    out["gpu_urban_mean_elevation_los_deg"] = _feature_float(
        row,
        "gpu_urban_mean_elevation_los_deg",
        "mean_elevation_los_deg",
    )
    out["gpu_urban_mean_elevation_nlos_deg"] = _feature_float(
        row,
        "gpu_urban_mean_elevation_nlos_deg",
        "mean_elevation_nlos_deg",
    )
    out["gpu_urban_expected_nlos_bias_m"] = _feature_float(
        row,
        "gpu_urban_expected_nlos_bias_m",
        "expected_nlos_bias_m",
    )
    out["gpu_urban_route_weight_delta_log"] = _feature_float(
        row,
        "gpu_urban_route_weight_delta_log",
        "route_weight_delta_log",
    )
    out["gpu_urban_particle_blocked_mean"] = _feature_float(
        row,
        "gpu_urban_particle_blocked_mean",
        "particle_blocked_mean",
    )
    out["gpu_urban_particle_blocked_std"] = _feature_float(
        row,
        "gpu_urban_particle_blocked_std",
        "particle_blocked_std",
    )
    out["gpu_urban_particle_shadow_contrast"] = _feature_float(
        row,
        "gpu_urban_particle_shadow_contrast",
        "mean_particle_shadow_contrast",
        "particle_shadow_contrast",
    )
    out["gpu_urban_shadow_risk_score"] = _feature_float(
        row,
        "gpu_urban_shadow_risk_score",
        default=-1.0,
    )
    if out["gpu_urban_shadow_risk_score"] < 0.0:
        out["gpu_urban_shadow_risk_score"] = _shadow_risk_score(
            mean_blocked=out["gpu_urban_mean_blocked_ratio"],
            max_blocked=out["gpu_urban_max_blocked_ratio"],
            low_elev_blocked=out["gpu_urban_low_elev_blocked_ratio"],
            expected_bias_m=out["gpu_urban_expected_nlos_bias_m"],
            particle_contrast=out["gpu_urban_particle_shadow_contrast"],
        )

    source = str(_first_present(row, "gpu_urban_satellite_source", "satellite_source") or "").lower()
    out["gpu_urban_source_nav_rinex"] = _feature_float(
        row,
        "gpu_urban_source_nav_rinex",
        default=float(source == "nav_rinex"),
    )
    out["gpu_urban_source_satellite_json"] = _feature_float(
        row,
        "gpu_urban_source_satellite_json",
        default=float(source == "satellite_json"),
    )
    out["gpu_urban_route_backend_cuda"] = _feature_float(
        row,
        "gpu_urban_route_backend_cuda",
        default=_backend_is_cuda(_first_present(row, "gpu_urban_backend", "route_backend")),
    )
    out["gpu_urban_particle_backend_cuda"] = _feature_float(
        row,
        "gpu_urban_particle_backend_cuda",
        default=_backend_is_cuda(_first_present(row, "gpu_urban_particle_backend", "particle_backend")),
    )

    total = _first_present(row, "gpu_total_risk_score", "total_risk_score")
    out["gpu_total_risk_score"] = (
        _to_float(total)
        if total not in (None, "")
        else out["gpu_urban_shadow_risk_score"]
    )

    failure_label = str(_first_present(row, "gpu_failure_label", "failure_label") or "nominal")
    out.update(_failure_one_hot(failure_label))
    for col in GPU_FAILURE_FEATURE_COLUMNS:
        if col == "gpu_total_risk_score":
            continue
        value = row.get(col)
        if value not in (None, ""):
            out[col] = _to_float(value)

    return out


def _load_epoch_features(path: Path) -> dict[tuple[str, float], dict]:
    out: dict[tuple[str, float], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (str(row["run_id"]), round(float(row["tow"]), 1))
            except (KeyError, ValueError):
                continue
            out[key] = _normalize_gpu_feature_row(row)
    return out


def augment_selector_features(
    *,
    base_selector_csv: Path,
    epoch_feature_csv: Path,
) -> tuple[list[dict], int]:
    feature_by_key = _load_epoch_features(epoch_feature_csv)
    neutral = _neutral_gpu_features()
    merged = []
    missing = 0
    with base_selector_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (str(row["run_id"]), round(float(row["tow"]), 1))
            except (KeyError, ValueError):
                key = ("", float("nan"))
            features = feature_by_key.get(key)
            if features is None:
                features = dict(neutral)
                missing += 1
            out = dict(row)
            out.update(features)
            merged.append(out)
    return merged, missing


def _normalized_epoch_id(row: dict, index: int) -> str:
    key = "|".join(
        [
            str(row.get("run_id", "")),
            str(row.get("tow", "")),
            str(row.get("label", "")),
            str(index),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def normalize_epoch_feature_catalog(epoch_feature_csv: Path) -> list[dict]:
    rows = []
    with epoch_feature_csv.open(newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            run_id = str(row.get("run_id", "gpu_epoch"))
            tow = round(_to_float(row.get("tow")), 1)
            failure_label = str(_first_present(row, "gpu_failure_label", "failure_label") or "nominal")
            features = _normalize_gpu_feature_row(row)
            out = {
                "run_id": run_id,
                "tow": tow,
                "label": str(row.get("label") or f"{run_id}_tow{tow:.1f}"),
                "gpu_stress_scenario_id": str(
                    row.get("gpu_stress_scenario_id") or _normalized_epoch_id(row, index)
                ),
                "gpu_failure_label": failure_label,
                "gpu_stress_target_bad": float(
                    _to_bool(row.get("gpu_stress_target_bad"))
                    or failure_label != "nominal"
                ),
            }
            out.update(features)
            rows.append(out)
    return rows


def _fieldnames_for_augmented(rows: list[dict]) -> list[str]:
    if not rows:
        return GPU_FEATURE_COLUMNS
    base = [col for col in rows[0].keys() if col not in GPU_FEATURE_COLUMNS]
    return [*base, *GPU_FEATURE_COLUMNS]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("catalog", "augment", "normalize"), default="catalog")
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CATALOG_CSV)
    parser.add_argument("--run-id", default="gpu_sweep")
    parser.add_argument("--tow-step-s", type=float, default=1.0)
    parser.add_argument("--base-selector-csv", type=Path, default=None)
    parser.add_argument("--epoch-feature-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "catalog":
        rows = build_catalog(args.sweep_csv, run_id=args.run_id, tow_step_s=args.tow_step_s)
        _write_rows(args.out_csv, rows, CATALOG_COLUMNS)
        print(f"[gpu-stress-features] wrote catalog {args.out_csv} rows={len(rows)}")
        return 0

    if args.mode == "normalize":
        epoch_feature_csv = args.epoch_feature_csv or args.sweep_csv
        rows = normalize_epoch_feature_catalog(epoch_feature_csv)
        _write_rows(args.out_csv, rows, CATALOG_COLUMNS)
        print(f"[gpu-stress-features] wrote normalized {args.out_csv} rows={len(rows)}")
        return 0

    if args.base_selector_csv is None:
        raise SystemExit("--base-selector-csv is required in augment mode")
    epoch_feature_csv = args.epoch_feature_csv or args.sweep_csv
    rows, missing = augment_selector_features(
        base_selector_csv=args.base_selector_csv,
        epoch_feature_csv=epoch_feature_csv,
    )
    _write_rows(args.out_csv, rows, _fieldnames_for_augmented(rows))
    print(
        f"[gpu-stress-features] wrote augmented {args.out_csv} "
        f"rows={len(rows)} missing_epoch_features={missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
