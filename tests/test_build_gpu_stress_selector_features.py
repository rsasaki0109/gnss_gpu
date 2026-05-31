import csv
import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "build_gpu_stress_selector_features.py"
_SPEC = importlib.util.spec_from_file_location("build_gpu_stress_selector_features", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_FEATURES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FEATURES
_SPEC.loader.exec_module(_FEATURES)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_catalog_normalizes_sweep_rows(tmp_path):
    sweep = tmp_path / "sweep.csv"
    _write_csv(
        sweep,
        [
            {
                "jammer_jnr_db": "16",
                "spoof_delay_samples": "112",
                "building_height_scale": "1.4",
                "particles_per_epoch": "64",
                "target_snr": "3.5",
                "code_phase_error_samples": "112",
                "doppler_error_hz": "500",
                "false_lock": "True",
                "acquisition_miss": "False",
                "interference_detected": "True",
                "max_false_prn_snr": "1.2",
                "mean_blocked_ratio": "0.75",
                "max_blocked_ratio": "0.8",
                "mean_particle_shadow_contrast": "0.12",
                "total_risk_score": "0.91",
                "failure_label": "false_lock",
            }
        ],
    )

    rows = _FEATURES.build_catalog(sweep, run_id="gpu_test", tow_step_s=0.5)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "gpu_test"
    assert row["tow"] == 0.0
    assert row["gpu_rf_code_phase_abs_error_samples"] == 112.0
    assert row["gpu_rf_false_lock"] == 1.0
    assert row["gpu_failure_false_lock"] == 1.0
    assert row["gpu_stress_target_bad"] == 1.0
    assert row["gpu_urban_shadow_risk_score"] > 0.0
    assert row["gpu_urban_route_backend_cuda"] == 0.0


def test_augment_selector_features_merges_on_run_id_tow(tmp_path):
    base = tmp_path / "base.csv"
    epoch = tmp_path / "epoch.csv"
    _write_csv(
        base,
        [
            {"run_id": "tokyo_run1", "tow": "1.0", "label": "a", "rms": "0.1"},
            {"run_id": "tokyo_run1", "tow": "2.0", "label": "b", "rms": "0.2"},
        ],
    )
    epoch_row = {
        "run_id": "tokyo_run1",
        "tow": "1.0",
    }
    epoch_row.update(_FEATURES._neutral_gpu_features())
    epoch_row["gpu_total_risk_score"] = "0.7"
    epoch_row["gpu_failure_degraded"] = "1.0"
    epoch_row["gpu_failure_nominal"] = "0.0"
    _write_csv(epoch, [epoch_row])

    rows, missing = _FEATURES.augment_selector_features(
        base_selector_csv=base,
        epoch_feature_csv=epoch,
    )

    assert len(rows) == 2
    assert missing == 1
    assert rows[0]["gpu_total_risk_score"] == 0.7
    assert rows[1]["gpu_total_risk_score"] == 0.0
    assert rows[1]["gpu_failure_nominal"] == 1.0


def test_augment_selector_features_keeps_real_urban_nav_columns(tmp_path):
    base = tmp_path / "base.csv"
    epoch = tmp_path / "epoch.csv"
    _write_csv(
        base,
        [
            {"run_id": "tokyo_run1_nav", "tow": "187470.0", "label": "candidate_a"},
        ],
    )
    _write_csv(
        epoch,
        [
            {
                "run_id": "tokyo_run1_nav",
                "tow": "187470.0",
                "gpu_urban_backend": "cuda_bvh_batch",
                "gpu_urban_particle_backend": "cuda_bvh_batch",
                "gpu_urban_satellite_source": "nav_rinex",
                "gpu_urban_n_sat": "10",
                "gpu_urban_n_los": "8",
                "gpu_urban_n_nlos": "2",
                "gpu_urban_mean_blocked_ratio": "0.2",
                "gpu_urban_max_blocked_ratio": "0.2",
                "gpu_urban_low_elev_blocked_ratio": "0.5",
                "gpu_urban_expected_nlos_bias_m": "18.0",
                "gpu_urban_particle_shadow_contrast": "0.3",
                "gpu_urban_particles_per_epoch": "24",
                "gpu_urban_building_height_scale": "1.0",
            }
        ],
    )

    rows, missing = _FEATURES.augment_selector_features(
        base_selector_csv=base,
        epoch_feature_csv=epoch,
    )

    assert missing == 0
    row = rows[0]
    assert row["gpu_urban_n_sat"] == 10.0
    assert row["gpu_urban_n_nlos"] == 2.0
    assert row["gpu_urban_source_nav_rinex"] == 1.0
    assert row["gpu_urban_route_backend_cuda"] == 1.0
    assert row["gpu_urban_shadow_risk_score"] > 0.2
    assert row["gpu_total_risk_score"] == row["gpu_urban_shadow_risk_score"]


def test_normalize_epoch_feature_catalog_outputs_catalog_columns(tmp_path):
    epoch = tmp_path / "epoch.csv"
    _write_csv(
        epoch,
        [
            {
                "run_id": "tokyo_run1_nav",
                "tow": "187470.0",
                "gpu_urban_satellite_source": "nav_rinex",
                "gpu_urban_n_sat": "10",
                "gpu_urban_mean_blocked_ratio": "0.1",
                "gpu_urban_max_blocked_ratio": "0.2",
                "gpu_urban_particle_shadow_contrast": "0.4",
            }
        ],
    )

    rows = _FEATURES.normalize_epoch_feature_catalog(epoch)

    assert list(rows[0]) == _FEATURES.CATALOG_COLUMNS
    assert rows[0]["label"] == "tokyo_run1_nav_tow187470.0"
    assert rows[0]["gpu_urban_source_nav_rinex"] == 1.0
    assert rows[0]["gpu_failure_nominal"] == 1.0
