import csv
import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "build_gpu_stress_lab_site.py"
_SPEC = importlib.util.spec_from_file_location("build_gpu_stress_lab_site", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SITE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SITE
_SPEC.loader.exec_module(_SITE)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_rf_counts_false_locks_and_detections():
    rows = [
        {
            "scenario": "clean",
            "acquisition_backend": "cuda_acquisition",
            "target_snr": "70",
            "false_lock": "False",
            "interference_detected": "False",
        },
        {
            "scenario": "delayed_replica",
            "acquisition_backend": "cuda_acquisition",
            "target_snr": "4",
            "false_lock": "True",
            "interference_detected": "True",
        },
    ]

    summary = _SITE._summarize_rf(rows)

    assert summary["n_scenarios"] == 2
    assert summary["backend"] == "cuda_acquisition"
    assert summary["false_lock_count"] == 1
    assert summary["interference_detected_count"] == 1
    assert summary["lowest_snr_scenario"] == "delayed_replica"


def test_build_snapshot_and_page_with_tmp_paths(tmp_path, monkeypatch):
    results = tmp_path / "experiments" / "results"
    docs = tmp_path / "docs"
    rf_dir = results / "gnss_security_lab"
    urban_dir = results / "urban_shadow_lab"
    sweep_dir = results / "gpu_scenario_sweeper"
    feature_csv = results / "gpu_stress_selector_features.csv"
    ppc_shadow_csv = results / "ppc_gpu_urban_shadow_features_tokyo_run1_nav_smoke.csv"
    ppc_shadow_selector_csv = results / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv"
    probe_dir = results / "gpu_shadow_selector_probe"
    real_probe_dir = results / "gpu_shadow_selector_probe_real_candidates_tokyo_run1"
    real_sweep_dir = results / "gpu_shadow_selector_probe_real_candidates_tokyo_run1_sweep"

    _write_csv(
        rf_dir / "gnss_security_lab_summary.csv",
        [
            {
                "scenario": "clean",
                "acquisition_backend": "cuda_acquisition",
                "target_snr": "72",
                "false_lock": "False",
                "interference_detected": "False",
            },
            {
                "scenario": "delayed_replica",
                "acquisition_backend": "cuda_acquisition",
                "target_snr": "5",
                "false_lock": "True",
                "interference_detected": "False",
            },
        ],
    )
    (rf_dir / "gnss_security_lab_summary.json").write_text("{}", encoding="utf-8")
    (urban_dir / "urban_shadow_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (urban_dir / "urban_shadow_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "particle_backend": "cuda_bvh_batch",
                    "particle_rays": 100,
                    "particle_los_ms": 0.5,
                    "mean_blocked_ratio": 0.4,
                    "max_blocked_ratio": 0.8,
                    "mean_particle_shadow_contrast": 0.1,
                    "route_los_ms": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_csv(urban_dir / "urban_shadow_epoch_summary.csv", [{"epoch": "0", "blocked_ratio": "0.4"}])
    (sweep_dir / "gpu_scenario_sweep_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "gpu_scenario_sweep_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "n_rows": 2,
                    "mean_risk": 0.5,
                    "max_risk": 0.9,
                    "label_counts": {"nominal": 1, "false_lock": 1},
                    "worst": {
                        "failure_label": "false_lock",
                        "total_risk_score": 0.9,
                        "jammer_jnr_db": 16,
                        "spoof_delay_samples": 64,
                        "code_phase_error_samples": 64,
                        "mean_blocked_ratio": 0.4,
                        "acquisition_backend": "cuda_acquisition",
                        "particle_backend": "cuda_bvh_batch",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_csv(sweep_dir / "gpu_scenario_sweep_summary.csv", [{"failure_label": "false_lock"}])
    _write_csv(
        feature_csv,
        [
            {
                "run_id": "gpu",
                "tow": "0",
                "label": "a",
                "gpu_stress_scenario_id": "abc",
                "gpu_failure_label": "nominal",
                "gpu_stress_target_bad": "0",
                "gpu_total_risk_score": "0.1",
            },
            {
                "run_id": "gpu",
                "tow": "1",
                "label": "b",
                "gpu_stress_scenario_id": "def",
                "gpu_failure_label": "false_lock",
                "gpu_stress_target_bad": "1",
                "gpu_total_risk_score": "0.9",
            },
        ],
    )
    _write_csv(
        ppc_shadow_csv,
        [
            {
                "run_id": "tokyo_run1_nav",
                "tow": "187470.0",
                "gpu_urban_backend": "cuda_bvh_batch",
                "gpu_urban_satellite_source": "nav_rinex",
                "gpu_urban_n_sat": "10",
                "gpu_urban_mean_blocked_ratio": "0.1",
                "gpu_urban_max_blocked_ratio": "0.2",
            }
        ],
    )
    _write_csv(
        ppc_shadow_selector_csv,
        [
            {
                "run_id": "tokyo_run1_nav",
                "tow": "187470.0",
                "gpu_urban_shadow_risk_score": "0.15",
            }
        ],
    )
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "gpu_shadow_selector_probe_summary.json").write_text(
        json.dumps(
            {
                "epochs": 1,
                "changed_epochs": 1,
                "change_rate": 1.0,
                "mean_gpu_minus_baseline_truth_cost": -0.05,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        probe_dir / "gpu_shadow_selector_probe_rows.csv",
        [{"run_id": "tokyo_run1_nav", "tow": "187470.0", "selection_changed": "1"}],
    )
    _write_csv(
        probe_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv",
        [{"risk_bucket": "0.20+", "epochs": "1", "changed": "1"}],
    )
    real_probe_dir.mkdir(parents=True, exist_ok=True)
    (real_probe_dir / "gpu_shadow_selector_probe_summary.json").write_text(
        json.dumps(
            {
                "epochs": 1,
                "source_rows": 2,
                "changed_epochs": 1,
                "change_rate": 1.0,
                "mean_gpu_minus_baseline_truth_cost": -0.01,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        real_probe_dir / "gpu_shadow_selector_probe_rows.csv",
        [{"run_id": "tokyo_run1", "tow": "187470.0", "selection_changed": "1"}],
    )
    _write_csv(
        real_probe_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv",
        [{"risk_bucket": "0.20+", "epochs": "1", "changed": "1"}],
    )
    real_sweep_dir.mkdir(parents=True, exist_ok=True)
    (real_sweep_dir / "gpu_shadow_selector_probe_weight_sweep_summary.json").write_text(
        json.dumps({"grid_rows": 1, "best": {"mean_gpu_minus_baseline_truth_cost": -0.01}}),
        encoding="utf-8",
    )
    _write_csv(
        real_sweep_dir / "gpu_shadow_selector_probe_weight_sweep.csv",
        [{"penalty_weight": "0", "rescue_weight": "1", "mean_gpu_minus_baseline_truth_cost": "-0.01"}],
    )

    monkeypatch.setattr(_SITE, "RESULTS_DIR", results)
    monkeypatch.setattr(_SITE, "DOCS_DIR", docs)
    monkeypatch.setattr(_SITE, "ASSETS_DIR", docs / "assets")
    monkeypatch.setattr(_SITE, "DATA_DIR", docs / "assets" / "data")
    monkeypatch.setattr(_SITE, "SNAPSHOT_JSON", docs / "assets" / "gpu_stress_lab_snapshot.json")
    monkeypatch.setattr(_SITE, "SNAPSHOT_JS", docs / "assets" / "gpu_stress_lab_snapshot.js")
    monkeypatch.setattr(_SITE, "PAGE_PATH", docs / "gpu_gnss_stress_lab.html")
    monkeypatch.setattr(_SITE, "RF_SUMMARY_CSV", rf_dir / "gnss_security_lab_summary.csv")
    monkeypatch.setattr(_SITE, "RF_SUMMARY_JSON", rf_dir / "gnss_security_lab_summary.json")
    monkeypatch.setattr(_SITE, "URBAN_SUMMARY_JSON", urban_dir / "urban_shadow_summary.json")
    monkeypatch.setattr(_SITE, "URBAN_EPOCH_CSV", urban_dir / "urban_shadow_epoch_summary.csv")
    monkeypatch.setattr(_SITE, "SWEEP_SUMMARY_JSON", sweep_dir / "gpu_scenario_sweep_summary.json")
    monkeypatch.setattr(_SITE, "SWEEP_SUMMARY_CSV", sweep_dir / "gpu_scenario_sweep_summary.csv")
    monkeypatch.setattr(_SITE, "GPU_FEATURE_CSV", feature_csv)
    monkeypatch.setattr(_SITE, "PPC_SHADOW_CSV", ppc_shadow_csv)
    monkeypatch.setattr(_SITE, "PPC_SHADOW_SELECTOR_CSV", ppc_shadow_selector_csv)
    monkeypatch.setattr(_SITE, "GPU_SHADOW_PROBE_SUMMARY_JSON", probe_dir / "gpu_shadow_selector_probe_summary.json")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_PROBE_ROWS_CSV", probe_dir / "gpu_shadow_selector_probe_rows.csv")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_PROBE_BUCKET_CSV", probe_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_REAL_PROBE_SUMMARY_JSON", real_probe_dir / "gpu_shadow_selector_probe_summary.json")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_REAL_PROBE_ROWS_CSV", real_probe_dir / "gpu_shadow_selector_probe_rows.csv")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_REAL_PROBE_BUCKET_CSV", real_probe_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_REAL_SWEEP_SUMMARY_JSON", real_sweep_dir / "gpu_shadow_selector_probe_weight_sweep_summary.json")
    monkeypatch.setattr(_SITE, "GPU_SHADOW_REAL_SWEEP_CSV", real_sweep_dir / "gpu_shadow_selector_probe_weight_sweep.csv")

    snapshot = _SITE.build()

    assert snapshot["rf"]["false_lock_count"] == 1
    assert snapshot["sweep"]["n_rows"] == 2
    assert snapshot["features"]["target_bad"] == 1
    assert snapshot["ppc_shadow"]["n_epochs"] == 1
    assert snapshot["ppc_shadow"]["mean_shadow_risk"] == 0.15
    assert snapshot["selector_probe"]["changed_epochs"] == 1
    assert snapshot["real_selector_probe"]["source_rows"] == 2
    assert snapshot["real_selector_sweep"]["best"]["mean_gpu_minus_baseline_truth_cost"] == -0.01
    assert (docs / "gpu_gnss_stress_lab.html").exists()
    assert (docs / "assets" / "gpu_stress_lab_snapshot.js").exists()
    assert (docs / "assets" / "data" / "gpu_stress_selector_features.csv").exists()
    assert (docs / "assets" / "data" / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv").exists()
    assert (docs / "assets" / "data" / "gpu_shadow_selector_probe_summary.json").exists()
    assert (docs / "assets" / "data" / "real_gpu_shadow_selector_probe_summary.json").exists()
    assert (docs / "assets" / "data" / "real_gpu_shadow_selector_probe_weight_sweep_summary.json").exists()
