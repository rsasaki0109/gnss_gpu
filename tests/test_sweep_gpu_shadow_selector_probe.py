import argparse
import csv
import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "sweep_gpu_shadow_selector_probe.py"
_SPEC = importlib.util.spec_from_file_location("sweep_gpu_shadow_selector_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SWEEP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SWEEP
_SPEC.loader.exec_module(_SWEEP)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_sweep_rows_selects_weight_with_best_truth_delta():
    rows = [
        {
            "run_id": "r",
            "tow": "1.0",
            "label": "fragile",
            "rms": "0.01",
            "err_3d_m": "5.0",
            "gpu_urban_shadow_risk_score": "0.2",
            "gpu_shadow_penalty_coeff": "1.0",
            "gpu_shadow_rescue_coeff": "0.0",
        },
        {
            "run_id": "r",
            "tow": "1.0",
            "label": "robust",
            "rms": "0.02",
            "err_3d_m": "1.0",
            "gpu_urban_shadow_risk_score": "0.2",
            "gpu_shadow_penalty_coeff": "0.0",
            "gpu_shadow_rescue_coeff": "1.0",
        },
    ]

    sweep, summary = _SWEEP.sweep_rows(
        rows,
        risk_col="gpu_urban_shadow_risk_score",
        base_score_col="rms",
        truth_cost_col="err_3d_m",
        penalty_weights=[0.0, 1.0],
        rescue_weights=[0.0, 1.0],
    )

    assert len(sweep) == 4
    assert summary["best"]["penalty_weight"] in {0.0, 1.0}
    assert summary["best"]["rescue_weight"] == 1.0
    assert summary["best"]["mean_gpu_minus_baseline_truth_cost"] == -4.0


def test_run_sweep_writes_grid_outputs(tmp_path):
    candidates = tmp_path / "candidates.csv"
    features = tmp_path / "features.csv"
    _write_csv(
        candidates,
        [
            {
                "run_id": "tokyo_run1",
                "tow": "10.0",
                "label": "fragile",
                "rms": "0.01",
                "err_3d_m": "5.0",
                "cluster_size_50cm": "1",
                "n_candidates_in_epoch": "2",
                "is_in_max_cluster_50cm": "0",
                "rank_by_rms": "1",
            },
            {
                "run_id": "tokyo_run1",
                "tow": "10.0",
                "label": "robust",
                "rms": "0.02",
                "err_3d_m": "1.0",
                "cluster_size_50cm": "2",
                "n_candidates_in_epoch": "2",
                "is_in_max_cluster_50cm": "1",
                "rank_by_rms": "2",
            },
        ],
    )
    _write_csv(
        features,
        [{"run_id": "tokyo_run1_nav", "tow": "10.0", "gpu_urban_shadow_risk_score": "0.2"}],
    )

    result = _SWEEP.run_sweep(
        argparse.Namespace(
            input_csv=candidates,
            gpu_feature_csv=features,
            out_dir=tmp_path / "out",
            input_mode="candidates",
            candidate_run_id="tokyo_run1",
            feature_source_run_id="tokyo_run1_nav",
            feature_target_run_id="tokyo_run1",
            keep_only_feature_epochs=True,
            derive_shadow_coeffs=True,
            risk_col="gpu_urban_shadow_risk_score",
            base_score_col="rms",
            truth_cost_col="err_3d_m",
            penalty_weights="0,1",
            rescue_weights="0,1",
            robust_base_penalty=0.06,
            robust_score_gain=0.80,
        )
    )

    assert result["summary"]["grid_rows"] == 4
    assert result["summary"]["feature_matched_rows"] == 2
    assert Path(result["sweep_csv"]).exists()
    assert Path(result["summary_json"]).exists()
