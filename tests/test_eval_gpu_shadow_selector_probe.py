import argparse
import csv
import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "eval_gpu_shadow_selector_probe.py"
_SPEC = importlib.util.spec_from_file_location("eval_gpu_shadow_selector_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_synthesize_shadow_candidate_rows_creates_two_options_per_epoch():
    rows = [
        {"run_id": "r", "tow": "1.0", "gpu_urban_shadow_risk_score": "0.01"},
        {"run_id": "r", "tow": "2.0", "gpu_urban_shadow_risk_score": "0.20"},
    ]

    candidates = _PROBE.synthesize_shadow_candidate_rows(rows)

    assert len(candidates) == 4
    assert [row["label"] for row in candidates[:2]] == ["nominal_gnss", "shadow_robust"]
    assert candidates[0]["selector_base_score"] == 0.0
    assert candidates[1]["selector_base_score"] > 0.0
    assert candidates[1]["proxy_truth_cost"] < candidates[0]["proxy_truth_cost"] + 0.06


def test_evaluate_rows_flips_high_risk_candidate_and_reports_proxy_gain():
    candidates = _PROBE.synthesize_shadow_candidate_rows(
        [
            {"run_id": "r", "tow": "1.0", "gpu_urban_shadow_risk_score": "0.01"},
            {"run_id": "r", "tow": "2.0", "gpu_urban_shadow_risk_score": "0.20"},
        ],
    )

    rows, summary, buckets = _PROBE.evaluate_rows(
        candidates,
        truth_cost_col="proxy_truth_cost",
    )

    assert len(rows) == 2
    assert rows[0]["baseline_label"] == "nominal_gnss"
    assert rows[0]["gpu_label"] == "nominal_gnss"
    assert rows[1]["baseline_label"] == "nominal_gnss"
    assert rows[1]["gpu_label"] == "shadow_robust"
    assert summary["changed_epochs"] == 1
    assert summary["improved_epochs"] == 1
    assert summary["mean_gpu_minus_baseline_truth_cost"] < 0.0
    assert {row["risk_bucket"] for row in buckets} == {"0.00-0.02", "0.20+"}


def test_evaluate_rows_accepts_real_candidate_level_scores():
    rows = [
        {
            "run_id": "r",
            "tow": "1.0",
            "label": "fast",
            "selector_base_score": "0.0",
            "gpu_urban_shadow_risk_score": "0.2",
            "gpu_shadow_penalty_coeff": "1.0",
            "gpu_shadow_rescue_coeff": "0.0",
            "truth_distance_m": "5.0",
        },
        {
            "run_id": "r",
            "tow": "1.0",
            "label": "robust",
            "selector_base_score": "0.08",
            "gpu_urban_shadow_risk_score": "0.2",
            "gpu_shadow_penalty_coeff": "0.0",
            "gpu_shadow_rescue_coeff": "0.3",
            "truth_distance_m": "2.0",
        },
    ]

    selected, summary, _buckets = _PROBE.evaluate_rows(rows, truth_cost_col="truth_distance_m")

    assert selected[0]["baseline_label"] == "fast"
    assert selected[0]["gpu_label"] == "robust"
    assert summary["changed_epochs"] == 1
    assert summary["mean_gpu_minus_baseline_truth_cost"] == -3.0


def test_run_probe_writes_outputs_for_epoch_feature_csv(tmp_path):
    input_csv = tmp_path / "features.csv"
    _write_csv(
        input_csv,
        [
            {"run_id": "r", "tow": "1.0", "gpu_urban_shadow_risk_score": "0.01"},
            {"run_id": "r", "tow": "2.0", "gpu_urban_shadow_risk_score": "0.20"},
        ],
    )
    out_dir = tmp_path / "out"

    result = _PROBE.run_probe(
        argparse.Namespace(
            input_csv=input_csv,
            out_dir=out_dir,
            input_mode="auto",
            risk_col="gpu_urban_shadow_risk_score",
            base_score_col="selector_base_score",
            truth_cost_col="",
            penalty_weight=1.0,
            rescue_weight=1.0,
            robust_base_penalty=0.06,
            robust_score_gain=0.80,
        )
    )

    assert result["summary"]["synthesized_candidates"] is True
    assert result["summary"]["candidate_rows"] == 4
    assert Path(result["summary_json"]).exists()
    assert Path(result["rows_csv"]).exists()
    assert (out_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv").exists()


def test_run_probe_can_join_gpu_features_and_derive_real_candidate_coeffs(tmp_path):
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
                "dist_to_median_m": "4",
                "candidate_jump_m": "6",
                "delta_pos_vertical_m": "2",
                "abs_max": "5",
                "sats": "6",
            },
            {
                "run_id": "tokyo_run1",
                "tow": "10.0",
                "label": "clustered",
                "rms": "0.02",
                "err_3d_m": "1.0",
                "cluster_size_50cm": "2",
                "n_candidates_in_epoch": "2",
                "is_in_max_cluster_50cm": "1",
                "rank_by_rms": "2",
                "dist_to_median_m": "0.1",
                "candidate_jump_m": "0.1",
                "delta_pos_vertical_m": "0.1",
                "abs_max": "1",
                "sats": "12",
            },
        ],
    )
    _write_csv(
        features,
        [
            {
                "run_id": "tokyo_run1_nav",
                "tow": "10.0",
                "gpu_urban_shadow_risk_score": "0.2",
            }
        ],
    )

    result = _PROBE.run_probe(
        argparse.Namespace(
            input_csv=candidates,
            out_dir=tmp_path / "out",
            input_mode="candidates",
            candidate_run_id="tokyo_run1",
            gpu_feature_csv=features,
            feature_source_run_id="tokyo_run1_nav",
            feature_target_run_id="tokyo_run1",
            keep_only_feature_epochs=True,
            derive_shadow_coeffs=True,
            risk_col="gpu_urban_shadow_risk_score",
            base_score_col="rms",
            truth_cost_col="err_3d_m",
            penalty_weight=1.0,
            rescue_weight=1.0,
            robust_base_penalty=0.06,
            robust_score_gain=0.80,
        )
    )

    assert result["summary"]["synthesized_candidates"] is False
    assert result["summary"]["feature_matched_rows"] == 2
    assert result["summary"]["changed_epochs"] == 1
    assert result["summary"]["mean_gpu_minus_baseline_truth_cost"] == -4.0
