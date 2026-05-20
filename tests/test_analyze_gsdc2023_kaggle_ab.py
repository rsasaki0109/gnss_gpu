"""Integration test for the kaggle A/B CLI runner.

The pure-logic modules are covered by their own unit tests.  This test ensures
the orchestration wires them together correctly: small synthetic inputs,
outputs verified to be on disk and self-consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.analyze_gsdc2023_kaggle_ab import run_kaggle_ab_audit
from experiments.gsdc2023_ab_gates import DDAnchorGate, NoTdcpCoexistGate


def _write_bridge_metrics(root: Path, trip: str, phone: str, payload: dict) -> None:
    path = root / trip / phone
    path.mkdir(parents=True, exist_ok=True)
    (path / "bridge_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    bridge_root = tmp_path / "bridge"
    _write_bridge_metrics(
        bridge_root,
        "2021-09-14-20-32-us-ca-mtv-k",
        "pixel4",
        {"n_epochs": 200, "selected_source_counts": {"baseline": 200}},
    )
    _write_bridge_metrics(
        bridge_root,
        "2022-04-04-16-31-us-ca-lax-x",
        "pixel5",
        {"n_epochs": 200, "selected_source_counts": {"baseline": 200}},
    )
    _write_bridge_metrics(
        bridge_root,
        "2022-02-23-22-35-us-ca-lax-m",
        "pixel5",
        {"n_epochs": 200, "selected_source_counts": {"baseline": 200}},
    )

    summary = {
        "trip_metrics": [
            {
                # Bad: low DD anchor coverage -> reverted
                "trip": "test/2021-09-14-20-32-us-ca-mtv-k/pixel4",
                "n_epochs": 200,
                "dd_carrier_accepted_anchor_epochs": 50,
                "dd_carrier_dd_epochs": 70,
                "dd_carrier_base_snapped_epochs": 40,
                "dd_carrier_dd_pairs_mean": 4.0,
                "selected_source_counts": {"baseline": 100, "fgo_dd_carrier": 100},
                "chunk_selection_records": [
                    {
                        "gated_source": "fgo_dd_carrier",
                        "start_epoch": 0,
                        "end_epoch": 100,
                        "candidates": {
                            "fgo_no_tdcp": {
                                "quality_score": 0.7,
                                "mse_pr": 8.0,
                                "baseline_gap_max_m": 12.0,
                                "step_p95_m": 20.0,
                                "accel_p95_m": 1.5,
                            }
                        },
                    }
                ],
            },
            {
                # Bad: DD + no_tdcp coexist (still reverted under combined gate)
                "trip": "test/2022-04-04-16-31-us-ca-lax-x/pixel5",
                "n_epochs": 200,
                "dd_carrier_accepted_anchor_epochs": 180,
                "dd_carrier_dd_epochs": 180,
                "dd_carrier_base_snapped_epochs": 100,
                "dd_carrier_dd_pairs_mean": 5.0,
                "selected_source_counts": {
                    "baseline": 0,
                    "fgo_dd_carrier": 100,
                    "fgo_no_tdcp": 100,
                },
                "chunk_selection_records": [
                    {
                        "gated_source": "fgo_no_tdcp",
                        "start_epoch": 100,
                        "end_epoch": 200,
                        "candidates": {
                            "fgo_no_tdcp": {
                                "quality_score": 0.5,
                                "mse_pr": 6.0,
                                "baseline_gap_max_m": 9.0,
                                "step_p95_m": 18.0,
                                "accel_p95_m": 1.0,
                            }
                        },
                    }
                ],
            },
            {
                # Good: high DD anchor, no no_tdcp -> kept
                "trip": "test/2022-02-23-22-35-us-ca-lax-m/pixel5",
                "n_epochs": 200,
                "dd_carrier_accepted_anchor_epochs": 160,
                "dd_carrier_dd_epochs": 160,
                "dd_carrier_base_snapped_epochs": 100,
                "dd_carrier_dd_pairs_mean": 5.5,
                "selected_source_counts": {"baseline": 100, "fgo_dd_carrier": 100},
                "chunk_selection_records": [],
            },
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # Row delta: each trip moves a few rows by 5m
    rows = []
    for trip in (
        "2021-09-14-20-32-us-ca-mtv-k/pixel4",
        "2022-04-04-16-31-us-ca-lax-x/pixel5",
        "2022-02-23-22-35-us-ca-lax-m/pixel5",
    ):
        for _ in range(3):
            rows.append((trip, 5.0))
    row_delta_path = tmp_path / "row_deltas.csv"
    pd.DataFrame(rows, columns=["tripId", "delta_m"]).to_csv(row_delta_path, index=False)

    return summary_path, bridge_root, row_delta_path


def test_run_kaggle_ab_audit_produces_expected_outputs(tmp_path: Path):
    summary_path, bridge_root, row_delta_path = _build_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "out"

    summary = run_kaggle_ab_audit(
        taroz_summary_json=summary_path,
        bridge_metrics_root=bridge_root,
        row_delta_csv=row_delta_path,
        output_dir=out_dir,
        dd_gate=DDAnchorGate(min_anchor_coverage=0.6),
        ntdc_gate=NoTdcpCoexistGate(min_anchor_coverage=0.6),
    )

    # All four CSVs + summary.json on disk
    for name in (
        "per_trip_source_count_diff.csv",
        "dd_signals.csv",
        "no_tdcp_promoted_chunks.csv",
        "combined_gate_disposition.csv",
        "row_deltas_with_gate_sim.csv",
        "summary.json",
    ):
        assert (out_dir / name).is_file(), f"missing output: {name}"

    # Disposition: 2 reverted (mtv-k low anchor, lax-x DD+ntdc combo), 1 kept (lax-m)
    assert summary["disposition_counts"]["reverted"] == 2
    assert summary["disposition_counts"]["kept"] == 1

    # The sim should zero out the 2 reverted trips' 3 rows each (6 rows) and
    # keep the 3 rows from lax-m (kept).
    sim_csv = pd.read_csv(out_dir / "row_deltas_with_gate_sim.csv")
    assert (sim_csv["sim_delta_m"] > 0).sum() == 3
    assert summary["row_delta_original"]["rows_gt_5m"] == 0  # all = 5.0, strict gt
    assert summary["row_delta_simulated"]["sum_m"] == 15.0  # 3 rows * 5m


def test_run_kaggle_ab_audit_respects_disabled_coexistence_guards(tmp_path: Path):
    summary_path, bridge_root, row_delta_path = _build_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "out"

    # Disable the DD's no_tdcp coexistence guard -> lax-x (DD+ntdc) becomes kept
    summary = run_kaggle_ab_audit(
        taroz_summary_json=summary_path,
        bridge_metrics_root=bridge_root,
        row_delta_csv=row_delta_path,
        output_dir=out_dir,
        dd_gate=DDAnchorGate(min_anchor_coverage=0.6, require_no_tdcp_absent=False),
        ntdc_gate=NoTdcpCoexistGate(min_anchor_coverage=0.6),
    )
    assert summary["disposition_counts"]["kept"] == 2
    assert summary["disposition_counts"]["reverted"] == 1


def _candidate_payload(*, mse_pr: float, step_p95: float = 6.0, gap_p95: float = 4.0) -> dict:
    return {
        "mse_pr": mse_pr,
        "step_mean_m": 3.0,
        "step_p95_m": step_p95,
        "accel_mean_m": 0.8,
        "accel_p95_m": 1.5,
        "bridge_jump_m": 0.0,
        "baseline_gap_mean_m": gap_p95 / 2.0,
        "baseline_gap_p95_m": gap_p95,
        "baseline_gap_max_m": gap_p95 * 1.5,
        "quality_score": 0.7,
    }


def _build_chunk_merge_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Synthetic inputs with chunk_selection_records on both sides.

    One trip has a single 200-epoch bridge chunk that gated to ``fgo``
    (low fgo MSE) plus two 100-epoch TaroZ sub-chunks that both gate to
    ``baseline`` because of slightly elevated MSE.  This mirrors the
    Phase 76 sjc-q failure pattern at a tiny scale.
    """

    bridge_root = tmp_path / "bridge"

    # Bridge: one chunk, 0..200, gated=fgo (low fgo_mse vs higher baseline_mse)
    bridge_trip_dir = bridge_root / "trip_a" / "pixel5"
    bridge_trip_dir.mkdir(parents=True)
    bridge_metrics = {
        "n_epochs": 200,
        "selected_source_counts": {"baseline": 0, "fgo": 200},
        "chunk_selection_records": [
            {
                "start_epoch": 0,
                "end_epoch": 200,
                "auto_source": "fgo",
                "gated_source": "fgo",
                "candidates": {
                    "baseline": _candidate_payload(mse_pr=30.0, step_p95=10.0, gap_p95=0.0),
                    "fgo": _candidate_payload(mse_pr=10.0, step_p95=6.0, gap_p95=5.0),
                },
            }
        ],
    }
    (bridge_trip_dir / "bridge_metrics.json").write_text(json.dumps(bridge_metrics), encoding="utf-8")

    # TaroZ: same trip, two 100-epoch sub-chunks; pairwise merge should
    # reconstruct a 200-epoch chunk whose fgo candidate again wins the gate.
    summary = {
        "trip_metrics": [
            {
                "trip": "test/trip_a/pixel5",
                "n_epochs": 200,
                "dd_carrier_accepted_anchor_epochs": 160,
                "dd_carrier_dd_epochs": 160,
                "dd_carrier_base_snapped_epochs": 100,
                "dd_carrier_dd_pairs_mean": 5.0,
                "selected_source_counts": {"baseline": 200},
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 100,
                        "auto_source": "fgo",
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": _candidate_payload(mse_pr=30.0),
                            "fgo": _candidate_payload(mse_pr=10.0),
                        },
                    },
                    {
                        "start_epoch": 100,
                        "end_epoch": 200,
                        "auto_source": "fgo",
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": _candidate_payload(mse_pr=30.0),
                            "fgo": _candidate_payload(mse_pr=10.0),
                        },
                    },
                ],
            }
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    row_delta_path = tmp_path / "row_deltas.csv"
    pd.DataFrame([("trip_a/pixel5", 0.0)], columns=["tripId", "delta_m"]).to_csv(row_delta_path, index=False)

    return summary_path, bridge_root, row_delta_path


def test_run_kaggle_ab_audit_chunk_merge_verify_produces_outputs(tmp_path: Path):
    summary_path, bridge_root, row_delta_path = _build_chunk_merge_inputs(tmp_path)
    out_dir = tmp_path / "out"

    summary = run_kaggle_ab_audit(
        taroz_summary_json=summary_path,
        bridge_metrics_root=bridge_root,
        row_delta_csv=row_delta_path,
        output_dir=out_dir,
        chunk_merge_verify=True,
    )

    # Chunk-merge CSVs land on disk.
    assert (out_dir / "chunk_merge_verification_per_chunk.csv").is_file()
    assert (out_dir / "chunk_merge_verification_per_trip.csv").is_file()

    # Summary embeds the verification block.
    cmv = summary["chunk_merge_verification"]
    assert cmv["trips_compared"] == 1
    assert cmv["total_rows"] == 200
    assert 0.0 <= cmv["match_pct"] <= 1.0
    # Either zero mismatches or a documented mismatch class.
    assert all(
        k in {"dd_or_raw_enable_diff", "aggregation_approximation_noise", "other"}
        for k in cmv["mismatch_classes"]
    )


def test_run_kaggle_ab_audit_skips_chunk_merge_outputs_by_default(tmp_path: Path):
    summary_path, bridge_root, row_delta_path = _build_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "out"

    summary = run_kaggle_ab_audit(
        taroz_summary_json=summary_path,
        bridge_metrics_root=bridge_root,
        row_delta_csv=row_delta_path,
        output_dir=out_dir,
    )

    assert "chunk_merge_verification" not in summary
    assert not (out_dir / "chunk_merge_verification_per_chunk.csv").exists()
    assert not (out_dir / "chunk_merge_verification_per_trip.csv").exists()
