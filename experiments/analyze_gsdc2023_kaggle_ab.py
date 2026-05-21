"""CLI wrapper for the GSDC2023 Kaggle bridge-vs-candidate A/B audit.

Given:

- the candidate submission's ``summary.json`` (TaroZ DD-guarded, etc.)
- the bridge submission's per-trip ``bridge_metrics.json`` directory tree
- the row-level haversine delta CSV produced by ``analyze_gsdc2023_source_ab``

this script combines the modular dataclasses in :mod:`gsdc2023_ab_source_mix`,
:mod:`gsdc2023_ab_dd_signals`, :mod:`gsdc2023_ab_ntdc_chunks`,
:mod:`gsdc2023_ab_gates`, :mod:`gsdc2023_ab_revert`, and (optionally)
:mod:`gsdc2023_ab_chunk_merge` to reproduce the Phase 73-78 audit outputs.
All heavy lifting lives in the modules; this file is intentionally a thin
orchestration layer so it can be reviewed and modified without touching the
tested core logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from experiments.gsdc2023_ab_chunk_merge import (
    compare_bridge_to_merged_candidate,
    load_bridge_chunk_records,
    load_chunk_records_from_summary,
    predict_merged_gated_sources,
)
from experiments.gsdc2023_ab_dd_signals import load_dd_signals_from_summary
from experiments.gsdc2023_ab_gates import (
    CombinedGate,
    DDAnchorGate,
    NoTdcpCoexistGate,
    apply_gate,
    disposition_counts,
)
from experiments.gsdc2023_ab_ntdc_chunks import load_promoted_ntdc_chunks
from experiments.gsdc2023_ab_revert import (
    compute_delta_stats,
    per_trip_shift_proxy,
    simulate_revert,
)
from experiments.gsdc2023_ab_source_mix import (
    aggregate_source_totals,
    diff_source_counts,
    load_bridge_source_counts,
    load_taroz_source_counts,
)


def _write_dataclasses_csv(path: Path, rows: list, columns: list[str]) -> None:
    """Write a list of ``@dataclass(frozen=True)`` rows to CSV."""

    payload = [{col: getattr(row, col) for col in columns} for row in rows]
    pd.DataFrame(payload, columns=columns).to_csv(path, index=False)


def _write_source_diff_csv(path: Path, diffs: list) -> None:
    rows = []
    for d in diffs:
        rows.append(
            {
                "tripId": d.trip_id,
                "n_epochs_bridge": d.n_epochs_bridge,
                "n_epochs_taroz": d.n_epochs_taroz,
                "b_baseline": d.bridge.baseline,
                "b_fgo": d.bridge.fgo,
                "b_fgo_dd_carrier": d.bridge.fgo_dd_carrier,
                "b_fgo_no_tdcp": d.bridge.fgo_no_tdcp,
                "b_raw_wls": d.bridge.raw_wls,
                "t_baseline": d.taroz.baseline,
                "t_fgo": d.taroz.fgo,
                "t_fgo_dd_carrier": d.taroz.fgo_dd_carrier,
                "t_fgo_no_tdcp": d.taroz.fgo_no_tdcp,
                "t_raw_wls": d.taroz.raw_wls,
                "d_baseline": d.d_baseline,
                "d_fgo": d.d_fgo,
                "d_fgo_dd_carrier": d.d_fgo_dd_carrier,
                "d_fgo_no_tdcp": d.d_fgo_no_tdcp,
                "d_raw_wls": d.d_raw_wls,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_dispositions_csv(path: Path, dispositions: list) -> None:
    pd.DataFrame(
        [
            {
                "tripId": d.trip_id,
                "disposition": d.disposition.value,
                "dd_pass": int(d.dd_pass),
                "ntdc_pass": int(d.ntdc_pass),
            }
            for d in dispositions
        ]
    ).to_csv(path, index=False)


def _classify_chunk_merge_mismatch(bridge: str, predicted: str) -> str:
    """Bucket a per-chunk mismatch so the summary can quantify root causes."""

    if bridge == "baseline" and predicted in ("fgo_dd_carrier", "raw_wls"):
        return "dd_or_raw_enable_diff"
    if bridge in ("fgo", "fgo_no_tdcp") and predicted == "baseline":
        return "aggregation_approximation_noise"
    return "other"


def run_chunk_merge_verification(
    *,
    taroz_summary_json: Path,
    bridge_metrics_root: Path,
    output_dir: Path,
) -> dict:
    """Predict 200-epoch ``gated_source`` from TaroZ 100-epoch sub-chunks.

    For each trip, ``gsdc2023_ab_chunk_merge`` is used to merge adjacent
    sub-chunks pairwise and re-evaluate the production gate on the merged
    chunks.  The predictions are compared against the bridge's actual per-
    epoch ``gated_source`` and a per-chunk / per-trip CSV is written.

    The returned dict is a summary suitable to embed in the orchestration's
    ``summary.json`` next to the gate audit results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taroz_records = load_chunk_records_from_summary(Path(taroz_summary_json))
    bridge_records = load_bridge_chunk_records(Path(bridge_metrics_root))

    per_chunk_rows: list[dict] = []
    per_trip_rows: list[dict] = []

    for trip_id, t in taroz_records.items():
        b = bridge_records.get(trip_id)
        if b is None:
            continue
        predictions = predict_merged_gated_sources(t["chunks"])
        comparisons = compare_bridge_to_merged_candidate(
            trip_id=trip_id,
            n_epochs=t["n_epochs"],
            bridge_chunks=b["chunks"],
            merged_predictions=predictions,
        )
        if not comparisons:
            continue
        total_rows = sum(c.n_rows for c in comparisons)
        match_rows = sum(c.n_rows for c in comparisons if c.matches)
        per_trip_rows.append(
            {
                "tripId": trip_id,
                "bridge_chunks": len(b["chunks"]),
                "taroz_chunks": len(t["chunks"]),
                "merged_chunks": len(comparisons),
                "match_rows": match_rows,
                "total_rows": total_rows,
                "match_pct": match_rows / total_rows if total_rows else 0.0,
            }
        )
        for c in comparisons:
            per_chunk_rows.append(
                {
                    "tripId": c.trip_id,
                    "start_epoch": c.start_epoch,
                    "end_epoch": c.end_epoch,
                    "n_rows": c.n_rows,
                    "bridge_gated": c.bridge_gated_source,
                    "merged_pred": c.candidate_merged_gated_source,
                    "matches": c.matches,
                    "mismatch_class": (
                        ""
                        if c.matches
                        else _classify_chunk_merge_mismatch(c.bridge_gated_source, c.candidate_merged_gated_source)
                    ),
                }
            )

    per_chunk_df = pd.DataFrame(per_chunk_rows)
    per_trip_df = pd.DataFrame(per_trip_rows)
    per_chunk_df.to_csv(output_dir / "chunk_merge_verification_per_chunk.csv", index=False)
    per_trip_df.to_csv(output_dir / "chunk_merge_verification_per_trip.csv", index=False)

    if per_chunk_df.empty:
        return {
            "total_rows": 0,
            "match_rows": 0,
            "match_pct": 0.0,
            "mismatch_classes": {},
            "trips_compared": 0,
        }

    total_rows = int(per_chunk_df["n_rows"].sum())
    match_rows = int(per_chunk_df.loc[per_chunk_df["matches"], "n_rows"].sum())
    mismatch = per_chunk_df.loc[~per_chunk_df["matches"]]
    mismatch_classes: dict[str, dict[str, int]] = {}
    if not mismatch.empty:
        agg = mismatch.groupby("mismatch_class").agg(
            chunks=("n_rows", "size"),
            rows=("n_rows", "sum"),
        )
        mismatch_classes = {
            str(k): {"chunks": int(v["chunks"]), "rows": int(v["rows"])}
            for k, v in agg.to_dict(orient="index").items()
        }

    return {
        "total_rows": total_rows,
        "match_rows": match_rows,
        "match_pct": float(match_rows) / float(total_rows) if total_rows else 0.0,
        "mismatch_classes": mismatch_classes,
        "trips_compared": int(len(per_trip_df)),
    }


def _summary_payload(
    *,
    bridge_path: Path,
    summary_path: Path,
    row_delta_path: Path,
    bridge_total: dict[str, int],
    taroz_total: dict[str, int],
    disposition_summary: dict[str, int],
    original_stats,
    simulated_stats,
    per_trip_proxy_original_mean: float,
    per_trip_proxy_simulated_mean: float,
    dd_gate: DDAnchorGate,
    ntdc_gate: NoTdcpCoexistGate,
) -> dict:
    return {
        "bridge_metrics_root": str(bridge_path),
        "taroz_summary_json": str(summary_path),
        "row_delta_csv": str(row_delta_path),
        "bridge_source_totals": bridge_total,
        "taroz_source_totals": taroz_total,
        "disposition_counts": disposition_summary,
        "row_delta_original": asdict(original_stats),
        "row_delta_simulated": asdict(simulated_stats),
        "per_trip_shift_proxy_mean_m": {
            "original": per_trip_proxy_original_mean,
            "simulated": per_trip_proxy_simulated_mean,
        },
        "gate_config": {
            "dd_anchor_gate": asdict(dd_gate),
            "no_tdcp_gate": asdict(ntdc_gate),
        },
    }


def run_kaggle_ab_audit(
    *,
    taroz_summary_json: Path,
    bridge_metrics_root: Path,
    row_delta_csv: Path,
    output_dir: Path,
    dd_gate: DDAnchorGate = DDAnchorGate(),
    ntdc_gate: NoTdcpCoexistGate = NoTdcpCoexistGate(),
    chunk_merge_verify: bool = False,
) -> dict:
    """Run the audit and write all CSV/JSON outputs.  Returns the summary dict.

    When ``chunk_merge_verify`` is true, the chunk-epochs hypothesis is also
    verified by analytically merging the candidate's sub-chunks back to
    ~200 epochs and replaying the production gate; results are written
    alongside the gate outputs and added to the returned summary under the
    ``chunk_merge_verification`` key.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_counts = load_bridge_source_counts(Path(bridge_metrics_root))
    taroz_counts = load_taroz_source_counts(Path(taroz_summary_json))
    dd_signals = load_dd_signals_from_summary(Path(taroz_summary_json))
    promoted_ntdc = load_promoted_ntdc_chunks(Path(taroz_summary_json))

    diffs = diff_source_counts(bridge_counts, taroz_counts)

    gate = CombinedGate(dd=dd_gate, ntdc=ntdc_gate)
    dispositions = apply_gate(gate, dd_signals, taroz_counts)

    row_delta = pd.read_csv(row_delta_csv)
    simulated = simulate_revert(row_delta, dispositions)

    original_stats = compute_delta_stats(simulated["delta_m"])
    simulated_stats = compute_delta_stats(simulated["sim_delta_m"])

    original_proxy = per_trip_shift_proxy(simulated, column="delta_m")
    simulated_proxy = per_trip_shift_proxy(simulated, column="sim_delta_m")

    # Files
    _write_source_diff_csv(output_dir / "per_trip_source_count_diff.csv", diffs)
    _write_dataclasses_csv(
        output_dir / "dd_signals.csv",
        dd_signals,
        columns=[
            "trip_id",
            "n_epochs",
            "dd_anchor_epochs",
            "dd_dd_epochs",
            "dd_base_snapped_epochs",
            "dd_pairs_mean",
        ],
    )
    _write_dataclasses_csv(
        output_dir / "no_tdcp_promoted_chunks.csv",
        promoted_ntdc,
        columns=[
            "trip_id",
            "start_epoch",
            "end_epoch",
            "ntdc_quality_score",
            "ntdc_mse_pr",
            "ntdc_baseline_gap_max_m",
            "ntdc_step_p95_m",
            "ntdc_accel_p95_m",
        ],
    )
    _write_dispositions_csv(output_dir / "combined_gate_disposition.csv", dispositions)
    simulated.to_csv(output_dir / "row_deltas_with_gate_sim.csv", index=False)

    summary = _summary_payload(
        bridge_path=Path(bridge_metrics_root),
        summary_path=Path(taroz_summary_json),
        row_delta_path=Path(row_delta_csv),
        bridge_total=aggregate_source_totals(bridge_counts),
        taroz_total=aggregate_source_totals(taroz_counts),
        disposition_summary=disposition_counts(dispositions),
        original_stats=original_stats,
        simulated_stats=simulated_stats,
        per_trip_proxy_original_mean=float(original_proxy.mean()),
        per_trip_proxy_simulated_mean=float(simulated_proxy.mean()),
        dd_gate=dd_gate,
        ntdc_gate=ntdc_gate,
    )

    if chunk_merge_verify:
        summary["chunk_merge_verification"] = run_chunk_merge_verification(
            taroz_summary_json=Path(taroz_summary_json),
            bridge_metrics_root=Path(bridge_metrics_root),
            output_dir=output_dir,
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taroz-summary", type=Path, required=True)
    parser.add_argument("--bridge-metrics-root", type=Path, required=True)
    parser.add_argument("--row-delta-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dd-min-anchor-coverage", type=float, default=0.6)
    parser.add_argument(
        "--dd-allow-no-tdcp-coexist",
        action="store_true",
        help="disable the DD gate's no_tdcp coexistence guard",
    )
    parser.add_argument("--ntdc-min-anchor-coverage", type=float, default=0.6)
    parser.add_argument(
        "--ntdc-allow-dd-coexist",
        action="store_true",
        help="disable the no_tdcp gate's DD coexistence guard",
    )
    parser.add_argument(
        "--chunk-merge-verify",
        action="store_true",
        help="also run the Phase 78 chunk_epochs hypothesis verification",
    )
    args = parser.parse_args(argv)

    summary = run_kaggle_ab_audit(
        taroz_summary_json=args.taroz_summary,
        bridge_metrics_root=args.bridge_metrics_root,
        row_delta_csv=args.row_delta_csv,
        output_dir=args.output_dir,
        dd_gate=DDAnchorGate(
            min_anchor_coverage=args.dd_min_anchor_coverage,
            require_no_tdcp_absent=not args.dd_allow_no_tdcp_coexist,
        ),
        ntdc_gate=NoTdcpCoexistGate(
            min_anchor_coverage=args.ntdc_min_anchor_coverage,
            require_no_dd_carrier=not args.ntdc_allow_dd_coexist,
        ),
        chunk_merge_verify=args.chunk_merge_verify,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
