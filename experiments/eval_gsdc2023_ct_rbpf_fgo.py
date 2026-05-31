#!/usr/bin/env python3
"""Evaluate the GSDC2023 CT-RBPF/FGO bridge candidate on train trips."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    BridgeConfig,
    CT_RBPF_FGO_SOURCE,
    DEFAULT_CT_RBPF_MOTION_SIGMA_M,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    validate_raw_gsdc2023_trip,
)


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_ct_rbpf_fgo_eval_20260519.csv")


def discover_train_trips(data_root: Path) -> list[str]:
    trips: list[str] = []
    train_root = data_root / "train"
    for phone_dir in sorted(train_root.glob("*/*")):
        if not phone_dir.is_dir():
            continue
        if (phone_dir / "device_gnss.csv").exists() and (phone_dir / "ground_truth.csv").exists():
            trips.append(str(phone_dir.relative_to(data_root)))
    return trips


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def score_delta(a: object, b: object) -> float | None:
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(af) or not np.isfinite(bf):
        return None
    return float(af - bf)


def ct_candidate_summary(payload: dict[str, object]) -> dict[str, object]:
    records = payload.get("chunk_selection_records")
    if not isinstance(records, list):
        return {
            "ct_candidate_chunks": 0,
            "ct_candidate_mean_mse_pr": np.nan,
            "ct_candidate_min_mse_pr": np.nan,
            "ct_candidate_mean_quality_score": np.nan,
        }
    mse_pr: list[float] = []
    quality: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        candidates = record.get("candidates")
        if not isinstance(candidates, dict):
            continue
        candidate = candidates.get(CT_RBPF_FGO_SOURCE)
        if not isinstance(candidate, dict):
            continue
        try:
            mse = float(candidate.get("mse_pr"))
            q = float(candidate.get("quality_score"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(mse):
            mse_pr.append(mse)
        if np.isfinite(q):
            quality.append(q)
    return {
        "ct_candidate_chunks": len(mse_pr),
        "ct_candidate_mean_mse_pr": float(np.mean(mse_pr)) if mse_pr else np.nan,
        "ct_candidate_min_mse_pr": float(np.min(mse_pr)) if mse_pr else np.nan,
        "ct_candidate_mean_quality_score": float(np.mean(quality)) if quality else np.nan,
    }


def metrics_row(case: str, trip: str, sigma: float | None, payload: dict[str, object]) -> dict[str, object]:
    counts = payload.get("selected_source_counts")
    selected_counts = counts if isinstance(counts, dict) else {}
    row: dict[str, object] = {
        "case": case,
        "trip": trip,
        "ct_rbpf_motion_sigma_m": sigma,
        "n_epochs": payload.get("n_epochs"),
        "selected_source_mode": payload.get("selected_source_mode"),
        "selected_score_m": payload.get("selected_score_m"),
        "baseline_score_m": payload.get("kaggle_wls_score_m"),
        "raw_wls_score_m": payload.get("raw_wls_score_m"),
        "fgo_score_m": payload.get("fgo_score_m"),
        "selected_mse_pr": payload.get("selected_mse_pr"),
        "baseline_mse_pr": payload.get("baseline_mse_pr"),
        "raw_wls_mse_pr": payload.get("raw_wls_mse_pr"),
        "fgo_mse_pr": payload.get("fgo_mse_pr"),
        "failed_chunks": payload.get("failed_chunks"),
        "fgo_iters": payload.get("fgo_iters"),
        "selected_baseline_epochs": int(selected_counts.get("baseline", 0) or 0),
        "selected_raw_wls_epochs": int(selected_counts.get("raw_wls", 0) or 0),
        "selected_fgo_epochs": int(selected_counts.get("fgo", 0) or 0),
        "selected_ct_rbpf_epochs": int(selected_counts.get(CT_RBPF_FGO_SOURCE, 0) or 0),
    }
    row.update(ct_candidate_summary(payload))
    return row


def build_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        fgo_iters=args.fgo_iters,
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
        gated_baseline_threshold=args.gated_threshold,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
    )


def run_eval(args: argparse.Namespace) -> pd.DataFrame:
    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    base_config = build_config(args)
    rows: list[dict[str, object]] = []
    total = len(trips)
    for idx, trip in enumerate(trips, start=1):
        start = time.time()
        base = validate_raw_gsdc2023_trip(
            args.data_root,
            trip,
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            config=base_config,
        ).metrics_payload()
        rows.append(metrics_row("base", trip, None, base))
        for sigma in args.ct_rbpf_motion_sigma_m:
            ct_config = replace(
                base_config,
                ct_rbpf_fgo_enabled=True,
                ct_rbpf_motion_sigma_m=float(sigma),
            )
            ct = validate_raw_gsdc2023_trip(
                args.data_root,
                trip,
                max_epochs=args.max_epochs,
                start_epoch=args.start_epoch,
                config=ct_config,
            ).metrics_payload()
            row = metrics_row("ct_rbpf", trip, float(sigma), ct)
            row["delta_selected_score_m_vs_base"] = score_delta(row["selected_score_m"], base.get("selected_score_m"))
            row["delta_selected_mse_pr_vs_base"] = score_delta(row["selected_mse_pr"], base.get("selected_mse_pr"))
            rows.append(row)
        print(f"[{idx}/{total}] {trip} done in {time.time() - start:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone trip; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--ct-rbpf-motion-sigma-m", type=parse_float_list, default=[DEFAULT_CT_RBPF_MOTION_SIGMA_M])
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--position-source", choices=("auto", "gated"), default="gated")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    frame = run_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    ct = frame[frame["case"] == "ct_rbpf"].copy()
    print(f"wrote: {args.output}")
    if not ct.empty:
        finite_score = pd.to_numeric(ct["delta_selected_score_m_vs_base"], errors="coerce")
        finite_mse = pd.to_numeric(ct["delta_selected_mse_pr_vs_base"], errors="coerce")
        selected_ct = pd.to_numeric(ct["selected_ct_rbpf_epochs"], errors="coerce").fillna(0)
        print(f"ct rows: {len(ct)}")
        print(f"score wins: {int((finite_score < 0).sum())}/{int(finite_score.notna().sum())}")
        print(f"mse wins: {int((finite_mse < 0).sum())}/{int(finite_mse.notna().sum())}")
        print(f"selected ct epochs: {int(selected_ct.sum())}")


if __name__ == "__main__":
    main()
