#!/usr/bin/env python3
"""Analyze narrow FGO-over-raw-WLS proxy rescue thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_gsdc2023_ct_rbpf_fgo import parse_float_list  # noqa: E402


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_fgo_proxy_rescue_thresholds.csv")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce")


def phone_series(frame: pd.DataFrame) -> pd.Series:
    if "phone" in frame:
        return frame["phone"].astype(str).str.lower()
    if "trip" not in frame:
        return pd.Series("", index=frame.index, dtype=object)
    return frame["trip"].astype(str).str.split("/").str[-1].str.lower()


def phone_group_mask(frame: pd.DataFrame, phones: tuple[str, ...] | None) -> pd.Series:
    if phones is None:
        return pd.Series(True, index=frame.index, dtype=bool)
    allowed = {phone.lower() for phone in phones}
    return phone_series(frame).isin(allowed)


def parse_phone_groups(values: list[str]) -> list[tuple[str, tuple[str, ...] | None]]:
    groups: list[tuple[str, tuple[str, ...] | None]] = []
    for value in values:
        for raw_group in value.split(";"):
            item = raw_group.strip()
            if not item:
                continue
            if ":" in item:
                name, raw_phones = item.split(":", 1)
            else:
                name, raw_phones = item, item
            name = name.strip()
            raw_phones = raw_phones.strip()
            if not name:
                raise argparse.ArgumentTypeError(f"empty phone group name in {raw_group!r}")
            if raw_phones == "*" or raw_phones.lower() == "all":
                groups.append((name, None))
                continue
            phones = tuple(phone.strip().lower() for phone in raw_phones.split(",") if phone.strip())
            if not phones:
                raise argparse.ArgumentTypeError(f"empty phone group phones in {raw_group!r}")
            groups.append((name, phones))
    if not groups:
        raise argparse.ArgumentTypeError("expected at least one phone group")
    return groups


def rescue_mask(
    frame: pd.DataFrame,
    *,
    ratio_max: float,
    gap_ratio_max: float,
    quality_delta_max: float,
    mse_delta_vs_baseline_max: float,
    source_prefix: str = "fgo",
) -> pd.Series:
    return (
        (_numeric(frame, f"{source_prefix}_candidate_chunks") > 0)
        & (_numeric(frame, f"{source_prefix}_raw_wls_mse_block_chunks") > 0)
        & (_numeric(frame, f"{source_prefix}_mean_mse_ratio_vs_raw_wls") <= float(ratio_max))
        & (_numeric(frame, f"{source_prefix}_mean_baseline_gap_step_p95_ratio") <= float(gap_ratio_max))
        & (_numeric(frame, f"{source_prefix}_mean_quality_delta_vs_baseline") <= float(quality_delta_max))
        & (_numeric(frame, f"{source_prefix}_mean_mse_delta_vs_baseline") <= float(mse_delta_vs_baseline_max))
    )


def score_thresholds(
    frame: pd.DataFrame,
    *,
    ratio_max_values: list[float],
    gap_ratio_max_values: list[float],
    quality_delta_max_values: list[float],
    mse_delta_vs_baseline_max_values: list[float],
    source_prefix: str = "fgo",
    phone_group_name: str = "all",
    phone_allow: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    baseline_score = _numeric(frame, "baseline_score_m")
    source_score = _numeric(frame, fgo_score_column(source_prefix))
    score_delta = source_score - baseline_score
    valid_truth = score_delta.notna()

    rows: list[dict[str, object]] = []
    for ratio_max in ratio_max_values:
        for gap_ratio_max in gap_ratio_max_values:
            for quality_delta_max in quality_delta_max_values:
                for mse_delta_vs_baseline_max in mse_delta_vs_baseline_max_values:
                    mask = rescue_mask(
                        frame,
                        ratio_max=float(ratio_max),
                        gap_ratio_max=float(gap_ratio_max),
                        quality_delta_max=float(quality_delta_max),
                        mse_delta_vs_baseline_max=float(mse_delta_vs_baseline_max),
                        source_prefix=source_prefix,
                    )
                    selected = mask & phone_group_mask(frame, phone_allow) & valid_truth
                    n_selected = int(selected.sum())
                    if n_selected == 0:
                        continue
                    selected_delta = score_delta[selected]
                    wins = selected_delta < 0.0
                    rows.append(
                        {
                            "source_prefix": source_prefix,
                            "phone_group": phone_group_name,
                            "phone_allow": "*" if phone_allow is None else ",".join(phone_allow),
                            "ratio_max": float(ratio_max),
                            "gap_ratio_max": float(gap_ratio_max),
                            "quality_delta_max": float(quality_delta_max),
                            "mse_delta_vs_baseline_max": float(mse_delta_vs_baseline_max),
                            "selected_rows": n_selected,
                            "win_rows": int(wins.sum()),
                            "loss_rows": int((~wins).sum()),
                            "win_rate": float(wins.mean()),
                            "sum_score_delta_m": float(selected_delta.sum()),
                            "mean_score_delta_m": float(selected_delta.mean()),
                            "best_score_delta_m": float(selected_delta.min()),
                            "worst_score_delta_m": float(selected_delta.max()),
                        },
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_prefix",
                "phone_group",
                "phone_allow",
                "ratio_max",
                "gap_ratio_max",
                "quality_delta_max",
                "mse_delta_vs_baseline_max",
                "selected_rows",
                "win_rows",
                "loss_rows",
                "win_rate",
                "sum_score_delta_m",
                "mean_score_delta_m",
                "best_score_delta_m",
                "worst_score_delta_m",
            ],
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["win_rate", "sum_score_delta_m", "worst_score_delta_m", "selected_rows"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def fgo_score_column(source_prefix: str) -> str:
    if source_prefix == "fgo":
        return "fgo_score_m"
    return f"{source_prefix}_score_m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="diagnostic TDCP/FGO sweep CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-prefix", default="fgo")
    parser.add_argument("--ratio-max", type=parse_float_list, default=[1.05, 1.10, 1.15, 1.20, 1.30])
    parser.add_argument("--gap-ratio-max", type=parse_float_list, default=[0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--quality-delta-max", type=parse_float_list, default=[-0.40, -0.35, -0.30, -0.25, -0.20])
    parser.add_argument("--mse-delta-vs-baseline-max", type=parse_float_list, default=[0.0])
    parser.add_argument(
        "--phone-group",
        action="append",
        default=["all:*"],
        help="phone group as name:phone1,phone2; repeatable or semicolon-separated; use all:* for no filter",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    parts = [
        score_thresholds(
            frame,
            ratio_max_values=args.ratio_max,
            gap_ratio_max_values=args.gap_ratio_max,
            quality_delta_max_values=args.quality_delta_max,
            mse_delta_vs_baseline_max_values=args.mse_delta_vs_baseline_max,
            source_prefix=args.source_prefix,
            phone_group_name=phone_group_name,
            phone_allow=phone_allow,
        )
        for phone_group_name, phone_allow in parse_phone_groups(args.phone_group)
    ]
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(
            ["win_rate", "sum_score_delta_m", "worst_score_delta_m", "selected_rows"],
            ascending=[False, True, True, False],
        ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")
    if not result.empty:
        best = result.iloc[0]
        print(
            "best: "
            f"selected={int(best['selected_rows'])} "
            f"wins={int(best['win_rows'])} "
            f"sum_delta={float(best['sum_score_delta_m']):+.4f}m "
            f"worst={float(best['worst_score_delta_m']):+.4f}m",
        )


if __name__ == "__main__":
    main()
