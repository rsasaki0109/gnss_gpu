#!/usr/bin/env python3
"""Build GSDC2023 source-family ablation submissions from bridge row sources."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from experiments.build_gsdc2023_pre_submit_manifest import (
    DELTA_CHANGED_THRESHOLD_M,
    REQUIRED_COLUMNS,
    sha256_file,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


KEY_COLUMNS = ("tripId", "UnixTimeMillis")
COORDINATE_COLUMNS = ("LatitudeDegrees", "LongitudeDegrees")
SOURCE_COLUMN = "SelectedSource"
INTERPOLATED_SOURCE = "interpolated_missing"


@dataclass(frozen=True)
class SourceFamilySpec:
    name: str
    sources: tuple[str, ...]
    phones: tuple[str, ...] = ()


DEFAULT_SOURCE_FAMILIES = (
    SourceFamilySpec("pixel4_fgo", ("fgo",), ("pixel4",)),
    SourceFamilySpec("pixel6pro_fgo", ("fgo",), ("pixel6pro",)),
    SourceFamilySpec("pixel5_fgo_no_tdcp", ("fgo_no_tdcp",), ("pixel5",)),
    SourceFamilySpec("raw_wls_all", ("raw_wls",)),
    SourceFamilySpec("interpolated_missing_all", (INTERPOLATED_SOURCE,)),
    SourceFamilySpec("nonbaseline_all", ("raw_wls", "fgo", "fgo_no_tdcp", INTERPOLATED_SOURCE)),
)


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _assert_same_keys(reference: pd.DataFrame, target: pd.DataFrame) -> None:
    if len(reference) != len(target):
        raise SystemExit(f"row count mismatch: {len(reference)} != {len(target)}")
    for column in KEY_COLUMNS:
        if not reference[column].equals(target[column]):
            raise SystemExit(f"{column} mismatch")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return slug.strip("_").lower()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def parse_source_family(raw: str) -> SourceFamilySpec:
    """Parse NAME=source1,source2[@phone1,phone2]."""

    if "=" not in raw:
        raise SystemExit(f"source family must be NAME=sources[@phones]: {raw}")
    name, expression = raw.split("=", 1)
    name = _slug(name)
    if not name:
        raise SystemExit(f"source family name is empty: {raw}")
    source_text, sep, phone_text = expression.partition("@")
    sources = _split_csv(source_text)
    phones = _split_csv(phone_text) if sep else ()
    if not sources:
        raise SystemExit(f"source family must include at least one source: {raw}")
    return SourceFamilySpec(name=name, sources=sources, phones=phones)


def _phone_from_trip(trip_id: str) -> str:
    return str(trip_id).split("/")[-1].lower()


def _read_bridge_position_sources(bridge_output_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(bridge_output_root.rglob("bridge_positions.csv"), key=lambda item: str(item)):
        trip_id = path.parent.relative_to(bridge_output_root).as_posix()
        frame = pd.read_csv(path, usecols=["UnixTimeMillis", SOURCE_COLUMN])
        frame.insert(0, "tripId", trip_id)
        frame["phone"] = _phone_from_trip(trip_id)
        frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].astype(str).str.lower()
        rows.append(frame)
    if not rows:
        raise SystemExit(f"no bridge_positions.csv files found under {bridge_output_root}")
    return pd.concat(rows, ignore_index=True)


def _read_missing_sources(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=[*KEY_COLUMNS, SOURCE_COLUMN, "phone"])
    frame = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    output = frame.loc[:, list(KEY_COLUMNS)].copy()
    output[SOURCE_COLUMN] = INTERPOLATED_SOURCE
    output["phone"] = output["tripId"].map(_phone_from_trip)
    return output


def load_bridge_row_sources(bridge_output_root: Path, missing_rows_path: Path | None = None) -> pd.DataFrame:
    sources = pd.concat(
        [
            _read_bridge_position_sources(bridge_output_root.expanduser().resolve()),
            _read_missing_sources(missing_rows_path.expanduser().resolve() if missing_rows_path else None),
        ],
        ignore_index=True,
    )
    duplicate_mask = sources.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicates = sources.loc[duplicate_mask, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise SystemExit(f"duplicate bridge row source keys: {duplicates}")
    return sources


def _merge_sources(target: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    merged = target.loc[:, list(KEY_COLUMNS)].merge(sources, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    missing_mask = merged[SOURCE_COLUMN].isna()
    if missing_mask.any():
        examples = merged.loc[missing_mask, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise SystemExit(f"bridge row sources do not cover {int(missing_mask.sum())} target rows: {examples}")
    merged[SOURCE_COLUMN] = merged[SOURCE_COLUMN].astype(str).str.lower()
    merged["phone"] = merged["phone"].astype(str).str.lower()
    return merged


def _family_mask(source_rows: pd.DataFrame, spec: SourceFamilySpec) -> np.ndarray:
    source_set = set(item.lower() for item in spec.sources)
    mask = source_rows[SOURCE_COLUMN].isin(source_set).to_numpy()
    if spec.phones:
        phone_set = set(item.lower() for item in spec.phones)
        mask &= source_rows["phone"].isin(phone_set).to_numpy()
    return mask


def _delta_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | int]:
    deltas = haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )
    score = gsdc_score_m(deltas)
    return {
        "changed_rows": int(np.count_nonzero(deltas > DELTA_CHANGED_THRESHOLD_M)),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "max_m": float(score["max_m"]),
    }


def _prefixed_summary(prefix: str, summary: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _build_candidate(
    *,
    reference: pd.DataFrame,
    target: pd.DataFrame,
    selected_mask: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    if mode == "only":
        candidate = reference.copy()
        candidate.loc[selected_mask, list(COORDINATE_COLUMNS)] = target.loc[
            selected_mask,
            list(COORDINATE_COLUMNS),
        ].to_numpy()
        return candidate
    if mode == "revert":
        candidate = target.copy()
        candidate.loc[selected_mask, list(COORDINATE_COLUMNS)] = reference.loc[
            selected_mask,
            list(COORDINATE_COLUMNS),
        ].to_numpy()
        return candidate
    raise SystemExit(f"unknown mode: {mode}")


def _write_candidate(candidate: pd.DataFrame, output_dir: Path, *, mode: str, spec: SourceFamilySpec, tag: str) -> Path:
    output_path = output_dir / f"submission_source_family_{mode}_{spec.name}_{tag}.csv"
    candidate.to_csv(output_path, index=False)
    return output_path


def build_source_family_ablation_candidates(
    *,
    reference_path: Path,
    target_path: Path,
    bridge_output_root: Path,
    output_dir: Path,
    tag: str,
    missing_rows_path: Path | None = None,
    specs: tuple[SourceFamilySpec, ...] = DEFAULT_SOURCE_FAMILIES,
    modes: tuple[str, ...] = ("only", "revert"),
) -> list[dict[str, Any]]:
    reference = _read_submission(reference_path.expanduser().resolve())
    target = _read_submission(target_path.expanduser().resolve())
    _assert_same_keys(reference, target)
    source_rows = _merge_sources(target, load_bridge_row_sources(bridge_output_root, missing_rows_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        selected_mask = _family_mask(source_rows, spec)
        selected_count = int(np.count_nonzero(selected_mask))
        if selected_count == 0:
            continue
        selected = source_rows.loc[selected_mask]
        source_counts = selected[SOURCE_COLUMN].value_counts().sort_index().to_dict()
        phone_counts = selected["phone"].value_counts().sort_index().to_dict()
        selected_reference = reference.loc[selected_mask].reset_index(drop=True)
        selected_target = target.loc[selected_mask].reset_index(drop=True)
        selected_summary = _delta_summary(selected_reference, selected_target)

        for mode in modes:
            candidate = _build_candidate(
                reference=reference,
                target=target,
                selected_mask=selected_mask,
                mode=mode,
            )
            output_path = _write_candidate(candidate, output_dir, mode=mode, spec=spec, tag=tag)
            rows.append(
                {
                    "family": spec.name,
                    "mode": mode,
                    "sources": ",".join(spec.sources),
                    "phones": ",".join(spec.phones),
                    "selected_rows": selected_count,
                    "selected_trip_count": int(selected["tripId"].nunique()),
                    "selected_source_counts": json.dumps(source_counts, sort_keys=True),
                    "selected_phone_counts": json.dumps(phone_counts, sort_keys=True),
                    "output": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    **_prefixed_summary("selected_vs_reference", selected_summary),
                    **_prefixed_summary("candidate_vs_reference", _delta_summary(reference, candidate)),
                    **_prefixed_summary("candidate_vs_target", _delta_summary(target, candidate)),
                },
            )

    manifest_path = output_dir / f"source_family_ablation_manifest_{tag}.csv"
    fieldnames = list(rows[0]) if rows else [
        "family",
        "mode",
        "sources",
        "phones",
        "selected_rows",
        "selected_trip_count",
        "selected_source_counts",
        "selected_phone_counts",
        "output",
        "output_sha256",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / f"source_family_ablation_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(
            {
                "reference": str(reference_path),
                "target": str(target_path),
                "bridge_output_root": str(bridge_output_root),
                "missing_rows": str(missing_rows_path) if missing_rows_path is not None else None,
                "output_dir": str(output_dir),
                "tag": tag,
                "modes": list(modes),
                "families": [
                    {"name": spec.name, "sources": list(spec.sources), "phones": list(spec.phones)}
                    for spec in specs
                ],
                "candidate_count": len(rows),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {manifest_path}")
    print(f"saved: {summary_path}")
    print(f"prepared: {len(rows)} candidate(s)")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--bridge-output-root", type=Path, required=True)
    parser.add_argument("--missing-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--family", action="append", help="NAME=source1,source2[@phone1,phone2].")
    parser.add_argument("--mode", action="append", choices=("only", "revert"))
    args = parser.parse_args(argv)

    build_source_family_ablation_candidates(
        reference_path=args.reference,
        target_path=args.target,
        bridge_output_root=args.bridge_output_root,
        missing_rows_path=args.missing_rows,
        output_dir=args.output_dir,
        tag=args.tag,
        specs=tuple(parse_source_family(item) for item in args.family) if args.family else DEFAULT_SOURCE_FAMILIES,
        modes=tuple(args.mode) if args.mode else ("only", "revert"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
