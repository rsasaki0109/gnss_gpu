"""Load PLATEAU-style LOS/NLOS mask CSVs and map them to soft satellite weights."""

from __future__ import annotations

import csv
import warnings
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MIN_WEIGHT = 1.0e-3
DEFAULT_TOW_TOLERANCE_S = 0.11


@dataclass(frozen=True)
class NlosMaskTables:
    """Per-epoch NLOS PRN sets loaded from mask CSV files."""

    weak: dict[int, set[str]]
    strong: dict[int, set[str]]
    weak_by_tow: dict[float, set[str]] = field(default_factory=dict)
    strong_by_tow: dict[float, set[str]] = field(default_factory=dict)
    tow_keys: tuple[float, ...] = ()

    @classmethod
    def empty(cls) -> NlosMaskTables:
        return cls(weak={}, strong={})


def _aggregate_nlos_rows(path: Path) -> tuple[dict[int, set[str]], dict[float, set[str]]]:
    by_epoch: dict[int, set[str]] = {}
    by_tow: dict[float, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                is_los = int(row["is_los"])
            except (KeyError, ValueError):
                continue
            if is_los != 0:
                continue
            try:
                epoch_idx = int(row["epoch_idx"])
                prn = str(row["prn"]).strip()
                tow = float(row["tow"])
            except (KeyError, ValueError):
                continue
            if not prn:
                continue
            by_epoch.setdefault(epoch_idx, set()).add(prn)
            by_tow.setdefault(tow, set()).add(prn)
    return by_epoch, by_tow


def _warn_strong_not_subset_of_weak(
    *,
    label: str,
    weak: dict[int, set[str]],
    strong: dict[int, set[str]],
) -> None:
    extra = set()
    for epoch_idx, strong_set in strong.items():
        weak_set = weak.get(epoch_idx, set())
        extra |= strong_set - weak_set
    if extra:
        warnings.warn(
            f"{label}: strong-only NLOS PRNs {sorted(extra)[:5]} "
            f"(count={len(extra)}) are not in the weak set; "
            "treating strong membership as NLOS.",
            stacklevel=3,
        )


def load_nlos_prn_sets(path: str | Path | None) -> dict[int, set[str]]:
    """Load ``tow,epoch_idx,prn,is_los`` CSV rows with ``is_los=0`` as NLOS PRNs.

    Returns ``{epoch_idx: {prn, ...}}``. Missing files yield an empty mapping.
    """
    if not path:
        return {}
    try:
        by_epoch, _ = _aggregate_nlos_rows(Path(path))
    except FileNotFoundError:
        return {}
    return by_epoch


def load_nlos_mask_tables(
    weak_path: str | Path | None,
    strong_path: str | Path | None = None,
) -> NlosMaskTables:
    """Load weak and optional strong NLOS mask CSV files."""
    if not weak_path:
        return NlosMaskTables.empty()
    weak_path = Path(weak_path)
    try:
        weak, weak_by_tow = _aggregate_nlos_rows(weak_path)
    except FileNotFoundError:
        return NlosMaskTables.empty()

    strong: dict[int, set[str]] = {}
    strong_by_tow: dict[float, set[str]] = {}
    if strong_path:
        try:
            strong, strong_by_tow = _aggregate_nlos_rows(Path(strong_path))
        except FileNotFoundError:
            strong = {}
            strong_by_tow = {}
    _warn_strong_not_subset_of_weak(label="weak mask", weak=weak, strong=strong)

    tow_keys = tuple(sorted(weak_by_tow.keys()))
    return NlosMaskTables(
        weak=weak,
        strong=strong,
        weak_by_tow=weak_by_tow,
        strong_by_tow=strong_by_tow,
        tow_keys=tow_keys,
    )


def lookup_nlos_sets(
    tables: NlosMaskTables,
    epoch_idx: int,
    *,
    tow: float | None = None,
    tow_tolerance: float = DEFAULT_TOW_TOLERANCE_S,
) -> tuple[set[str], set[str]]:
    """Resolve weak/strong NLOS PRN sets, preferring ``tow`` when available."""
    if tow is not None and tables.tow_keys:
        keys = tables.tow_keys
        pos = bisect_left(keys, float(tow))
        candidates: list[float] = []
        if pos < len(keys):
            candidates.append(keys[pos])
        if pos > 0:
            candidates.append(keys[pos - 1])
        best_key = None
        best_delta = float(tow_tolerance) + 1.0
        for key in candidates:
            delta = abs(float(key) - float(tow))
            if delta <= float(tow_tolerance) and delta < best_delta:
                best_key = key
                best_delta = delta
        if best_key is not None:
            return (
                tables.weak_by_tow.get(best_key, set()),
                tables.strong_by_tow.get(best_key, set()),
            )
    return (
        tables.weak.get(int(epoch_idx), set()),
        tables.strong.get(int(epoch_idx), set()),
    )


def nlos_weight_factor(
    *,
    is_nlos: bool,
    is_strong: bool,
    k_weak: float,
    k_strong: float,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> float:
    """Map LOS/NLOS flags to a soft weight multiplier (never zero)."""
    if not is_nlos:
        return 1.0
    if is_strong and float(k_strong) > 0.0:
        scale = float(k_strong)
    elif float(k_weak) > 0.0:
        scale = float(k_weak)
    else:
        return max(float(min_weight), DEFAULT_MIN_WEIGHT)
    return max(float(min_weight), 1.0 / scale)


def epoch_prn_weights(
    epoch_idx: int,
    prns: Iterable[str],
    tables: NlosMaskTables,
    *,
    tow: float | None = None,
    tow_tolerance: float = DEFAULT_TOW_TOLERANCE_S,
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> dict[str, float]:
    """Return per-PRN multipliers for one epoch."""
    weak, strong = lookup_nlos_sets(
        tables,
        int(epoch_idx),
        tow=tow,
        tow_tolerance=tow_tolerance,
    )
    out: dict[str, float] = {}
    for prn in prns:
        prn_key = str(prn).strip()
        if not prn_key:
            continue
        is_nlos = prn_key in weak or prn_key in strong
        is_strong = prn_key in strong
        out[prn_key] = nlos_weight_factor(
            is_nlos=is_nlos,
            is_strong=is_strong,
            k_weak=k_weak,
            k_strong=k_strong,
            min_weight=min_weight,
        )
    return out


def apply_mask_to_weights(
    epoch_idx: int,
    prns: Iterable[str],
    base_weights: Iterable[float],
    tables: NlosMaskTables,
    *,
    tow: float | None = None,
    tow_tolerance: float = DEFAULT_TOW_TOLERANCE_S,
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> list[float]:
    """Multiply base weights by geometry mask factors for one epoch."""
    prn_list = [str(p).strip() for p in prns]
    factors = epoch_prn_weights(
        epoch_idx,
        prn_list,
        tables,
        tow=tow,
        tow_tolerance=tow_tolerance,
        k_weak=k_weak,
        k_strong=k_strong,
        min_weight=min_weight,
    )
    return [
        float(w) * factors.get(prn, 1.0)
        for prn, w in zip(prn_list, base_weights)
    ]


def normalize_mask_prn(sat_id: str) -> str:
    """Strip optional DD-carrier suffixes such as ``G01@L1`` -> ``G01``."""
    return str(sat_id).split("@", 1)[0].strip()


def dd_pair_nlos_factors(
    epoch_idx: int,
    sat_ids: Iterable[str],
    ref_sat_ids: Iterable[str],
    tables: NlosMaskTables,
    *,
    tow: float | None = None,
    tow_tolerance: float = DEFAULT_TOW_TOLERANCE_S,
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> list[float]:
    """Return per-DD-pair multipliers using min(factor(k), factor(ref))."""
    sat_list = [normalize_mask_prn(s) for s in sat_ids]
    ref_list = [normalize_mask_prn(s) for s in ref_sat_ids]
    prn_factors = epoch_prn_weights(
        epoch_idx,
        set(sat_list) | set(ref_list),
        tables,
        tow=tow,
        tow_tolerance=tow_tolerance,
        k_weak=k_weak,
        k_strong=k_strong,
        min_weight=min_weight,
    )
    return [
        min(prn_factors.get(sat, 1.0), prn_factors.get(ref, 1.0))
        for sat, ref in zip(sat_list, ref_list)
    ]


def scale_dd_result_weights_by_nlos_mask(
    dd_result,
    epoch_idx: int,
    tables: NlosMaskTables | None,
    *,
    tow: float | None = None,
    tow_tolerance: float = DEFAULT_TOW_TOLERANCE_S,
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> None:
    """Multiply ``dd_result.dd_weights`` in place when a geometry mask is active."""
    if tables is None or not (tables.weak or tables.strong):
        return
    if dd_result is None or int(getattr(dd_result, "n_dd", 0)) <= 0:
        return
    sat_ids = getattr(dd_result, "sat_ids", ()) or ()
    ref_sat_ids = getattr(dd_result, "ref_sat_ids", ()) or ()
    if not sat_ids or not ref_sat_ids:
        return
    factors = dd_pair_nlos_factors(
        epoch_idx,
        sat_ids,
        ref_sat_ids,
        tables,
        tow=tow,
        tow_tolerance=tow_tolerance,
        k_weak=k_weak,
        k_strong=k_strong,
        min_weight=min_weight,
    )
    if not factors:
        return
    dd_result.dd_weights = (
        np.asarray(dd_result.dd_weights, dtype=np.float64)
        * np.asarray(factors, dtype=np.float64)
    )
