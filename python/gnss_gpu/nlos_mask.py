"""Load PLATEAU-style LOS/NLOS mask CSVs and map them to soft satellite weights."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_MIN_WEIGHT = 1.0e-3


@dataclass(frozen=True)
class NlosMaskTables:
    """Per-epoch NLOS PRN sets loaded from mask CSV files."""

    weak: dict[int, set[str]]
    strong: dict[int, set[str]]

    @classmethod
    def empty(cls) -> NlosMaskTables:
        return cls(weak={}, strong={})


def load_nlos_prn_sets(path: str | Path | None) -> dict[int, set[str]]:
    """Load ``tow,epoch_idx,prn,is_los`` CSV rows with ``is_los=0`` as NLOS PRNs.

    Returns ``{epoch_idx: {prn, ...}}``. Missing files yield an empty mapping.
    """
    if not path:
        return {}
    resolved = str(Path(path))
    out: dict[int, set[str]] = {}
    try:
        with Path(resolved).open(newline="", encoding="utf-8") as handle:
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
                except (KeyError, ValueError):
                    continue
                if prn:
                    out.setdefault(epoch_idx, set()).add(prn)
    except FileNotFoundError:
        return {}
    return out


def load_nlos_mask_tables(
    weak_path: str | Path | None,
    strong_path: str | Path | None = None,
) -> NlosMaskTables:
    """Load weak and optional strong NLOS mask CSV files."""
    return NlosMaskTables(
        weak=load_nlos_prn_sets(weak_path),
        strong=load_nlos_prn_sets(strong_path),
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
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> dict[str, float]:
    """Return per-PRN multipliers for one epoch."""
    weak = tables.weak.get(int(epoch_idx), set())
    strong = tables.strong.get(int(epoch_idx), set())
    out: dict[str, float] = {}
    for prn in prns:
        prn_key = str(prn).strip()
        if not prn_key:
            continue
        is_nlos = prn_key in weak
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
        k_weak=k_weak,
        k_strong=k_strong,
        min_weight=min_weight,
    )
    return [
        float(w) * factors.get(prn, 1.0)
        for prn, w in zip(prn_list, base_weights)
    ]
