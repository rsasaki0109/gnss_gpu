"""File I/O and CLI-list parsing helpers for ``exp_ppc_ctrbpf_fgo``.

Moved verbatim out of ``exp_ppc_ctrbpf_fgo`` (which re-exports every name) so
the loaders shared by the PPC experiment scripts can be imported without the
heavy particle-filter module. Numpy-only; no behaviour change.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _load_hybrid_pos_file(
    path: Path,
) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Parse a libgnss++ ``.pos`` file into TOW-keyed ECEF + Status lookups.

    File format (whitespace-separated, '%' comments):
        GPS_Week  GPS_TOW  X  Y  Z  Lat  Lon  Height  Status  ...

    Returns ``(positions, statuses)``. Rows with non-finite coords or
    status==0 (no fix) are dropped. Keys are rounded to 0.1 s to match
    the rover-epoch TOW grid used elsewhere.

    Note: libgnss++ Status semantics are inverted from RTKLIB; on the
    Phase 6 baseline pos files Status=4 is the high-quality (cm-class)
    bucket, while Status in {1, 3} are m-class regions where Phase 4
    FGO+LAMBDA is more likely to help than to hurt.
    """
    positions: dict[float, np.ndarray] = {}
    statuses: dict[float, int] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                status = int(float(parts[8]))
            except (ValueError, IndexError):
                continue
            if status == 0:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            positions[tow] = np.array([x, y, z], dtype=np.float64)
            statuses[tow] = int(status)
    return positions, statuses


def _load_full_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    """Load every reference.csv row in order (TOW seconds, ECEF)."""
    rows: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            ecef = np.array(
                [float(row[5]), float(row[6]), float(row[7])],
                dtype=np.float64,
            )
            rows.append((tow, ecef))
    return rows


def _reference_position_map(rows: list[tuple[float, np.ndarray]]) -> dict[float, np.ndarray]:
    """Return reference ECEF positions keyed by rounded TOW for diagnostics."""
    return {round(float(tow), 1): np.asarray(ecef, dtype=np.float64) for tow, ecef in rows}


_RANKER_PREDICTIONS_CACHE: dict[str, dict[tuple[str, float, str], float]] = {}


def _load_ranker_predictions(path: str) -> dict[tuple[str, float, str], float]:
    """Load ranker predictions CSV into {(run_id, tow, label): p_pass}. Cached."""
    if not path:
        return {}
    if path in _RANKER_PREDICTIONS_CACHE:
        return _RANKER_PREDICTIONS_CACHE[path]
    lookup: dict[tuple[str, float, str], float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                run_id = str(row["run_id"])
                tow = round(float(row["tow"]), 1)
                label = str(row["label"])
                p = float(row["p_pass"])
                lookup[(run_id, tow, label)] = p
            except (ValueError, KeyError):
                continue
    _RANKER_PREDICTIONS_CACHE[path] = lookup
    return lookup


def _ranker_lookup_for_run(path: str, city: str, run: str) -> dict[tuple[float, str], float]:
    """Slice ranker predictions for a single run into {(tow, label): p_pass}."""
    if not path:
        return {}
    full = _load_ranker_predictions(path)
    run_id = f"{city}_{run}"
    return {(t, lbl): p for (rid, t, lbl), p in full.items() if rid == run_id}


_NLOS_MASK_CACHE: dict[str, dict[int, set[str]]] = {}


def _load_nlos_mask_csv(path: str) -> dict[int, set[str]]:
    """Load PLATEAU NLOS mask CSV → {epoch_idx: {prn1, prn2, ...}} of NLOS PRNs.

    Expected columns: tow, epoch_idx, prn, is_los (1=LOS, 0=NLOS).
    Only rows with is_los=0 contribute to the returned set.
    """
    if not path:
        return {}
    if path in _NLOS_MASK_CACHE:
        return _NLOS_MASK_CACHE[path]
    out: dict[int, set[str]] = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
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
                if not prn:
                    continue
                out.setdefault(epoch_idx, set()).add(prn)
    except FileNotFoundError:
        pass
    _NLOS_MASK_CACHE[path] = out
    return out


def _write_pos_file(
    path: Path,
    times: np.ndarray,
    positions: np.ndarray,
    status: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("% CT-RBPF-FGO PPC port (gnss_gpu)\n")
        fh.write("% GPST                tow         x-ecef(m)        y-ecef(m)        z-ecef(m)   Q  ns   sdx    sdy    sdz   age  ratio\n")
        for tow, pos in zip(times, positions, strict=True):
            fh.write(
                f"0 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"   1   0  0.000  0.000  0.000  0.00  0.0   {status}\n"
            )


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_path_list(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_label_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_label_factor_list(raw: str) -> tuple[tuple[str, float], ...]:
    factors: list[tuple[str, float]] = []
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"invalid label factor spec {spec!r}; expected label=factor"
            )
        label, factor_raw = (part.strip() for part in spec.split("=", 1))
        if not label:
            raise ValueError(
                f"invalid label factor spec {spec!r}; expected label=factor"
            )
        factors.append((label, float(factor_raw)))
    return tuple(factors)


def _parse_diag_threshold_list(raw: str) -> tuple[tuple[str, float], ...]:
    thresholds: list[tuple[str, float]] = []
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "=" in spec:
            key, value_raw = (part.strip() for part in spec.split("=", 1))
        elif ":" in spec:
            key, value_raw = (part.strip() for part in spec.split(":", 1))
        else:
            raise ValueError(
                f"invalid diagnostic threshold {spec!r}; expected field=value"
            )
        if not key:
            raise ValueError(
                f"invalid diagnostic threshold {spec!r}; expected field=value"
            )
        thresholds.append((key, float(value_raw)))
    return tuple(thresholds)


def _parse_run_label_blocks(raw: str) -> dict[tuple[str, str], set[str]]:
    blocks: dict[tuple[str, str], set[str]] = {}
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        run_key, labels_raw = spec.split("=", 1)
        if "/" not in run_key:
            raise ValueError(
                f"invalid run label block key {run_key!r}; expected city/run"
            )
        city, run = (part.strip() for part in run_key.split("/", 1))
        labels = {p.strip() for p in labels_raw.replace(",", "+").split("+") if p.strip()}
        if not city or not run or not labels:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        blocks[(city, run)] = set(labels)
    return blocks
