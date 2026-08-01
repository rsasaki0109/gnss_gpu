#!/usr/bin/env python3
"""Audit PPC carrier ambiguity arcs without using truth or extra sensors."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
ROUTES = tuple(
    (city, f"run{run}")
    for city in ("tokyo", "nagoya")
    for run in range(1, 4)
)


@dataclass
class Arc:
    arc_id: str
    satellite: str
    signal: str
    start_tow: float
    end_tow: float
    epochs: int
    started_by_lli: bool
    fold: int


def _tow(header: str) -> float:
    fields = header[1:].split()
    if len(fields) < 6:
        raise ValueError(f"invalid RINEX epoch header: {header.rstrip()}")
    year, month, day, hour, minute = map(int, fields[:5])
    second = float(fields[5])
    whole = int(math.floor(second))
    stamp = datetime(
        year, month, day, hour, minute, whole, tzinfo=timezone.utc
    )
    return round(
        ((stamp - GPS_EPOCH).total_seconds() + second - whole) % 604800.0,
        3,
    )


def _fold(arc_id: str, fold_count: int, salt: str = "") -> int:
    digest = hashlib.sha256(f"{salt}:{arc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def audit_rinex_arcs(
    path: Path,
    route: str,
    *,
    maximum_gap_s: float = 1.5,
    fold_count: int = 5,
    fold_salt: str = "",
) -> dict[str, object]:
    if maximum_gap_s <= 0.0 or fold_count < 2:
        raise ValueError("maximum_gap_s must be positive and fold_count >= 2")

    observation_types: dict[str, list[str]] = {}
    current_system = ""
    arcs: list[Arc] = []
    active: dict[tuple[str, str], tuple[int, float]] = {}
    epoch_tows: list[float] = []

    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            if "SYS / # / OBS TYPES" in line:
                system = line[0].strip()
                if system:
                    current_system = system
                    observation_types.setdefault(system, [])
                if current_system:
                    observation_types[current_system].extend(line[7:60].split())
            if "END OF HEADER" in line:
                break
        else:
            raise ValueError(f"RINEX END OF HEADER not found in {path}")

        while True:
            header = stream.readline()
            if not header:
                break
            if not header.startswith(">"):
                continue
            fields = header.split()
            if len(fields) < 9:
                raise ValueError(f"invalid RINEX epoch header in {path}")
            tow = _tow(header)
            epoch_tows.append(tow)
            record_count = int(fields[-1])
            for _ in range(record_count):
                record = stream.readline()
                if not record:
                    raise ValueError(f"truncated RINEX epoch in {path}")
                satellite = record[:3]
                types = observation_types.get(satellite[:1], [])
                for index, observation_type in enumerate(types):
                    if not observation_type.startswith("L"):
                        continue
                    field = record[3 + 16 * index : 3 + 16 * (index + 1)]
                    try:
                        value = float(field[:14])
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        continue
                    lli = field[14:15].strip()
                    key = (satellite, observation_type)
                    prior = active.get(key)
                    starts_new = (
                        prior is None
                        or tow - prior[1] > maximum_gap_s + 1e-6
                        or (lli not in ("", "0"))
                    )
                    if starts_new:
                        sequence = sum(
                            arc.satellite == satellite and arc.signal == observation_type
                            for arc in arcs
                        )
                        arc_id = (
                            f"{route}:{satellite}:{observation_type}:"
                            f"{tow:.3f}:{sequence}"
                        )
                        arcs.append(
                            Arc(
                                arc_id,
                                satellite,
                                observation_type,
                                tow,
                                tow,
                                1,
                                lli not in ("", "0"),
                                _fold(arc_id, fold_count, fold_salt),
                            )
                        )
                        active[key] = (len(arcs) - 1, tow)
                    else:
                        arc = arcs[prior[0]]
                        arc.end_tow = tow
                        arc.epochs += 1
                        active[key] = (prior[0], tow)

    if not epoch_tows:
        raise ValueError(f"no observation epochs in {path}")
    start_tow = min(epoch_tows)
    end_tow = max(epoch_tows)
    boundaries = [
        start_tow + (end_tow - start_tow) * index / fold_count
        for index in range(1, fold_count)
    ]
    crossing = sum(
        any(arc.start_tow < boundary < arc.end_tow for boundary in boundaries)
        for arc in arcs
    )
    fold_counts = Counter(arc.fold for arc in arcs)
    system_counts = Counter(arc.satellite[:1] for arc in arcs)
    return {
        "route": route,
        "rinex": str(path.resolve()),
        "epochs": len(epoch_tows),
        "start_tow": start_tow,
        "end_tow": end_tow,
        "ambiguity_arcs": len(arcs),
        "lli_started_arcs": sum(arc.started_by_lli for arc in arcs),
        "arc_epochs": sum(arc.epochs for arc in arcs),
        "maximum_arc_epochs": max(arc.epochs for arc in arcs),
        "naive_time_block_crossing_arcs": crossing,
        "arc_fold_counts": {
            str(index): fold_counts[index] for index in range(fold_count)
        },
        "constellation_arc_counts": dict(sorted(system_counts.items())),
        "arc_ids": [arc.arc_id for arc in arcs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-gap", type=float, default=1.5)
    parser.add_argument("--salt", default="")
    args = parser.parse_args()

    routes = [
        audit_rinex_arcs(
            args.data_root / city / run / "rover.obs",
            f"{city}/{run}",
            maximum_gap_s=args.maximum_gap,
            fold_count=args.folds,
            fold_salt=args.salt,
        )
        for city, run in ROUTES
    ]
    arc_sets = {route["route"]: set(route["arc_ids"]) for route in routes}
    overlap = {
        f"{left}|{right}": len(arc_sets[left] & arc_sets[right])
        for index, left in enumerate(arc_sets)
        for right in tuple(arc_sets)[index + 1 :]
    }
    payload = {
        "schema": "gnss_gpu_ppc_ambiguity_arc_audit_v1",
        "inputs": "PPC rover RINEX carrier phase and LLI only",
        "fold_assignment": "sha256(salt,route,satellite,signal,arc-start) modulo folds",
        "fold_salt": args.salt,
        "route_namespaced_leave_one_run_out_overlap": overlap,
        "routes": routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
