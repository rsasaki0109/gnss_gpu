#!/usr/bin/env python3
"""Keep or drop complete PPC carrier ambiguity arcs for blocked CV."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

try:
    from experiments.audit_ppc_ambiguity_arcs import _fold, _tow
except ModuleNotFoundError:  # Direct `python experiments/...py` execution.
    from audit_ppc_ambiguity_arcs import _fold, _tow


def write_arc_fold_mask(
    input_path: Path,
    output_path: Path,
    route: str,
    selected_fold: int,
    *,
    fold_count: int = 5,
    maximum_gap_s: float = 1.5,
    keep_selected: bool = True,
    fold_salt: str = "",
) -> dict[str, object]:
    if not 0 <= selected_fold < fold_count:
        raise ValueError("selected_fold must be in [0, fold_count)")
    observation_types: dict[str, list[str]] = {}
    current_system = ""
    active: dict[tuple[str, str], tuple[str, float, int]] = {}
    sequences: Counter[tuple[str, str]] = Counter()
    kept_fields = 0
    masked_fields = 0
    arc_ids: set[str] = set()
    epoch_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    with input_path.open(encoding="ascii", errors="replace") as source, \
            temporary.open("w", encoding="ascii", newline="") as output:
        for line in source:
            output.write(line)
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
            raise ValueError(f"RINEX END OF HEADER not found in {input_path}")

        while True:
            header = source.readline()
            if not header:
                break
            output.write(header)
            if not header.startswith(">"):
                continue
            fields = header.split()
            if len(fields) < 9:
                raise ValueError(f"invalid RINEX epoch header in {input_path}")
            tow = _tow(header)
            epoch_count += 1
            for _ in range(int(fields[-1])):
                record = source.readline()
                if not record:
                    raise ValueError(f"truncated RINEX epoch in {input_path}")
                content = record.rstrip("\r\n")
                newline = record[len(content):]
                satellite = content[:3]
                types = observation_types.get(satellite[:1], [])
                mutable = list(content)
                for index, observation_type in enumerate(types):
                    if not observation_type.startswith("L"):
                        continue
                    start = 3 + 16 * index
                    stop = start + 16
                    if start >= len(content):
                        continue
                    field = content[start:stop].ljust(16)
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
                        or lli not in ("", "0")
                    )
                    if starts_new:
                        sequence = sequences[key]
                        sequences[key] += 1
                        arc_id = (
                            f"{route}:{satellite}:{observation_type}:"
                            f"{tow:.3f}:{sequence}"
                        )
                        fold = _fold(arc_id, fold_count, fold_salt)
                    else:
                        arc_id, _, fold = prior
                    active[key] = (arc_id, tow, fold)
                    arc_ids.add(arc_id)
                    selected = fold == selected_fold
                    keep = selected if keep_selected else not selected
                    if keep:
                        kept_fields += 1
                    else:
                        masked_fields += 1
                        mutable[start:min(stop, len(mutable))] = [
                            " "
                        ] * max(0, min(stop, len(mutable)) - start)
                output.write("".join(mutable) + newline)
    temporary.replace(output_path)
    return {
        "schema": "gnss_gpu_ppc_ambiguity_arc_mask_v1",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "route": route,
        "fold_count": fold_count,
        "selected_fold": selected_fold,
        "mode": "keep" if keep_selected else "drop",
        "fold_salt": fold_salt,
        "epochs": epoch_count,
        "ambiguity_arcs": len(arc_ids),
        "kept_carrier_fields": kept_fields,
        "masked_carrier_fields": masked_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-gap", type=float, default=1.5)
    parser.add_argument("--mode", choices=("keep", "drop"), default="keep")
    parser.add_argument("--salt", default="")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = write_arc_fold_mask(
        args.input,
        args.output,
        args.route,
        args.fold,
        fold_count=args.folds,
        maximum_gap_s=args.maximum_gap,
        keep_selected=args.mode == "keep",
        fold_salt=args.salt,
    )
    result["output_sha256"] = hashlib.sha256(
        args.output.read_bytes()
    ).hexdigest()
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
