#!/usr/bin/env python3
"""Soft-mask RINEX 3 observation file by lowering SNR (S*) values for NLOS sats.

Approach: instead of removing NLOS sat data lines (hard mask, which breaks AR),
replace each NLOS sat's signal-strength (S*) observations with a low dBHz value
(default 20). With ``gnss_solve --rtk-snr-weighting --rtk-snr-reference-dbhz 45
--rtk-snr-max-variance-scale 25``, low-SNR observations get variance inflated,
effectively down-weighting NLOS satellites WITHOUT removing them. AR still runs
on the full obs set.

Parses RINEX 3 obs type header per system (``SYS / # / OBS TYPES``) to locate
S* columns. Each obs slot is F14.3 + I1 + I1 = 16 chars wide.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

GPS_EPOCH = datetime(1980, 1, 6)
LOW_SNR = "        20.000"  # 14-char F14.3 right-justified


def rinex_dt_to_gps_tow(year, month, day, hour, minute, sec):
    sec_f = float(sec)
    sec_int = int(sec_f)
    frac = sec_f - sec_int
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), sec_int)
    delta = dt - GPS_EPOCH
    days = delta.days
    sec_in_day = delta.seconds + frac
    dow = days % 7
    tow = dow * 86400.0 + sec_in_day
    return round(tow, 1)


def parse_obs_types_header(in_obs: Path) -> dict[str, list[str]]:
    """Read SYS / # / OBS TYPES from header. Returns {sys_char: [obs_type, ...]}.

    Each obs type entry has a one-char header prefix (sys char) and a list of
    3-char type strings (C1C, L1C, D1C, S1C, ...). The list can span continuation
    lines (no sys char on continuation, just spaces+types).
    """
    obs_types: dict[str, list[str]] = {}
    cur_sys = None
    cur_n = 0
    with open(in_obs) as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if "SYS / # / OBS TYPES" not in line:
                continue
            sys_char = line[0]
            if sys_char == " ":
                # Continuation line; reuse cur_sys
                pass
            else:
                cur_sys = sys_char
                try:
                    cur_n = int(line[3:6].strip())
                except ValueError:
                    cur_n = 0
                obs_types[cur_sys] = []
            # Parse types from this line. They start at col 7 onwards (after
            # "S NN " of header) for first line, or after leading whitespace
            # for continuation. Each type is 4 chars (3-char code + space).
            type_section = line[6:60]  # 54 chars to cover ~13 types
            for j in range(0, len(type_section), 4):
                t = type_section[j:j + 4].strip()
                if t and len(t) == 3 and cur_sys is not None:
                    obs_types[cur_sys].append(t)
    return obs_types


def find_snr_indices(obs_types_per_sys: dict[str, list[str]]) -> dict[str, list[int]]:
    """Find indices of S* observation types within each system's obs list."""
    snr_idx: dict[str, list[int]] = {}
    for sys_char, types in obs_types_per_sys.items():
        snr_idx[sys_char] = [i for i, t in enumerate(types) if t.startswith("S")]
    return snr_idx


def load_nlos_csv(p: Path) -> dict[float, set[str]]:
    nlos_map: dict[float, set[str]] = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                tow = round(float(row["tow"]), 1)
                if int(row["is_los"]) == 0:
                    nlos_map.setdefault(tow, set()).add(row["prn"])
            except (ValueError, KeyError):
                continue
    return nlos_map


def patch_sat_line(sat_line: str, snr_indices_for_sys: list[int]) -> str:
    """Replace S* observation values in sat_line with LOW_SNR."""
    # Each obs is 16 chars wide starting at col 3. value: chars [c, c+14), LLI: c+14, SS: c+15
    # We modify the value but keep LLI and SS bytes intact.
    line = list(sat_line.rstrip("\n"))
    for idx in snr_indices_for_sys:
        start = 3 + 16 * idx
        end = start + 14  # F14.3 value section
        if end > len(line):
            # Sat line shorter than expected (obs not present); skip
            continue
        # Only modify if the slot currently has a non-blank value (sat actually
        # reported this signal). RINEX 3 leaves blank for missing obs.
        existing = "".join(line[start:end]).strip()
        if not existing:
            continue
        # Replace
        new_chars = list(LOW_SNR)
        line[start:end] = new_chars
    return "".join(line) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-obs", type=Path, required=True)
    ap.add_argument("--nlos-csv", type=Path, required=True)
    ap.add_argument("--out-obs", type=Path, required=True)
    args = ap.parse_args()

    print("Parsing obs types from header ...")
    obs_types = parse_obs_types_header(args.in_obs)
    snr_idx = find_snr_indices(obs_types)
    for sys, idx in snr_idx.items():
        types = obs_types.get(sys, [])
        s_types = [types[i] for i in idx]
        print(f"  System {sys}: {len(types)} obs types, S* indices={idx} ({s_types})")

    nlos_map = load_nlos_csv(args.nlos_csv)
    print(f"\nNLOS map: {len(nlos_map)} epochs with NLOS sats")

    n_epoch = 0
    n_modified = 0

    args.out_obs.parent.mkdir(parents=True, exist_ok=True)
    with open(args.in_obs) as f_in, open(args.out_obs, "w") as f_out:
        in_header = True
        for line in f_in:
            if in_header:
                f_out.write(line)
                if "END OF HEADER" in line:
                    in_header = False
                continue
            if line.startswith(">"):
                parts = line[1:].split()
                if len(parts) < 8:
                    f_out.write(line)
                    continue
                tow = rinex_dt_to_gps_tow(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
                n_sat = int(parts[7])
                nlos_prns_here = nlos_map.get(tow, set())
                f_out.write(line)
                n_epoch += 1
                for _ in range(n_sat):
                    sat_line = next(f_in)
                    prn = sat_line[:3]
                    if prn in nlos_prns_here:
                        sys_char = prn[0]
                        idx = snr_idx.get(sys_char, [])
                        sat_line = patch_sat_line(sat_line, idx)
                        n_modified += 1
                    f_out.write(sat_line)
            else:
                f_out.write(line)

    print(f"\nProcessed {n_epoch} epochs")
    print(f"  Modified SNR fields on {n_modified} sat-line(s) (NLOS soft mask)")


if __name__ == "__main__":
    main()
