#!/usr/bin/env python3
"""Fingerprint ``exp_ppc_ctrbpf_fgo._run_ctrbpf_on_segment`` on synthetic data.

Refactor guard for the CT-RBPF segment runner (internal_docs/plan.md item 6).
It builds a deterministic synthetic PPC-like segment with satellite velocity,
Doppler, a hybrid RTK track and three RTK-diagnostic candidates. There is no
base station, so the DD branches are not exercised. It then runs the
requested ``--methods`` variants, passing inputs per variant the way ``main``
does, and prints a SHA-256 of every returned value per variant.

Run it before and after a change on the same machine and compare. Hashes are
only comparable on the same GPU and build, because the particle filter runs
on CUDA and needs the ``_gnss_gpu_pf_device`` extension.

    PYTHONPATH=python python experiments/fingerprint_ctrbpf_segment.py > before.json
    # ... refactor ...
    PYTHONPATH=python python experiments/fingerprint_ctrbpf_segment.py > after.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import exp_ppc_ctrbpf_fgo as ctrbpf  # noqa: E402

DEFAULT_METHODS = "pf,pf+pu,rbpf,rbpf+pu"
_GPS_L1_HZ = 1575.42e6
_C = 299_792_458.0


def synthetic_segment(n_epochs: int = 60, seed: int = 3) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return ``(data, wls_positions, truth)`` shaped like the PPC loader output."""

    rng = np.random.default_rng(seed)
    lat, lon, h = np.radians(35.65), np.radians(139.78), 40.0
    a, e2 = 6378137.0, 6.69437999014e-3
    n_rad = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x0 = np.array([
        (n_rad + h) * np.cos(lat) * np.cos(lon),
        (n_rad + h) * np.cos(lat) * np.sin(lon),
        (n_rad * (1.0 - e2) + h) * np.sin(lat),
    ])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = x0 / np.linalg.norm(x0)
    el = np.radians([70, 55, 40, 35, 30, 25, 50, 20])
    az = np.radians([0, 60, 120, 180, 240, 300, 200, 90])
    dirs = (np.cos(el) * np.sin(az))[:, None] * east + (np.cos(el) * np.cos(az))[:, None] * north
    dirs = dirs + np.sin(el)[:, None] * up
    sats0 = x0 + 2.02e7 * dirs
    sat_vel = np.cross(dirs, up) * 3000.0
    rx_vel = 5.0 * east + 0.5 * north
    dt = 0.2
    wavelength = _C / _GPS_L1_HZ
    clock_bias0, clock_drift = 1234.0, 2.0

    times, sat_l, pr_l, w_l, sv_l, dop_l, truth, cb = [], [], [], [], [], [], [], []
    for k in range(n_epochs):
        x = x0 + rx_vel * (k * dt)
        sat = sats0 + sat_vel * (k * dt)
        los = (sat - x) / np.linalg.norm(sat - x, axis=1)[:, None]
        rng_m = np.linalg.norm(sat - x, axis=1)
        clk = clock_bias0 + clock_drift * k * dt
        pr = rng_m + clk + rng.normal(0.0, 3.0, len(sat))
        range_rate = np.sum(los * (sat_vel - rx_vel), axis=1) + clock_drift
        doppler_hz = -range_rate / wavelength + rng.normal(0.0, 0.5, len(sat))
        times.append(300000.0 + k * dt)
        sat_l.append(sat)
        pr_l.append(pr)
        w_l.append(np.ones(len(sat)))
        sv_l.append(sat_vel.copy())
        dop_l.append(doppler_hz)
        truth.append(x)
        cb.append(clk)

    data = dict(
        n_epochs=n_epochs,
        times=np.asarray(times),
        dt=dt,
        sat_ecef=sat_l,
        pseudoranges=pr_l,
        weights=w_l,
        sat_velocity=sv_l,
        doppler_hz=dop_l,
        system_ids=[np.zeros(len(el), dtype=np.int32) for _ in range(n_epochs)],
        used_prns=[[f"G{p:02d}" for p in range(1, len(el) + 1)] for _ in range(n_epochs)],
    )
    wls = np.hstack([
        np.asarray(truth) + rng.normal(0.0, 2.0, (n_epochs, 3)),
        np.asarray(cb)[:, None],
    ])
    return data, wls, np.asarray(truth)


def synthetic_side_inputs(data: dict, truth: np.ndarray, seed: int = 5) -> dict:
    """Hybrid RTK track, reference map and RTK-diag candidates keyed by 0.1 s TOW."""

    rng = np.random.default_rng(seed)
    tows = [round(float(t), 1) for t in data["times"]]
    hybrid_pos = {t: x + rng.normal(0.0, 0.05, 3) for t, x in zip(tows, truth)}
    hybrid_status = {t: (4 if k % 3 else 5) for k, t in enumerate(tows)}
    reference_pos = {t: x.copy() for t, x in zip(tows, truth)}
    candidates = []
    for label, offset_m, status, ratio in (("near", 0.05, "4", 8.0), ("mid", 0.4, "4", 3.2), ("far", 3.0, "5", 1.1)):
        pos = {t: x + rng.normal(0.0, offset_m, 3) for t, x in zip(tows, truth)}
        diag = {
            t: {
                "tow": f"{t:.1f}",
                "final_status": status if k % 4 else "5",
                "final_ratio": f"{ratio + 0.1 * (k % 5):.3f}",
                "final_residual_rms": f"{0.02 + 0.01 * (k % 7):.3f}",
                "output_added": "1",
            }
            for k, t in enumerate(tows)
        }
        candidates.append((label, pos, diag))
    return dict(hybrid_pos=hybrid_pos, hybrid_status=hybrid_status, reference_pos=reference_pos, candidates=candidates)


def _digest(value: object) -> str:
    h = hashlib.sha256()

    def feed(v: object) -> None:
        if isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif dataclasses.is_dataclass(v) and not isinstance(v, type):
            for f in dataclasses.fields(v):
                h.update(f.name.encode())
                feed(getattr(v, f.name))
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                feed(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(f"{type(v).__name__}{len(v)}".encode())
            for item in v:
                feed(item)
        else:
            h.update(repr(v).encode())

    feed(value)
    return h.hexdigest()


def fingerprint(methods: str, n_epochs: int, seed: int) -> dict[str, dict[str, str]]:
    args = ctrbpf._build_arg_parser().parse_args(["--methods", methods])
    data, wls, truth = synthetic_segment(n_epochs=n_epochs, seed=seed)
    side = synthetic_side_inputs(data, truth)
    result: dict[str, dict[str, str]] = {}
    for config in ctrbpf._config_variants(args):
        hybrid = config.enable_hybrid_pu
        out = ctrbpf._run_ctrbpf_on_segment(
            data,
            wls,
            config,
            hybrid_pos=side["hybrid_pos"] if hybrid else None,
            hybrid_status=side["hybrid_status"] if hybrid else None,
            reference_pos=side["reference_pos"],
            rtkdiag_candidates=side["candidates"] if config.enable_rtkdiag_pf_rescue else None,
        )
        # out[1] is wall-clock ms/epoch and is not deterministic.
        named = {"positions": out[0], **{f"stats_{i}": v for i, v in enumerate(out[2:], start=2)}}
        result[config.method_label] = {key: _digest(val)[:16] for key, val in named.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fingerprint the CT-RBPF segment runner on synthetic data.")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(fingerprint(args.methods, args.epochs, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
