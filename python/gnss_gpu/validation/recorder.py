from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import os

import numpy as np

from .residuals import ResidualSample


def _as_1d_array(values, dtype, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a 1-D array") from exc

    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_prn_list(prn_list) -> list[object]:
    if isinstance(prn_list, np.ndarray):
        return list(prn_list.reshape(-1))

    try:
        return list(prn_list)
    except TypeError as exc:
        raise ValueError("prn_list must be a sequence") from exc


def _format_prn(prn: object) -> str:
    if isinstance(prn, str):
        return prn
    if isinstance(prn, (int, np.integer)) and not isinstance(prn, bool):
        return f"G{int(prn):02d}"
    return str(prn)


def _validate_lengths(n: int, **arrays: np.ndarray) -> None:
    for name, arr in arrays.items():
        if arr.size != n:
            raise ValueError(
                f"{name} length {arr.size} does not match prn_list length {n}"
            )


def records_from_epoch(
    epoch: int,
    prn_list,
    residual_m,
    elevations,
    azimuths,
    is_los,
    visible,
    cn0_dbhz=None,
) -> list[ResidualSample]:
    prns = _as_prn_list(prn_list)
    residuals = _as_1d_array(residual_m, np.float64, "residual_m")
    elev = _as_1d_array(elevations, np.float64, "elevations")
    azim = _as_1d_array(azimuths, np.float64, "azimuths")
    los = _as_1d_array(is_los, np.bool_, "is_los")
    vis = _as_1d_array(visible, np.bool_, "visible")

    n = len(prns)
    _validate_lengths(
        n,
        residual_m=residuals,
        elevations=elev,
        azimuths=azim,
        is_los=los,
        visible=vis,
    )

    if cn0_dbhz is None:
        cn0 = None
    else:
        cn0 = _as_1d_array(cn0_dbhz, np.float64, "cn0_dbhz")
        _validate_lengths(n, cn0_dbhz=cn0)

    records: list[ResidualSample] = []
    for i, prn in enumerate(prns):
        if not bool(vis[i]):
            continue

        records.append(
            ResidualSample(
                epoch=int(epoch),
                prn=_format_prn(prn),
                residual_m=float(residuals[i]),
                elevation_rad=float(elev[i]),
                azimuth_rad=float(azim[i]),
                cn0_dbhz=None if cn0 is None else float(cn0[i]),
                is_los=bool(los[i]),
            )
        )

    return records


def records_from_sim_result(
    epoch: int,
    prn_list,
    result: Mapping[str, object],
    residual_m,
    cn0_dbhz=None,
) -> list[ResidualSample]:
    required = ("elevations", "azimuths", "is_los", "visible")
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"result missing required keys: {', '.join(missing)}")

    return records_from_epoch(
        epoch=epoch,
        prn_list=prn_list,
        residual_m=residual_m,
        elevations=result["elevations"],
        azimuths=result["azimuths"],
        is_los=result["is_los"],
        visible=result["visible"],
        cn0_dbhz=cn0_dbhz,
    )


def write_csv(
    samples: Iterable[ResidualSample],
    path: str | os.PathLike[str],
) -> None:
    fieldnames = [
        "epoch",
        "prn",
        "residual_m",
        "elevation_rad",
        "azimuth_rad",
        "cn0_dbhz",
        "is_los",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            writer.writerow(
                {
                    "epoch": sample.epoch,
                    "prn": sample.prn,
                    "residual_m": sample.residual_m,
                    "elevation_rad": sample.elevation_rad,
                    "azimuth_rad": sample.azimuth_rad,
                    "cn0_dbhz": sample.cn0_dbhz,
                    "is_los": sample.is_los,
                }
            )
