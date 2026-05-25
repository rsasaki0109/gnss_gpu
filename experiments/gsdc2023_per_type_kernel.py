"""Per-Type robust kernel + motion-sigma overrides (taroz parameters.m bundle).

Mirrors the trip-Type/phone-conditional tuning from taroz ``parameters.m``:

* ``P_robust_prm`` (PR Huber threshold): 0.1 (Street / Mix), 0.2 (Highway).
* ``D_robust_prm`` (Doppler Huber): 0.4 (Street / Mix), 0.8 (Highway), 0.2 (pixel4).
* ``L_robust_prm`` (Carrier/TDCP Huber): 0.2 (Street / Mix), 0.5 (Highway).
* ``sigma_motion``: 0.05 (Street), 0.01 (Highway / Mix), 0.1 for the ``mi8`` phone.

The native CUDA FGO solver currently honours only a scalar ``huber_k``
(applied to PR factor residuals).  This module returns just the PR
``huber_k`` and motion ``sigma_m`` for now; Doppler / TDCP per-Type
tunings will land alongside their own factor signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRIP_TYPE_STREET = "Street"
TRIP_TYPE_HIGHWAY = "Highway"
TRIP_TYPE_MIX = "Mix"


PR_HUBER_K_BY_TYPE: dict[str, float] = {
    TRIP_TYPE_STREET: 0.1,
    TRIP_TYPE_HIGHWAY: 0.2,
    TRIP_TYPE_MIX: 0.1,
}
DOPPLER_HUBER_K_BY_TYPE: dict[str, float] = {
    TRIP_TYPE_STREET: 0.4,
    TRIP_TYPE_HIGHWAY: 0.8,
    TRIP_TYPE_MIX: 0.4,
}
CARRIER_HUBER_K_BY_TYPE: dict[str, float] = {
    TRIP_TYPE_STREET: 0.2,
    TRIP_TYPE_HIGHWAY: 0.5,
    TRIP_TYPE_MIX: 0.2,
}
MOTION_SIGMA_M_BY_TYPE: dict[str, float] = {
    TRIP_TYPE_STREET: 0.05,
    TRIP_TYPE_HIGHWAY: 0.01,
    TRIP_TYPE_MIX: 0.01,
}

DOPPLER_HUBER_K_PHONE_OVERRIDES: dict[str, float] = {
    "pixel4": 0.2,
}
MOTION_SIGMA_PHONE_OVERRIDES: dict[str, float] = {
    "mi8": 0.1,
    "xiaomimi8": 0.1,
}


@dataclass(frozen=True)
class PerTypeKernel:
    trip_type: str
    pr_huber_k: float
    doppler_huber_k: float
    carrier_huber_k: float
    motion_sigma_m: float


def per_type_kernel_for(trip_type: str, phone: str = "") -> PerTypeKernel:
    """Build a ``PerTypeKernel`` for the given trip Type and phone name."""
    t = str(trip_type)
    pr_k = PR_HUBER_K_BY_TYPE.get(t, PR_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY])
    doppler_k = DOPPLER_HUBER_K_BY_TYPE.get(t, DOPPLER_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY])
    carrier_k = CARRIER_HUBER_K_BY_TYPE.get(t, CARRIER_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY])
    motion_sigma = MOTION_SIGMA_M_BY_TYPE.get(t, MOTION_SIGMA_M_BY_TYPE[TRIP_TYPE_HIGHWAY])
    phone_l = str(phone).lower()
    for needle, override in DOPPLER_HUBER_K_PHONE_OVERRIDES.items():
        if needle in phone_l:
            doppler_k = override
            break
    for needle, override in MOTION_SIGMA_PHONE_OVERRIDES.items():
        if needle in phone_l:
            motion_sigma = override
            break
    return PerTypeKernel(
        trip_type=t,
        pr_huber_k=float(pr_k),
        doppler_huber_k=float(doppler_k),
        carrier_huber_k=float(carrier_k),
        motion_sigma_m=float(motion_sigma),
    )


def load_settings_lookup(settings_csv_path: Path) -> dict[tuple[str, str], str]:
    """Return ``{(course, phone): trip_type}`` lookup from a settings CSV."""
    df = pd.read_csv(settings_csv_path)
    if not {"Course", "Phone", "Type"}.issubset(df.columns):
        raise ValueError(
            f"settings CSV missing required columns Course/Phone/Type: {settings_csv_path}"
        )
    lookup: dict[tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        lookup[(str(row.Course), str(row.Phone))] = str(row.Type)
    return lookup


def trip_type_from_data_root(
    data_root: Path,
    trip: str,
    *,
    fallback_type: str = TRIP_TYPE_HIGHWAY,
) -> str:
    """Resolve trip Type from data-root layout.

    ``trip`` is a relative path like ``train/<course>/<phone>``; the
    corresponding settings file is ``settings_train.csv`` (or
    ``settings_test.csv`` for the ``test/`` split).  Returns
    ``fallback_type`` when no entry exists or the file is missing.
    """
    parts = Path(trip).parts
    if len(parts) < 3:
        return fallback_type
    split, course, phone = parts[0], parts[1], parts[2]
    settings_name = f"settings_{split}.csv"
    settings_path = data_root / settings_name
    if not settings_path.is_file():
        return fallback_type
    lookup = load_settings_lookup(settings_path)
    return lookup.get((course, phone), fallback_type)


__all__ = [
    "CARRIER_HUBER_K_BY_TYPE",
    "DOPPLER_HUBER_K_BY_TYPE",
    "DOPPLER_HUBER_K_PHONE_OVERRIDES",
    "MOTION_SIGMA_M_BY_TYPE",
    "MOTION_SIGMA_PHONE_OVERRIDES",
    "PR_HUBER_K_BY_TYPE",
    "PerTypeKernel",
    "TRIP_TYPE_HIGHWAY",
    "TRIP_TYPE_MIX",
    "TRIP_TYPE_STREET",
    "load_settings_lookup",
    "per_type_kernel_for",
    "trip_type_from_data_root",
]
