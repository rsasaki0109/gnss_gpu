from __future__ import annotations

import math

import numpy as np

EPS0 = 8.8541878128e-12
GPS_L1_FREQ = 1575.42e6

# Material -> (relative permittivity eps_r, conductivity sigma [S/m]).
# "metal" is modelled as a very high-loss dielectric so that it flows through
# the normal Fresnel equations: this correctly yields |R_par| = |R_perp| = 1
# while the RHCP co-pol coefficient (R_par + R_perp) / 2 -> 0, i.e. a metal
# surface flips an incident RHCP wave to LHCP and is rejected by an RHCP
# antenna. A flat all-ones special case would lose that polarization rejection.
MATERIALS: dict[str, tuple[float, float]] = {
    "concrete": (5.31, 0.0326),
    "brick": (3.75, 0.038),
    "glass": (6.27, 0.0043),
    "wood": (1.99, 0.0047),
    "dry_ground": (3.0, 0.015),
    "wet_ground": (20.0, 0.5),
    "metal": (1.0e9, 1.0e9),
}


def _material_key(material: str) -> str:
    return material.strip().lower()


def _clip_incidence_angle(incidence_angle_rad):
    angle = np.asarray(incidence_angle_rad, dtype=float)
    return np.clip(angle, 0.0, 0.5 * math.pi)


def _maybe_scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return value


def complex_permittivity(material, freq_hz=GPS_L1_FREQ) -> complex:
    """Return complex relative permittivity eps = eps_r - j*sigma/(2*pi*f*eps0)."""
    if isinstance(material, str):
        key = _material_key(material)
        if key not in MATERIALS:
            raise ValueError(f"unknown material: {material!r}")
        eps_r, sigma = MATERIALS[key]
    elif isinstance(material, tuple) and len(material) == 2:
        eps_r, sigma = material
    else:
        try:
            return complex(material)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "material must be a material name, an (eps_r, sigma) tuple, or complex"
            ) from exc

    freq = float(freq_hz)
    if freq <= 0.0:
        raise ValueError("freq_hz must be positive")

    return complex(float(eps_r), -float(sigma) / (2.0 * math.pi * freq * EPS0))


def fresnel_coefficients(incidence_angle_rad, eps) -> tuple[complex, complex]:
    """Return Fresnel field coefficients (R_parallel, R_perpendicular)."""
    theta = _clip_incidence_angle(incidence_angle_rad)
    eps_c = np.asarray(complex_permittivity(eps), dtype=np.complex128)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_theta_t_term = np.sqrt(eps_c - sin_theta * sin_theta + 0j)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_perp = (cos_theta - cos_theta_t_term) / (
            cos_theta + cos_theta_t_term
        )
        r_par = (eps_c * cos_theta - cos_theta_t_term) / (
            eps_c * cos_theta + cos_theta_t_term
        )

    return _maybe_scalar(r_par), _maybe_scalar(r_perp)


_POLARIZATION_ALIASES = {
    "rhcp": "rhcp",
    "rhcp_copol": "rhcp",
    "copol": "rhcp",
    "co": "rhcp",
    "rhcp_cross": "rhcp_cross",
    "cross": "rhcp_cross",
    "crosspol": "rhcp_cross",
    "lhcp": "rhcp_cross",
    "parallel": "parallel",
    "par": "parallel",
    "p": "parallel",
    "vertical": "parallel",
    "perpendicular": "perpendicular",
    "perp": "perpendicular",
    "s": "perpendicular",
    "horizontal": "perpendicular",
    "average": "average",
    "avg": "average",
}


def _canonical_polarization(polarization: str) -> str:
    key = str(polarization).strip().lower()
    if key not in _POLARIZATION_ALIASES:
        raise ValueError(f"unknown polarization: {polarization!r}")
    return _POLARIZATION_ALIASES[key]


def reflection_coefficient_complex(
    incidence_angle_rad,
    material="concrete",
    freq_hz=GPS_L1_FREQ,
    polarization="rhcp",
):
    """Return complex field reflection coefficient for the requested polarization."""
    pol = _canonical_polarization(polarization)

    eps = complex_permittivity(material, freq_hz=freq_hz)
    r_par, r_perp = fresnel_coefficients(incidence_angle_rad, eps)

    r_par = np.asarray(r_par, dtype=np.complex128)
    r_perp = np.asarray(r_perp, dtype=np.complex128)

    if pol == "rhcp":
        coeff = 0.5 * (r_par + r_perp)
    elif pol == "rhcp_cross":
        coeff = 0.5 * (r_par - r_perp)
    elif pol == "parallel":
        coeff = r_par
    elif pol == "perpendicular":
        coeff = r_perp
    elif pol == "average":
        coeff = 0.5 * (np.abs(r_par) + np.abs(r_perp))
    else:
        raise ValueError(f"unknown polarization: {polarization!r}")

    return _maybe_scalar(coeff)


def reflection_coefficient(
    incidence_angle_rad,
    material="concrete",
    freq_hz=GPS_L1_FREQ,
    polarization="rhcp",
) -> float:
    """Return reflection amplitude magnitude clipped to [0, 1]."""
    coeff = reflection_coefficient_complex(
        incidence_angle_rad,
        material=material,
        freq_hz=freq_hz,
        polarization=polarization,
    )
    magnitude = np.clip(np.abs(coeff), 0.0, 1.0)
    return _maybe_scalar(magnitude)


__all__ = [
    "EPS0",
    "GPS_L1_FREQ",
    "MATERIALS",
    "complex_permittivity",
    "fresnel_coefficients",
    "reflection_coefficient",
    "reflection_coefficient_complex",
]
