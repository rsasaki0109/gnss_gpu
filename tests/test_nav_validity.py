from __future__ import annotations

import math
from dataclasses import dataclass

from gnss_gpu.nav_validity import (
    BEIDOU_GEO_PRNS,
    is_beidou_geo,
    is_broadcast_usable,
    is_glonass_broadcast_valid,
    is_kepler_broadcast_valid,
)


@dataclass
class _StubNav:
    """Minimum duck-typed NavMessage surface used by validity predicates."""

    system: str = "G"
    prn: int = 5
    sqrt_a: float = 5153.5
    e: float = 0.01
    glo_px_m: float = 0.0
    glo_py_m: float = 0.0
    glo_pz_m: float = 0.0


# --- is_kepler_broadcast_valid -------------------------------------------


def test_kepler_validity_accepts_finite_in_range():
    assert is_kepler_broadcast_valid(_StubNav()) is True


def test_kepler_validity_rejects_none():
    assert is_kepler_broadcast_valid(None) is False


def test_kepler_validity_rejects_glonass_and_sbas():
    assert is_kepler_broadcast_valid(_StubNav(system="R")) is False
    assert is_kepler_broadcast_valid(_StubNav(system="S")) is False


def test_kepler_validity_rejects_nonpositive_sqrt_a():
    assert is_kepler_broadcast_valid(_StubNav(sqrt_a=0.0)) is False
    assert is_kepler_broadcast_valid(_StubNav(sqrt_a=-1.0)) is False


def test_kepler_validity_rejects_nonfinite_inputs():
    assert is_kepler_broadcast_valid(_StubNav(sqrt_a=float("nan"))) is False
    assert is_kepler_broadcast_valid(_StubNav(sqrt_a=float("inf"))) is False
    assert is_kepler_broadcast_valid(_StubNav(e=float("nan"))) is False


def test_kepler_validity_rejects_out_of_range_eccentricity():
    assert is_kepler_broadcast_valid(_StubNav(e=-0.1)) is False
    assert is_kepler_broadcast_valid(_StubNav(e=1.5)) is False


# --- is_glonass_broadcast_valid ------------------------------------------


def _glo_stub(px_m: float = 0.0, py_m: float = 0.0, pz_m: float = 0.0) -> _StubNav:
    return _StubNav(
        system="R",
        prn=1,
        sqrt_a=0.0,
        e=0.0,
        glo_px_m=px_m,
        glo_py_m=py_m,
        glo_pz_m=pz_m,
    )


def test_glonass_validity_accepts_realistic_orbit_radius():
    # GLONASS MEOs sit near r ≈ 25,500 km.
    assert is_glonass_broadcast_valid(_glo_stub(px_m=2.55e7, py_m=0.0, pz_m=0.0)) is True


def test_glonass_validity_rejects_other_systems():
    assert is_glonass_broadcast_valid(_StubNav(system="G")) is False


def test_glonass_validity_rejects_zero_radius():
    assert is_glonass_broadcast_valid(_glo_stub()) is False


def test_glonass_validity_rejects_too_small_or_too_large_radius():
    assert is_glonass_broadcast_valid(_glo_stub(px_m=1.0e6)) is False  # ~1000 km
    assert is_glonass_broadcast_valid(_glo_stub(px_m=1.0e8)) is False  # ~100,000 km


def test_glonass_validity_rejects_nonfinite_components():
    assert is_glonass_broadcast_valid(_glo_stub(px_m=math.nan)) is False


# --- is_beidou_geo --------------------------------------------------------


def test_beidou_geo_lists_legacy_prns():
    # BDS-2 GEO PRNs 1-5 must all be flagged.
    for prn in (1, 2, 3, 4, 5):
        assert is_beidou_geo(_StubNav(system="C", prn=prn)) is True


def test_beidou_geo_lists_bds3_geo_prns():
    for prn in (59, 60, 61, 62, 63):
        assert is_beidou_geo(_StubNav(system="C", prn=prn)) is True


def test_beidou_geo_does_not_flag_meo_or_igso():
    assert is_beidou_geo(_StubNav(system="C", prn=20)) is False  # BDS-3 MEO range


def test_beidou_geo_does_not_flag_other_constellations():
    assert is_beidou_geo(_StubNav(system="G", prn=1)) is False


def test_beidou_geo_constant_is_frozen_set():
    assert isinstance(BEIDOU_GEO_PRNS, frozenset)
    assert {1, 2, 3, 4, 5, 59, 60, 61, 62, 63}.issubset(BEIDOU_GEO_PRNS)


# --- is_broadcast_usable -------------------------------------------------


def test_broadcast_usable_routes_gps_to_kepler():
    assert is_broadcast_usable(_StubNav()) is True


def test_broadcast_usable_routes_glonass_via_state_vector():
    assert is_broadcast_usable(_glo_stub(px_m=2.55e7)) is True


def test_broadcast_usable_excludes_beidou_geo_even_if_kepler_valid():
    nav = _StubNav(system="C", prn=2)  # GEO, but Kepler elements look fine
    assert is_broadcast_usable(nav) is False


def test_broadcast_usable_returns_false_for_none():
    assert is_broadcast_usable(None) is False
