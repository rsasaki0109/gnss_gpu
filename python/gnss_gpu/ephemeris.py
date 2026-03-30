"""Python interface for GPU-accelerated broadcast ephemeris computation."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from gnss_gpu.io.nav_rinex import NavMessage

try:
    from gnss_gpu._gnss_gpu_ephemeris import (
        compute_satellite_position,
        compute_satellite_position_batch,
        EPHEMERIS_PARAMS_SIZE,
    )

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# GPS constants matching the CUDA implementation
GPS_MU = 3.986005e14  # Earth gravitational parameter [m^3/s^2]
GPS_OMEGA_E = 7.2921151467e-5  # Earth rotation rate [rad/s]
GPS_F = -4.442807633e-10  # relativistic correction constant [s/m^0.5]
GPS_WEEK_SEC = 604800.0  # seconds per GPS week


def _nav_to_params_bytes(nav: NavMessage) -> bytes:
    """Convert NavMessage to raw bytes matching EphemerisParams struct layout."""
    # Must match the C++ struct field order exactly:
    # sqrt_a, e, i0, omega0, omega, M0, delta_n, omega_dot, idot,
    # cuc, cus, crc, crs, cic, cis, toe, af0, af1, af2, toc, tgd, [pad], week
    doubles = [
        nav.sqrt_a,
        nav.e,
        nav.i0,
        nav.omega0,
        nav.omega,
        nav.M0,
        nav.delta_n,
        nav.omega_dot,
        nav.idot,
        nav.cuc,
        nav.cus,
        nav.crc,
        nav.crs,
        nav.cic,
        nav.cis,
        nav.toe,
        nav.af0,
        nav.af1,
        nav.af2,
        nav.toc_seconds,
        nav.tgd,
    ]
    # Pack as doubles + int (with padding for alignment)
    data = struct.pack(f"<{len(doubles)}d", *doubles)
    # week is an int at the end; struct has padding to align
    data += struct.pack("<i", nav.week)
    # Pad to match struct size (align to 8 bytes)
    remainder = len(data) % 8
    if remainder:
        data += b"\x00" * (8 - remainder)
    return data


class Ephemeris:
    """GPU-accelerated broadcast ephemeris computation.

    Computes GPS satellite positions from broadcast navigation messages
    using the IS-GPS-200 algorithm on the GPU.
    """

    def __init__(self, nav_messages: dict[int, list[NavMessage]]) -> None:
        """Initialize from parsed NAV RINEX messages.

        Args:
            nav_messages: dict mapping PRN to list of NavMessage, as returned
                by read_nav_rinex().
        """
        self._nav_messages = nav_messages
        self._prn_list = sorted(nav_messages.keys())

    @property
    def available_prns(self) -> list[int]:
        """List of PRN numbers with available ephemeris data."""
        return list(self._prn_list)

    def select_ephemeris(self, prn: int, gps_time: float) -> NavMessage | None:
        """Select best ephemeris for given PRN and GPS time of week.

        Chooses the ephemeris with toe closest to the requested time.

        Args:
            prn: satellite PRN number
            gps_time: GPS seconds of week

        Returns:
            Best matching NavMessage, or None if PRN not found.
        """
        if prn not in self._nav_messages:
            return None
        messages = self._nav_messages[prn]
        if not messages:
            return None

        best = None
        best_dt = float("inf")
        for msg in messages:
            dt = abs(gps_time - msg.toe)
            # Handle week crossover
            if dt > GPS_WEEK_SEC / 2.0:
                dt = GPS_WEEK_SEC - dt
            if dt < best_dt:
                best_dt = dt
                best = msg
        return best

    def _build_params(
        self, gps_time: float, prn_list: list[int] | None = None
    ) -> tuple[np.ndarray, list[int]]:
        """Build packed EphemerisParams array for GPU computation.

        Returns:
            params_flat: numpy array of raw bytes matching EphemerisParams struct
            used_prns: list of PRNs actually included
        """
        if prn_list is None:
            prn_list = self._prn_list

        used_prns = []
        params_bytes = b""
        for prn in prn_list:
            nav = self.select_ephemeris(prn, gps_time)
            if nav is not None:
                params_bytes += _nav_to_params_bytes(nav)
                used_prns.append(prn)

        if not used_prns:
            return np.array([], dtype=np.float64), []

        params_flat = np.frombuffer(params_bytes, dtype=np.float64)
        return params_flat, used_prns

    def compute(
        self, gps_time: float, prn_list: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Compute satellite positions at given GPS time of week.

        Args:
            gps_time: GPS seconds of week
            prn_list: list of PRNs to compute (None = all available)

        Returns:
            sat_ecef: array of shape [n_sat, 3] ECEF positions [m]
            sat_clk: array of shape [n_sat] clock corrections [s]
            used_prns: list of PRN numbers corresponding to output rows
        """
        if not HAS_GPU:
            return self._compute_cpu(gps_time, prn_list)

        params_flat, used_prns = self._build_params(gps_time, prn_list)
        if not used_prns:
            return np.array([]).reshape(0, 3), np.array([]), []

        n_sat = len(used_prns)
        sat_ecef, sat_clk = compute_satellite_position(params_flat, gps_time, n_sat)
        return np.asarray(sat_ecef), np.asarray(sat_clk), used_prns

    def compute_batch(
        self, gps_times: np.ndarray | list[float], prn_list: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Batch computation for multiple epochs.

        Args:
            gps_times: array of GPS seconds of week [n_epoch]
            prn_list: list of PRNs to compute (None = all available)

        Returns:
            sat_ecef: array of shape [n_epoch, n_sat, 3] ECEF positions [m]
            sat_clk: array of shape [n_epoch, n_sat] clock corrections [s]
            used_prns: list of PRN numbers corresponding to output columns
        """
        gps_times = np.asarray(gps_times, dtype=np.float64)

        if not HAS_GPU:
            return self._compute_batch_cpu(gps_times, prn_list)

        # Use first epoch time for ephemeris selection
        ref_time = gps_times[0] if len(gps_times) > 0 else 0.0
        params_flat, used_prns = self._build_params(ref_time, prn_list)
        if not used_prns:
            n_epoch = len(gps_times)
            return (
                np.array([]).reshape(n_epoch, 0, 3),
                np.array([]).reshape(n_epoch, 0),
                [],
            )

        n_sat = len(used_prns)
        sat_ecef, sat_clk = compute_satellite_position_batch(
            params_flat, gps_times, n_sat
        )
        return np.asarray(sat_ecef), np.asarray(sat_clk), used_prns

    # --- CPU fallback implementation ---

    @staticmethod
    def _kepler_cpu(M: float, e: float) -> float:
        """Solve Kepler's equation on CPU (Newton-Raphson)."""
        E = M
        for _ in range(10):
            dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
            E += dE
            if abs(dE) < 1e-15:
                break
        return E

    @staticmethod
    def _compute_single_cpu(nav: NavMessage, gps_time: float) -> tuple[np.ndarray, float]:
        """Compute single satellite position on CPU."""
        a = nav.sqrt_a ** 2
        n0 = np.sqrt(GPS_MU / (a ** 3))
        n = n0 + nav.delta_n

        tk = gps_time - nav.toe
        if tk > GPS_WEEK_SEC / 2.0:
            tk -= GPS_WEEK_SEC
        if tk < -GPS_WEEK_SEC / 2.0:
            tk += GPS_WEEK_SEC

        M = nav.M0 + n * tk
        E = Ephemeris._kepler_cpu(M, nav.e)

        sinE = np.sin(E)
        cosE = np.cos(E)
        denom = 1.0 - nav.e * cosE
        sinv = np.sqrt(1.0 - nav.e ** 2) * sinE / denom
        cosv = (cosE - nav.e) / denom
        v = np.arctan2(sinv, cosv)

        phi = v + nav.omega
        sin2phi = np.sin(2.0 * phi)
        cos2phi = np.cos(2.0 * phi)

        du = nav.cuc * cos2phi + nav.cus * sin2phi
        dr = nav.crc * cos2phi + nav.crs * sin2phi
        di = nav.cic * cos2phi + nav.cis * sin2phi

        u = phi + du
        r = a * (1.0 - nav.e * cosE) + dr
        i = nav.i0 + nav.idot * tk + di

        xp = r * np.cos(u)
        yp = r * np.sin(u)

        Omega = nav.omega0 + (nav.omega_dot - GPS_OMEGA_E) * tk - GPS_OMEGA_E * nav.toe

        cosO = np.cos(Omega)
        sinO = np.sin(Omega)
        cosi = np.cos(i)
        sini = np.sin(i)

        pos = np.array([
            xp * cosO - yp * cosi * sinO,
            xp * sinO + yp * cosi * cosO,
            yp * sini,
        ])

        # Clock correction
        dt = gps_time - nav.toc_seconds
        if dt > GPS_WEEK_SEC / 2.0:
            dt -= GPS_WEEK_SEC
        if dt < -GPS_WEEK_SEC / 2.0:
            dt += GPS_WEEK_SEC
        dtr = GPS_F * nav.e * nav.sqrt_a * sinE
        clk = nav.af0 + nav.af1 * dt + nav.af2 * dt * dt + dtr - nav.tgd

        return pos, clk

    def _compute_cpu(
        self, gps_time: float, prn_list: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """CPU fallback for single-epoch computation."""
        if prn_list is None:
            prn_list = self._prn_list

        positions = []
        clocks = []
        used_prns = []

        for prn in prn_list:
            nav = self.select_ephemeris(prn, gps_time)
            if nav is None:
                continue
            pos, clk = self._compute_single_cpu(nav, gps_time)
            positions.append(pos)
            clocks.append(clk)
            used_prns.append(prn)

        if not used_prns:
            return np.array([]).reshape(0, 3), np.array([]), []

        return np.array(positions), np.array(clocks), used_prns

    def _compute_batch_cpu(
        self, gps_times: np.ndarray, prn_list: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """CPU fallback for batch computation."""
        if prn_list is None:
            prn_list = self._prn_list

        # Determine valid PRNs from first epoch
        ref_time = gps_times[0] if len(gps_times) > 0 else 0.0
        used_prns = []
        navs = []
        for prn in prn_list:
            nav = self.select_ephemeris(prn, ref_time)
            if nav is not None:
                used_prns.append(prn)
                navs.append(nav)

        n_epoch = len(gps_times)
        n_sat = len(used_prns)

        if n_sat == 0:
            return (
                np.array([]).reshape(n_epoch, 0, 3),
                np.array([]).reshape(n_epoch, 0),
                [],
            )

        positions = np.zeros((n_epoch, n_sat, 3))
        clocks = np.zeros((n_epoch, n_sat))

        for i, t in enumerate(gps_times):
            for j, nav in enumerate(navs):
                pos, clk = self._compute_single_cpu(nav, t)
                positions[i, j] = pos
                clocks[i, j] = clk

        return positions, clocks, used_prns
