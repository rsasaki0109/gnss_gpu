"""Python interface for GPU-accelerated broadcast ephemeris computation."""

from __future__ import annotations

import math
import struct

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

GAL_MU = 3.986004418e14
GAL_OMEGA_E = 7.2921151467e-5
GAL_F = -4.442807309e-10

QZS_MU = GPS_MU
QZS_OMEGA_E = GPS_OMEGA_E
QZS_F = GPS_F


def _constellation_constants(system: str) -> tuple[float, float, float]:
    system = system.upper()
    if system == "E":
        return GAL_MU, GAL_OMEGA_E, GAL_F
    if system == "J":
        return QZS_MU, QZS_OMEGA_E, QZS_F
    return GPS_MU, GPS_OMEGA_E, GPS_F


def _normalize_sat_id(sat: int | str) -> int | str:
    if isinstance(sat, str):
        sat_id = sat.strip().upper()
        if not sat_id:
            return sat_id
        system = sat_id[0]
        prn = sat_id[1:].strip().replace(" ", "")
        if system.isalpha() and prn.isdigit():
            return f"{system}{int(prn):02d}"
        return sat_id.replace(" ", "")
    return int(sat)


def _obs_band(obs_code: str | None) -> str:
    if not obs_code or len(obs_code) < 2:
        return ""
    return obs_code[1]


def _galileo_priority(nav: NavMessage, obs_code: str | None) -> int:
    band = _obs_band(obs_code)
    data_sources = int(round(nav.data_sources or nav.codes_on_l2))
    if band == "1":
        return (
            4 * int(bool(data_sources & 0x001))
            + 2 * int(bool(data_sources & 0x200))
            + int(bool(data_sources & 0x004))
        )
    if band == "5":
        return 4 * int(bool(data_sources & 0x002)) + 2 * int(bool(data_sources & 0x100))
    if band == "7":
        return 4 * int(bool(data_sources & 0x004)) + 2 * int(bool(data_sources & 0x200))
    return 0


def _group_delay(nav: NavMessage, obs_code: str | None) -> float:
    if nav.system != "E":
        return nav.tgd

    band = _obs_band(obs_code)
    data_sources = int(round(nav.data_sources or nav.codes_on_l2))
    if band == "1":
        if data_sources & 0x200:
            return nav.bgd_e5b_e1
        if data_sources & 0x100:
            return nav.bgd_e5a_e1
        return nav.bgd_e5b_e1 if nav.bgd_e5b_e1 != 0.0 else nav.bgd_e5a_e1
    return nav.tgd


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

    def __init__(self, nav_messages: dict[int | str, list[NavMessage]]) -> None:
        """Initialize from parsed NAV RINEX messages.

        Args:
            nav_messages: dict mapping PRN to list of NavMessage, as returned
                by read_nav_rinex().
        """
        self._nav_messages = nav_messages
        self._prn_list = sorted(nav_messages.keys(), key=str)

    @property
    def available_prns(self) -> list[int | str]:
        """List of PRN numbers with available ephemeris data."""
        return list(self._prn_list)

    def select_ephemeris(
        self,
        prn: int | str,
        gps_time: float,
        obs_code: str | None = None,
    ) -> NavMessage | None:
        """Select best ephemeris for given PRN and GPS time of week.

        Chooses the ephemeris with toe closest to the requested time.

        Args:
            prn: satellite PRN number
            gps_time: GPS seconds of week

        Returns:
            Best matching NavMessage, or None if PRN not found.
        """
        prn = _normalize_sat_id(prn)
        if prn not in self._nav_messages and isinstance(prn, int):
            prn = f"G{prn:02d}"
        if prn not in self._nav_messages:
            return None
        messages = self._nav_messages[prn]
        if not messages:
            return None

        best = None
        best_key = None
        for msg in messages:
            dt = abs(gps_time - msg.toe)
            # Handle week crossover
            if dt > GPS_WEEK_SEC / 2.0:
                dt = GPS_WEEK_SEC - dt
            priority = 0
            if obs_code is not None and msg.system == "E":
                priority = _galileo_priority(msg, obs_code)
            key = (dt, -priority)
            if best_key is None or key < best_key:
                best_key = key
                best = msg
        return best

    def _build_params(
        self, gps_time: float, prn_list: list[int | str] | None = None
    ) -> tuple[np.ndarray, list[int | str]]:
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
                if nav.system != "G":
                    raise ValueError("GPU ephemeris path only supports GPS")
                params_bytes += _nav_to_params_bytes(nav)
                used_prns.append(prn)

        if not used_prns:
            return np.array([], dtype=np.float64), []

        return np.frombuffer(params_bytes, dtype=np.float64), used_prns

    @staticmethod
    def _build_params_from_navs(navs: list[NavMessage]) -> np.ndarray:
        """Build a packed EphemerisParams array from already-selected messages."""
        if not navs:
            return np.array([], dtype=np.float64)
        return np.frombuffer(
            b"".join(_nav_to_params_bytes(nav) for nav in navs),
            dtype=np.float64,
        )

    def compute(
        self,
        gps_time: float,
        prn_list: list[int | str] | None = None,
        obs_codes: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[int | str]]:
        """Compute satellite positions at given GPS time of week.

        Args:
            gps_time: GPS seconds of week
            prn_list: list of PRNs to compute (None = all available)

        Returns:
            sat_ecef: array of shape [n_sat, 3] ECEF positions [m]
            sat_clk: array of shape [n_sat] clock corrections [s]
            used_prns: list of PRN numbers corresponding to output rows
        """
        if prn_list is None:
            prn_list = self._prn_list
        if obs_codes is not None and len(obs_codes) != len(prn_list):
            raise ValueError("obs_codes must match prn_list length")

        navs = [
            self.select_ephemeris(
                prn,
                gps_time,
                None if obs_codes is None else obs_codes[i],
            )
            for i, prn in enumerate(prn_list)
        ]
        use_gpu = HAS_GPU and all(nav is not None and nav.system == "G" for nav in navs if nav is not None)

        if not use_gpu:
            return self._compute_cpu(gps_time, prn_list, obs_codes)

        params_flat, used_prns = self._build_params(gps_time, prn_list)
        if not used_prns:
            return np.array([]).reshape(0, 3), np.array([]), []

        n_sat = len(used_prns)
        sat_ecef, sat_clk = compute_satellite_position(params_flat, gps_time, n_sat)
        return np.asarray(sat_ecef), np.asarray(sat_clk), used_prns

    def compute_batch(
        self, gps_times: np.ndarray | list[float], prn_list: list[int | str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int | str]]:
        """Batch computation for multiple epochs.

        Args:
            gps_times: array of GPS seconds of week [n_epoch]
            prn_list: list of PRNs to compute (None = all available)

        Returns:
            sat_ecef: array of shape [n_epoch, n_sat, 3] ECEF positions [m]
            sat_clk: array of shape [n_epoch, n_sat] clock corrections [s]
            used_prns: list of PRN numbers corresponding to output columns

        Notes:
            Ephemeris records are re-selected independently for each epoch.
            The returned PRN list is the subset with a valid ephemeris across
            every requested epoch so that the output shape stays rectangular.
        """
        gps_times = np.asarray(gps_times, dtype=np.float64).reshape(-1)

        if prn_list is None:
            prn_list = self._prn_list

        n_epoch = len(gps_times)
        if n_epoch == 0:
            return (
                np.array([]).reshape(0, 0, 3),
                np.array([]).reshape(0, 0),
                [],
            )

        selected_navs_per_epoch = []
        common_mask = np.ones(len(prn_list), dtype=bool)
        for gps_time in gps_times:
            navs = [self.select_ephemeris(prn, float(gps_time)) for prn in prn_list]
            selected_navs_per_epoch.append(navs)
            common_mask &= np.array([nav is not None for nav in navs], dtype=bool)

        used_prns = [prn for prn, keep in zip(prn_list, common_mask) if keep]
        if not used_prns:
            return (
                np.array([]).reshape(n_epoch, 0, 3),
                np.array([]).reshape(n_epoch, 0),
                [],
            )

        keep_indices = [i for i, keep in enumerate(common_mask) if keep]
        selected_navs_per_epoch = [
            [navs[i] for i in keep_indices] for navs in selected_navs_per_epoch
        ]

        n_sat = len(used_prns)
        use_gpu = HAS_GPU and all(
            nav.system == "G"
            for navs in selected_navs_per_epoch
            for nav in navs
        )

        if use_gpu:
            positions = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
            clocks = np.zeros((n_epoch, n_sat), dtype=np.float64)
            groups: dict[bytes, dict[str, list]] = {}

            for epoch_idx, navs in enumerate(selected_navs_per_epoch):
                params_blob = self._build_params_from_navs(navs).tobytes()
                group = groups.setdefault(params_blob, {"epochs": [], "times": []})
                group["epochs"].append(epoch_idx)
                group["times"].append(float(gps_times[epoch_idx]))

            for params_blob, group in groups.items():
                params_flat = np.frombuffer(params_blob, dtype=np.float64)
                group_times = np.asarray(group["times"], dtype=np.float64)
                group_pos, group_clk = compute_satellite_position_batch(
                    params_flat, group_times, n_sat
                )
                positions[group["epochs"]] = np.asarray(group_pos)
                clocks[group["epochs"]] = np.asarray(group_clk)

            return positions, clocks, used_prns

        positions = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
        clocks = np.zeros((n_epoch, n_sat), dtype=np.float64)
        for i, (gps_time, navs) in enumerate(zip(gps_times, selected_navs_per_epoch)):
            for j, nav in enumerate(navs):
                pos, clk = self._compute_single_cpu(nav, float(gps_time))
                positions[i, j] = pos
                clocks[i, j] = clk

        return positions, clocks, used_prns

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
    def _compute_single_cpu(
        nav: NavMessage,
        gps_time: float,
        obs_code: str | None = None,
        *,
        apply_group_delay: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Compute single satellite position on CPU."""
        mu, omega_e, rel_f = _constellation_constants(nav.system)
        a = nav.sqrt_a ** 2
        n0 = math.sqrt(mu / (a ** 3))
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

        Omega = nav.omega0 + (nav.omega_dot - omega_e) * tk - omega_e * nav.toe

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
        dtr = rel_f * nav.e * nav.sqrt_a * sinE
        group_delay = _group_delay(nav, obs_code) if apply_group_delay else 0.0
        clk = nav.af0 + nav.af1 * dt + nav.af2 * dt * dt + dtr - group_delay

        return pos, clk

    def _compute_cpu(
        self,
        gps_time: float,
        prn_list: list[int | str] | None = None,
        obs_codes: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[int | str]]:
        """CPU fallback for single-epoch computation."""
        if prn_list is None:
            prn_list = self._prn_list
        if obs_codes is not None and len(obs_codes) != len(prn_list):
            raise ValueError("obs_codes must match prn_list length")

        positions = []
        clocks = []
        used_prns = []

        for i, prn in enumerate(prn_list):
            obs_code = None if obs_codes is None else obs_codes[i]
            nav = self.select_ephemeris(prn, gps_time, obs_code)
            if nav is None:
                continue
            pos, clk = self._compute_single_cpu(nav, gps_time, obs_code)
            positions.append(pos)
            clocks.append(clk)
            used_prns.append(prn)

        if not used_prns:
            return np.array([]).reshape(0, 3), np.array([]), []

        return np.array(positions), np.array(clocks), used_prns

    def _compute_batch_cpu(
        self, gps_times: np.ndarray, prn_list: list[int | str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[int | str]]:
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
