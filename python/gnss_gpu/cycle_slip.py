"""Cycle slip detection for carrier phase GNSS observations.

Pure Python (numpy) implementations of standard cycle slip detectors.
Each function operates on arrays shaped (n_epoch, n_sat) and returns
a boolean mask of the same shape where True indicates a detected slip.
"""

import numpy as np

# GPS L1 and L2 wavelengths [m]
L1_WAVELENGTH = 0.19029  # ~1575.42 MHz
L2_WAVELENGTH = 0.24421  # ~1227.60 MHz

# Wide-lane and narrow-lane wavelengths [m]
_F1 = 1575.42e6  # L1 frequency [Hz]
_F2 = 1227.60e6  # L2 frequency [Hz]
WIDELANE_WAVELENGTH = 299792458.0 / (_F1 - _F2)  # ~0.862 m


def detect_geometry_free(carrier_L1, carrier_L2, threshold=0.05):
    """Geometry-free (L4) combination cycle slip detector.

    The geometry-free combination L4 = L1 - L2 (in metres) removes the
    geometric range and isolates the ionospheric delay plus ambiguities.
    A jump in L4 between consecutive epochs indicates a cycle slip on at
    least one frequency.

    Parameters
    ----------
    carrier_L1 : array_like, shape (n_epoch, n_sat)
        L1 carrier phase observations [cycles].
    carrier_L2 : array_like, shape (n_epoch, n_sat)
        L2 carrier phase observations [cycles].
    threshold : float
        Detection threshold on |delta L4| [metres].  Default 0.05 m.

    Returns
    -------
    slip_mask : ndarray, shape (n_epoch, n_sat), dtype bool
        True where a cycle slip is detected.  First epoch is always False.
    """
    L1 = np.asarray(carrier_L1, dtype=np.float64)
    L2 = np.asarray(carrier_L2, dtype=np.float64)

    # Convert to metres for the combination
    L4 = L1 * L1_WAVELENGTH - L2 * L2_WAVELENGTH

    # Epoch-to-epoch difference
    dL4 = np.diff(L4, axis=0)

    slip = np.abs(dL4) > threshold

    # Prepend False row for the first epoch (no prior to compare against)
    first = np.zeros((1, L1.shape[1]), dtype=bool)
    return np.vstack([first, slip])


def detect_melbourne_wubbena(carrier_L1, carrier_L2, pr_L1, pr_L2,
                             threshold=1.0):
    """Melbourne-Wubbena (MW) combination cycle slip detector.

    The MW combination forms the wide-lane carrier minus narrow-lane
    pseudorange, which is geometry-free and ionosphere-free.  Its expected
    value is the wide-lane ambiguity (constant in the absence of slips).
    A jump between consecutive epochs flags a cycle slip.

    Parameters
    ----------
    carrier_L1 : array_like, shape (n_epoch, n_sat)
        L1 carrier phase [cycles].
    carrier_L2 : array_like, shape (n_epoch, n_sat)
        L2 carrier phase [cycles].
    pr_L1 : array_like, shape (n_epoch, n_sat)
        L1 pseudorange [metres].
    pr_L2 : array_like, shape (n_epoch, n_sat)
        L2 pseudorange [metres].
    threshold : float
        Detection threshold on |delta MW| [wide-lane cycles].  Default 1.0.

    Returns
    -------
    slip_mask : ndarray, shape (n_epoch, n_sat), dtype bool
    """
    phi1 = np.asarray(carrier_L1, dtype=np.float64)
    phi2 = np.asarray(carrier_L2, dtype=np.float64)
    P1 = np.asarray(pr_L1, dtype=np.float64)
    P2 = np.asarray(pr_L2, dtype=np.float64)

    # Wide-lane carrier phase [cycles]
    #   phi_W = (f1*phi1 - f2*phi2) / (f1 - f2)
    # which simplifies to phi1 - phi2 when phases are in cycles and we
    # express the result in wide-lane cycles.
    phi_W = phi1 - phi2  # wide-lane carrier [widelane cycles]

    # Narrow-lane pseudorange [metres] -> convert to wide-lane cycles
    #   P_N = (f1*P1 + f2*P2) / (f1 + f2)   [metres]
    P_N = (_F1 * P1 + _F2 * P2) / (_F1 + _F2)
    P_N_wl = P_N / WIDELANE_WAVELENGTH  # convert to wide-lane cycles

    MW = phi_W - P_N_wl  # should be ~constant (= widelane ambiguity)

    dMW = np.diff(MW, axis=0)

    slip = np.abs(dMW) > threshold

    first = np.zeros((1, phi1.shape[1]), dtype=bool)
    return np.vstack([first, slip])


def detect_time_difference(carrier, threshold=0.5):
    """Simple time-difference cycle slip detector.

    Flags any epoch where the carrier phase changes by more than
    *threshold* cycles from the previous epoch.  Works on a single
    frequency.

    This is the simplest possible detector and is sensitive to receiver
    dynamics (high velocity can trigger false alarms unless threshold is
    chosen carefully).

    Parameters
    ----------
    carrier : array_like, shape (n_epoch, n_sat)
        Carrier phase observations [cycles].
    threshold : float
        Detection threshold [cycles].  Default 0.5 cycles.

    Returns
    -------
    slip_mask : ndarray, shape (n_epoch, n_sat), dtype bool
    """
    phi = np.asarray(carrier, dtype=np.float64)

    dphi = np.diff(phi, axis=0)

    slip = np.abs(dphi) > threshold

    first = np.zeros((1, phi.shape[1]), dtype=bool)
    return np.vstack([first, slip])
