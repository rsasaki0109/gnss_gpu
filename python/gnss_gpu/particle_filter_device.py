"""GPU-accelerated Particle Filter with persistent device memory.

Particle state lives on GPU. No H2D/D2H transfers except:
- Satellite data per update (small: ~1KB for 8 satellites)
- Estimate output (32 bytes)
- Velocity per predict (24 bytes)
- Particle dump for visualization (on-demand)

This eliminates the #1 performance bottleneck: cudaMalloc/cudaFree and
full particle array H2D/D2H transfers on every call.
"""

from types import SimpleNamespace

import numpy as np


class ParticleFilterDevice:
    """High-performance particle filter with persistent GPU memory.

    Parameters
    ----------
    n_particles : int
        Number of particles (e.g., 1_000_000).
    sigma_pos : float
        Position noise standard deviation [m] for prediction.
    sigma_cb : float
        Clock bias noise standard deviation [m] for prediction.
    sigma_pr : float
        Pseudorange observation standard deviation [m] for weighting.
    resampling : str
        Resampling method: "megopolis" or "systematic".
    ess_threshold : float
        ESS ratio threshold for triggering resampling (0-1).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_particles=1_000_000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, nu=0.0, resampling="megopolis", ess_threshold=0.5,
                 seed=42):
        from gnss_gpu._gnss_gpu_pf_device import (
            pf_device_create,
            pf_device_destroy,
            pf_device_initialize,
            pf_device_predict,
            pf_device_weight,
            pf_device_weight_dd_pseudorange,
            pf_device_weight_gmm,
            pf_device_weight_carrier_afv,
            pf_device_weight_dd_carrier_afv,
            pf_device_position_update,
            pf_device_shift_clock_bias,
            pf_device_ess,
            pf_device_position_spread,
            pf_device_resample_systematic,
            pf_device_resample_megopolis,
            pf_device_estimate,
            pf_device_get_particles,
            pf_device_get_log_weights,
            pf_device_get_resample_ancestors,
            pf_device_sync,
        )
        self._pf_device_create = pf_device_create
        self._pf_device_destroy = pf_device_destroy
        self._pf_device_initialize = pf_device_initialize
        self._pf_device_predict = pf_device_predict
        self._pf_device_weight = pf_device_weight
        self._pf_device_weight_dd_pseudorange = pf_device_weight_dd_pseudorange
        self._pf_device_weight_gmm = pf_device_weight_gmm
        self._pf_device_weight_carrier_afv = pf_device_weight_carrier_afv
        self._pf_device_weight_dd_carrier_afv = pf_device_weight_dd_carrier_afv
        self._pf_device_position_update = pf_device_position_update
        self._pf_device_shift_clock_bias = pf_device_shift_clock_bias
        self._pf_device_ess = pf_device_ess
        self._pf_device_position_spread = pf_device_position_spread
        self._pf_device_resample_systematic = pf_device_resample_systematic
        self._pf_device_resample_megopolis = pf_device_resample_megopolis
        self._pf_device_estimate = pf_device_estimate
        self._pf_device_get_particles = pf_device_get_particles
        self._pf_device_get_log_weights = pf_device_get_log_weights
        self._pf_device_get_resample_ancestors = pf_device_get_resample_ancestors
        self._pf_device_sync = pf_device_sync

        self.n_particles = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.nu = nu  # Student's t DoF. 0=Gaussian, 1=Cauchy, 3-5=moderate
        self.resampling = resampling
        self.ess_threshold = ess_threshold
        self.seed = seed

        # Allocate GPU memory once
        self._state = self._pf_device_create(n_particles)
        self._initialized = False
        self._step = 0

    def __del__(self):
        # GPU resources are freed automatically by the pybind11 unique_ptr
        # custom deleter when the Python object is garbage collected.
        # Explicitly calling pf_device_destroy here would cause a double-free
        # since pybind11 already manages the lifetime.
        pass

    def initialize(self, position_ecef, clock_bias=0.0, spread_pos=100.0,
                   spread_cb=1000.0):
        """Scatter particles around an initial estimate.

        Runs entirely on GPU - no host-device transfer of particle data.

        Parameters
        ----------
        position_ecef : array_like, shape (3,)
            Initial ECEF position [m].
        clock_bias : float
            Initial receiver clock bias [m].
        spread_pos : float
            Standard deviation for initial position scatter [m].
        spread_cb : float
            Standard deviation for initial clock bias scatter [m].
        """
        pos = np.asarray(position_ecef, dtype=np.float64).ravel()

        self._pf_device_initialize(
            self._state,
            float(pos[0]), float(pos[1]), float(pos[2]), float(clock_bias),
            float(spread_pos), float(spread_cb),
            self.seed)

        self._initialized = True
        self._step = 0

    def predict(self, velocity=None, dt=1.0, sigma_pos=None):
        """Predict step with optional velocity.

        Only 24 bytes (velocity) transferred to GPU.

        Parameters
        ----------
        velocity : array_like, shape (3,), optional
            Velocity in ECEF [m/s]. Defaults to zero (stationary).
        dt : float
            Time step [s].
        sigma_pos : float, optional
            Per-step position random-walk sigma [m]. Defaults to ``self.sigma_pos``.
            Use a smaller value when a high-quality velocity guide (e.g. TDCP) is available.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        if velocity is None:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            vel = np.asarray(velocity, dtype=np.float64).ravel()
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

        sp = float(self.sigma_pos if sigma_pos is None else sigma_pos)

        self._step += 1
        self._pf_device_predict(
            self._state,
            vx, vy, vz,
            float(dt), sp, float(self.sigma_cb),
            self.seed, self._step)

    def update(self, sat_ecef, pseudoranges, weights=None, sigma_pr=None, resample=True):
        """Weight update with pseudorange observations.

        Only satellite data (~1KB) transferred to GPU.
        Particle arrays stay on device.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Per-satellite weights (1/sigma^2). Defaults to ones.
        sigma_pr : float, optional
            Per-call pseudorange sigma [m]. Defaults to ``self.sigma_pr``.
        resample : bool
            If True (default), run ESS-based adaptive resampling after weighting.
            Set False to snapshot log-weights and particles before resampling (FFBSi).
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()
        sp = float(self.sigma_pr if sigma_pr is None else sigma_pr)

        self._pf_device_weight(
            self._state,
            sat.ravel(), pr, weights,
            n_sat, sp, float(self.nu))

        if resample:
            _ = self.resample_if_needed()

    def update_gmm(self, sat_ecef, pseudoranges, weights=None, sigma_pr=None,
                   w_los=0.7, mu_nlos=15.0, sigma_nlos=30.0, resample=True):
        """Update weights using GMM likelihood (LOS + NLOS components).

        Models pseudorange errors as a mixture of a narrow LOS Gaussian
        and a wide NLOS Gaussian with positive bias. More robust than
        Student's t for NLOS multipath in urban environments.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Per-satellite weights (1/sigma^2). Defaults to ones.
        sigma_pr : float, optional
            LOS component sigma [m]. Defaults to ``self.sigma_pr``.
        w_los : float
            Weight of LOS component (0-1). Default 0.7.
        mu_nlos : float
            Mean of NLOS component [m]. Default 15.0.
        sigma_nlos : float
            Std of NLOS component [m]. Default 30.0.
        resample : bool
            If True (default), run ESS-based adaptive resampling after weighting.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        sp = float(self.sigma_pr if sigma_pr is None else sigma_pr)

        self._pf_device_weight_gmm(
            self._state,
            sat.ravel(), pr, weights,
            n_sat, sp, float(w_los), float(mu_nlos), float(sigma_nlos))

        if resample:
            _ = self.resample_if_needed()

    def update_dd_pseudorange(self, dd_result, sigma_pr=0.75, resample=True):
        """Update weights using DD pseudorange likelihood.

        Double-differenced pseudorange eliminates receiver clock bias from both
        rover and base, so weighting depends only on the 3D particle position.

        Parameters
        ----------
        dd_result : DDPseudorangeResult
            Output from :meth:`DDPseudorangeComputer.compute_dd`.
        sigma_pr : float
            Standard deviation of the DD pseudorange residual [m].
        resample : bool
            If True (default), run ESS-based adaptive resampling after weighting.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        self._pf_device_weight_dd_pseudorange(
            self._state,
            dd_result.sat_ecef_k.ravel(),
            dd_result.sat_ecef_ref.ravel(),
            dd_result.dd_pseudorange_m,
            dd_result.base_range_k,
            dd_result.base_range_ref,
            dd_result.dd_weights,
            dd_result.n_dd,
            float(sigma_pr),
        )

        if resample:
            _ = self.resample_if_needed()

    def update_carrier_afv(self, sat_ecef, carrier_phase_cycles, weights=None,
                           wavelength=0.190293673, sigma_cycles=0.05, resample=True):
        """Update weights using carrier phase AFV likelihood (no ambiguity needed).

        Uses the Ambiguity Function Value: fractional cycle residuals form a
        sharp likelihood (sigma ~ 0.05 cycles ~ 1 cm) without resolving integer
        ambiguities. Call AFTER update() (pseudorange) for the MUPF algorithm
        (Suzuki 2024).

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        carrier_phase_cycles : array_like, shape (n_sat,)
            Observed carrier phase measurements [cycles].
        weights : array_like, shape (n_sat,), optional
            Per-satellite elevation weights. Defaults to ones.
        wavelength : float
            Carrier wavelength [m]. Default is L1 GPS (0.190293673 m).
        sigma_cycles : float
            Standard deviation of AFV residual [cycles]. Default 0.05.
        resample : bool
            If True (default), run ESS-based adaptive resampling after weighting.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        cp = np.asarray(carrier_phase_cycles, dtype=np.float64).ravel()
        n_sat = len(cp)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        self._pf_device_weight_carrier_afv(
            self._state,
            sat.ravel(), cp, weights,
            n_sat, float(wavelength), float(sigma_cycles))

        if resample:
            _ = self.resample_if_needed()

    def update_dd_carrier_afv(self, dd_result, wavelength=None,
                               sigma_cycles=0.05, resample=True):
        """Update weights using DD carrier phase AFV likelihood.

        Double-differenced carrier phase eliminates receiver clock bias
        from both rover and base, and largely cancels atmospheric errors.
        This makes the AFV likelihood much more effective than undifferenced.

        Parameters
        ----------
        dd_result : DDResult
            Output from :meth:`DDCarrierComputer.compute_dd`.
        wavelength : float or None
            Optional wavelength override [m]. If omitted, use per-DD-pair
            wavelengths from ``dd_result`` when available.
        sigma_cycles : float
            Standard deviation of DD AFV residual [cycles]. Default 0.05.
        resample : bool
            If True (default), run ESS-based adaptive resampling after weighting.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        if wavelength is None:
            wavelengths = getattr(dd_result, "wavelengths_m", None)
            if wavelengths is None:
                wavelengths = np.full(dd_result.n_dd, 0.190293673, dtype=np.float64)
            else:
                wavelengths = np.asarray(wavelengths, dtype=np.float64).ravel()
        else:
            wavelengths = np.full(dd_result.n_dd, float(wavelength), dtype=np.float64)

        self._pf_device_weight_dd_carrier_afv(
            self._state,
            dd_result.sat_ecef_k.ravel(),
            dd_result.sat_ecef_ref.ravel(),
            dd_result.dd_carrier_cycles,
            dd_result.base_range_k,
            dd_result.base_range_ref,
            dd_result.dd_weights,
            wavelengths,
            dd_result.n_dd,
            float(sigma_cycles))

        if resample:
            _ = self.resample_if_needed()

    def position_update(self, ref_ecef, sigma_pos=30.0):
        """Apply position-domain soft constraint from external estimate.

        Adds a Gaussian log-likelihood penalty based on distance from
        a reference position (e.g., SPP solution). Particles far from
        the reference get lower weight, pulling the cloud center toward it.

        Parameters
        ----------
        ref_ecef : array_like, shape (3,)
            Reference ECEF position [m] (e.g., from SPP).
        sigma_pos : float
            Standard deviation of the soft constraint [m].
            Smaller = stronger pull toward reference.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        ref = np.asarray(ref_ecef, dtype=np.float64).ravel()
        self._pf_device_position_update(
            self._state,
            float(ref[0]), float(ref[1]), float(ref[2]),
            float(sigma_pos))

    def shift_clock_bias(self, shift: float):
        """Shift all particles' clock bias by a constant offset.

        Used to re-center cb around an external estimate each epoch,
        compensating for systematic receiver clock drift that the
        random-walk model cannot track.

        Parameters
        ----------
        shift : float
            Clock bias shift [m]. Positive shifts increase all cb values.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")
        self._pf_device_shift_clock_bias(self._state, float(shift))

    def correct_clock_bias(self, sat_ecef, pseudoranges):
        """Re-center particles' clock bias using pseudorange residuals.

        Computes the expected cb from the current position estimate and
        observed pseudoranges, then shifts all particles' cb to match.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        est = np.asarray(self.estimate(), dtype=np.float64)
        pos = est[:3]
        current_cb = est[3]

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()

        ranges = np.linalg.norm(sat - pos, axis=1)
        residuals = pr - ranges
        expected_cb = float(np.median(residuals))

        shift = expected_cb - current_cb
        self._pf_device_shift_clock_bias(self._state, shift)

    def _resample(self):
        """Perform resampling entirely on GPU."""
        if self.resampling == "megopolis":
            self._pf_device_resample_megopolis(self._state, 15, self.seed)
        else:
            self._pf_device_resample_systematic(self._state, self.seed)

    def resample_if_needed(self):
        """Resample when ESS falls below ``ess_threshold * n_particles``.

        Returns
        -------
        did_resample : bool
            True if resampling ran. For genealogy smoothers with systematic
            resampling, call ``get_resample_ancestors()`` only when True.
        """
        ess = self.get_ess()
        if ess < self.ess_threshold * self.n_particles:
            self._resample()
            return True
        return False

    def estimate(self):
        """Compute weighted mean position.

        Only 32 bytes (result) transferred from GPU.

        Returns
        -------
        result : ndarray, shape (4,)
            Estimated [x, y, z, clock_bias] in ECEF [m].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_estimate(self._state)

    def get_particles(self):
        """Get all particle states from GPU.

        WARNING: This is the only operation that transfers all particle data
        from GPU to host. Use sparingly (e.g., for visualization).

        Returns
        -------
        particles : ndarray, shape (n_particles, 4)
            Each row is [x, y, z, clock_bias].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_get_particles(self._state)

    def get_log_weights(self):
        """Copy per-particle log-weights from GPU (synchronizes stream).

        Values reflect the current step after ``update`` / ``position_update``.
        Intended for FFBSi and diagnostics; prefer ``get_ess`` / ``estimate`` for metrics.
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        out = np.empty(self.n_particles, dtype=np.float64)
        self._pf_device_get_log_weights(self._state, out)
        return out

    def get_resample_ancestors(self):
        """Ancestor indices from the **last** systematic resample: ``out[j]=i`` means slot ``j``
        copied state from slot ``i``. Only valid after a systematic resample (not Megopolis).
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")
        if self.resampling != "systematic":
            raise RuntimeError("get_resample_ancestors requires resampling='systematic'")

        out = np.empty(self.n_particles, dtype=np.int32)
        self._pf_device_get_resample_ancestors(self._state, out)
        return out

    def get_ess(self):
        """Compute current Effective Sample Size.

        Computed on GPU, only scalar result transferred.

        Returns
        -------
        ess : float
            Effective Sample Size in [1, n_particles].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_ess(self._state)

    def get_position_spread(self, center=None):
        """Compute weighted RMS particle spread around a reference point.

        Parameters
        ----------
        center : array_like, shape (3,), optional
            Reference ECEF position [m]. If omitted, uses the current weighted
            mean estimate.

        Returns
        -------
        spread_m : float
            Weighted RMS particle radius [m].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        if center is None:
            center_arr = np.asarray(self.estimate()[:3], dtype=np.float64)
        else:
            center_arr = np.asarray(center, dtype=np.float64).ravel()
            if center_arr.size != 3:
                raise ValueError("center must have shape (3,)")
        return self._pf_device_position_spread(
            self._state,
            float(center_arr[0]), float(center_arr[1]), float(center_arr[2]),
        )

    def sync(self):
        """Explicitly synchronize the CUDA stream.

        Waits for all pending GPU operations (async transfers and kernel
        launches) to complete. This is called automatically before any
        D2H transfer (estimate, get_particles, get_ess), but can be
        useful when benchmarking or coordinating with other GPU work.
        """
        if hasattr(self, '_state') and self._state is not None:
            self._pf_device_sync(self._state)

    # ----------------------------------------------------------------
    # Smoother support (forward-backward)
    # ----------------------------------------------------------------

    def enable_smoothing(self):
        """Enable epoch storage for offline forward-backward smoothing.

        Call this before the forward pass. After all epochs are processed,
        call ``smooth()`` to run a backward pass and return smoothed estimates.
        """
        self._smooth_epochs = []
        self._smooth_enabled = True

    def store_epoch(
        self,
        sat_ecef,
        pseudoranges,
        weights,
        velocity,
        dt,
        spp_ref=None,
        dd_pseudorange=None,
        dd_pseudorange_sigma=None,
        dd_carrier=None,
        dd_carrier_sigma=None,
        carrier_anchor_pseudorange=None,
        carrier_anchor_sigma=None,
        carrier_afv=None,
        carrier_afv_sigma=None,
        carrier_afv_wavelength=None,
    ):
        """Store observation data for the current epoch (call after update/estimate).

        Parameters
        ----------
        sat_ecef, pseudoranges, weights : array_like
            Same arrays passed to ``update()``.
        velocity : array_like or None
            Velocity used in ``predict()``.
        dt : float
            Time step.
        spp_ref : array_like or None
            SPP reference position for position_update (None to skip).
        dd_pseudorange : object or None
            DD pseudorange result used in the forward pass. When present, the
            backward pass replays the same DD update instead of undifferenced PR.
        dd_pseudorange_sigma : float or None
            Sigma used for the forward DD pseudorange update.
        dd_carrier : object or None
            DD carrier AFV result used in the forward pass. When present, the
            backward pass replays the same DD carrier update after DD PR / PR.
        dd_carrier_sigma : float or None
            Sigma used for the forward DD carrier AFV update.
        carrier_anchor_pseudorange : dict or None
            Carrier-bias-conditioned pseudorange-like update used in the
            forward pass. When present, the backward pass replays it after
            DD carrier / carrier AFV updates.
        carrier_anchor_sigma : float or None
            Sigma used for the forward carrier-anchor pseudorange update.
        carrier_afv : dict or None
            Undifferenced carrier AFV observation used in the forward pass.
            When present, the backward pass replays the same carrier AFV update.
        carrier_afv_sigma : float or None
            Sigma used for the forward undifferenced carrier AFV update.
        carrier_afv_wavelength : float or None
            Carrier wavelength used for the undifferenced AFV update.
        """
        if not getattr(self, '_smooth_enabled', False):
            return
        est = np.asarray(self.estimate(), dtype=np.float64)
        dd_pr_store = None
        if dd_pseudorange is not None:
            dd_pr_store = {
                'dd_pseudorange_m': np.asarray(dd_pseudorange.dd_pseudorange_m, dtype=np.float64).copy(),
                'sat_ecef_k': np.asarray(dd_pseudorange.sat_ecef_k, dtype=np.float64).copy(),
                'sat_ecef_ref': np.asarray(dd_pseudorange.sat_ecef_ref, dtype=np.float64).copy(),
                'base_range_k': np.asarray(dd_pseudorange.base_range_k, dtype=np.float64).copy(),
                'base_range_ref': np.asarray(dd_pseudorange.base_range_ref, dtype=np.float64).copy(),
                'dd_weights': np.asarray(dd_pseudorange.dd_weights, dtype=np.float64).copy(),
                'ref_sat_ids': tuple(getattr(dd_pseudorange, 'ref_sat_ids', ())),
                'n_dd': int(dd_pseudorange.n_dd),
            }
        dd_cp_store = None
        if dd_carrier is not None:
            dd_cp_store = {
                'dd_carrier_cycles': np.asarray(dd_carrier.dd_carrier_cycles, dtype=np.float64).copy(),
                'sat_ecef_k': np.asarray(dd_carrier.sat_ecef_k, dtype=np.float64).copy(),
                'sat_ecef_ref': np.asarray(dd_carrier.sat_ecef_ref, dtype=np.float64).copy(),
                'base_range_k': np.asarray(dd_carrier.base_range_k, dtype=np.float64).copy(),
                'base_range_ref': np.asarray(dd_carrier.base_range_ref, dtype=np.float64).copy(),
                'dd_weights': np.asarray(dd_carrier.dd_weights, dtype=np.float64).copy(),
                'wavelengths_m': np.asarray(dd_carrier.wavelengths_m, dtype=np.float64).copy(),
                'ref_sat_ids': tuple(getattr(dd_carrier, 'ref_sat_ids', ())),
                'n_dd': int(dd_carrier.n_dd),
            }
        carrier_anchor_store = None
        if carrier_anchor_pseudorange is not None:
            carrier_anchor_store = {
                'sat_ecef': np.asarray(
                    carrier_anchor_pseudorange['sat_ecef'], dtype=np.float64
                ).copy(),
                'pseudoranges': np.asarray(
                    carrier_anchor_pseudorange['pseudoranges'], dtype=np.float64
                ).copy(),
                'weights': np.asarray(
                    carrier_anchor_pseudorange['weights'], dtype=np.float64
                ).copy(),
                'n_sat': int(len(np.asarray(carrier_anchor_pseudorange['pseudoranges']).ravel())),
            }
        carrier_afv_store = None
        if carrier_afv is not None:
            carrier_afv_store = {
                'sat_ecef': np.asarray(carrier_afv['sat_ecef'], dtype=np.float64).copy(),
                'carrier_phase_cycles': np.asarray(
                    carrier_afv['carrier_phase_cycles'], dtype=np.float64
                ).copy(),
                'weights': np.asarray(carrier_afv['weights'], dtype=np.float64).copy(),
                'n_sat': int(len(np.asarray(carrier_afv['carrier_phase_cycles']).ravel())),
            }
        self._smooth_epochs.append({
            'estimate': est[:3].copy(),
            'sat_ecef': np.asarray(sat_ecef, dtype=np.float64).copy(),
            'pseudoranges': np.asarray(pseudoranges, dtype=np.float64).copy(),
            'weights': np.asarray(weights, dtype=np.float64).copy(),
            'velocity': np.asarray(velocity, dtype=np.float64).copy() if velocity is not None else None,
            'dt': float(dt),
            'spp_ref': np.asarray(spp_ref, dtype=np.float64).copy() if spp_ref is not None else None,
            'dd_pseudorange': dd_pr_store,
            'dd_pseudorange_sigma': (
                None if dd_pseudorange_sigma is None else float(dd_pseudorange_sigma)
            ),
            'dd_carrier': dd_cp_store,
            'dd_carrier_sigma': (
                None if dd_carrier_sigma is None else float(dd_carrier_sigma)
            ),
            'carrier_anchor_pseudorange': carrier_anchor_store,
            'carrier_anchor_sigma': (
                None if carrier_anchor_sigma is None else float(carrier_anchor_sigma)
            ),
            'carrier_afv': carrier_afv_store,
            'carrier_afv_sigma': (
                None if carrier_afv_sigma is None else float(carrier_afv_sigma)
            ),
            'carrier_afv_wavelength': (
                None if carrier_afv_wavelength is None else float(carrier_afv_wavelength)
            ),
        })

    def smooth(self, position_update_sigma=None):
        """Run backward pass and return smoothed (forward+backward averaged) estimates.

        Must be called after a complete forward pass with ``enable_smoothing()``
        and ``store_epoch()`` on every epoch.

        Parameters
        ----------
        position_update_sigma : float or None
            Sigma for SPP position-domain update in backward pass.
            If None, uses same as forward.

        Returns
        -------
        smoothed : ndarray, shape (N_epochs, 3)
            Smoothed ECEF positions.
        forward : ndarray, shape (N_epochs, 3)
            Forward-only estimates (for comparison).
        """
        if not getattr(self, '_smooth_enabled', False) or not self._smooth_epochs:
            raise RuntimeError("No stored epochs. Call enable_smoothing() before forward pass.")

        stored = self._smooth_epochs
        n_ep = len(stored)
        forward_pos = np.array([e['estimate'] for e in stored])

        # Backward pass: new PF instance, reversed epoch order
        last = stored[-1]
        init_pos = last['estimate']
        init_cb_candidates = last['pseudoranges'] - np.linalg.norm(
            last['sat_ecef'].reshape(-1, 3) - init_pos, axis=1)
        init_cb = float(np.median(init_cb_candidates))

        bwd_pf = ParticleFilterDevice(
            n_particles=self.n_particles,
            sigma_pos=self.sigma_pos,
            sigma_cb=self.sigma_cb,
            sigma_pr=self.sigma_pr,
            nu=self.nu,
            resampling=self.resampling,
            ess_threshold=self.ess_threshold,
            seed=self.seed + 1,  # different seed for diversity
        )
        bwd_pf.initialize(init_pos, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

        backward_pos = np.zeros((n_ep, 3))
        for i in range(n_ep - 1, -1, -1):
            ep = stored[i]
            vel = -ep['velocity'] if ep['velocity'] is not None else None
            bwd_pf.predict(velocity=vel, dt=ep['dt'])

            sat = ep['sat_ecef'].reshape(-1, 3)
            pr = ep['pseudoranges']
            w = ep['weights']
            dd_ep = ep.get('dd_pseudorange')
            dd_cp_ep = ep.get('dd_carrier')
            carrier_anchor_ep = ep.get('carrier_anchor_pseudorange')
            carrier_afv_ep = ep.get('carrier_afv')
            if dd_ep is not None:
                bwd_pf.update_dd_pseudorange(
                    SimpleNamespace(**dd_ep),
                    sigma_pr=(
                        float(ep['dd_pseudorange_sigma'])
                        if ep.get('dd_pseudorange_sigma') is not None
                        else self.sigma_pr
                    ),
                )
            else:
                bwd_pf.correct_clock_bias(sat, pr)
                bwd_pf.update(sat, pr, weights=w)

            if dd_cp_ep is not None:
                bwd_pf.resample_if_needed()
                bwd_pf.update_dd_carrier_afv(
                    SimpleNamespace(**dd_cp_ep),
                    sigma_cycles=(
                        float(ep['dd_carrier_sigma'])
                        if ep.get('dd_carrier_sigma') is not None
                        else 0.05
                    ),
                )
            if carrier_anchor_ep is not None:
                bwd_pf.update(
                    carrier_anchor_ep['sat_ecef'],
                    carrier_anchor_ep['pseudoranges'],
                    weights=carrier_anchor_ep['weights'],
                    sigma_pr=(
                        float(ep['carrier_anchor_sigma'])
                        if ep.get('carrier_anchor_sigma') is not None
                        else self.sigma_pr
                    ),
                )
            if carrier_afv_ep is not None:
                bwd_pf.resample_if_needed()
                bwd_pf.update_carrier_afv(
                    carrier_afv_ep['sat_ecef'],
                    carrier_afv_ep['carrier_phase_cycles'],
                    weights=carrier_afv_ep['weights'],
                    wavelength=(
                        float(ep['carrier_afv_wavelength'])
                        if ep.get('carrier_afv_wavelength') is not None
                        else 0.190293673
                    ),
                    sigma_cycles=(
                        float(ep['carrier_afv_sigma'])
                        if ep.get('carrier_afv_sigma') is not None
                        else 0.05
                    ),
                )

            pu_sigma = position_update_sigma if position_update_sigma is not None else None
            if pu_sigma is not None and ep['spp_ref'] is not None:
                bwd_pf.position_update(ep['spp_ref'][:3], sigma_pos=pu_sigma)

            backward_pos[i] = bwd_pf.estimate()[:3]

        # Combine: simple average (equal weight)
        smoothed = (forward_pos + backward_pos) / 2.0

        self._smooth_enabled = False
        self._smooth_epochs = []

        return smoothed, forward_pos
