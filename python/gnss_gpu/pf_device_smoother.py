"""Forward-backward smoother helpers for ParticleFilterDevice."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gnss_gpu.pf_device_config import clone_pf_device_init_kwargs


class ParticleFilterDeviceSmootherMixin:
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
        dd_pseudorange_source=None,
        dd_carrier=None,
        dd_carrier_sigma=None,
        carrier_anchor_pseudorange=None,
        carrier_anchor_sigma=None,
        carrier_afv=None,
        carrier_afv_sigma=None,
        carrier_afv_wavelength=None,
        doppler_update=None,
        doppler_sigma_mps=None,
        doppler_velocity_update_gain=None,
        doppler_max_velocity_update_mps=None,
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
        dd_pseudorange_source : str or None
            Optional source label for the DD pseudorange update.
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
        doppler_update : dict or None
            Per-particle Doppler velocity update used in the forward pass.
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
        doppler_store = None
        if doppler_update is not None:
            doppler_store = {
                'sat_ecef': np.asarray(doppler_update['sat_ecef'], dtype=np.float64).copy(),
                'sat_vel': np.asarray(doppler_update['sat_vel'], dtype=np.float64).copy(),
                'doppler_hz': np.asarray(doppler_update['doppler_hz'], dtype=np.float64).copy(),
                'weights': np.asarray(doppler_update['weights'], dtype=np.float64).copy(),
                'wavelength_m': float(doppler_update.get('wavelength_m', 0.19029367279836488)),
                'n_sat': int(len(np.asarray(doppler_update['doppler_hz']).ravel())),
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
            'dd_pseudorange_source': (
                None if dd_pseudorange_source is None else str(dd_pseudorange_source)
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
            'doppler_update': doppler_store,
            'doppler_sigma_mps': (
                None if doppler_sigma_mps is None else float(doppler_sigma_mps)
            ),
            'doppler_velocity_update_gain': (
                None
                if doppler_velocity_update_gain is None
                else float(doppler_velocity_update_gain)
            ),
            'doppler_max_velocity_update_mps': (
                None
                if doppler_max_velocity_update_mps is None
                else float(doppler_max_velocity_update_mps)
            ),
        })

    def smooth(self, position_update_sigma=None, skip_widelane_dd_pseudorange=False):
        """Run backward pass and return smoothed (forward+backward averaged) estimates.

        Must be called after a complete forward pass with ``enable_smoothing()``
        and ``store_epoch()`` on every epoch.

        Parameters
        ----------
        position_update_sigma : float or None
            Sigma for SPP position-domain update in backward pass.
            If None, uses same as forward.
        skip_widelane_dd_pseudorange : bool
            If True, do not replay DD pseudorange updates tagged as wide-lane in
            the backward pass; replay undifferenced pseudorange instead.

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

        # Resolve the public class lazily.  Besides avoiding an import cycle
        # with ``particle_filter_device``, this keeps the smoother usable with
        # test doubles and subclasses whose ``self`` is only a state shell.
        from gnss_gpu import particle_filter_device as pfd

        bwd_pf = pfd.ParticleFilterDevice(**clone_pf_device_init_kwargs(self))
        bwd_pf.initialize(
            init_pos,
            clock_bias=init_cb,
            spread_pos=10.0,
            spread_cb=100.0,
            velocity_init_sigma=float(getattr(self, "_velocity_init_sigma", 0.0)),
        )

        backward_pos = np.zeros((n_ep, 3))
        for i in range(n_ep - 1, -1, -1):
            ep = stored[i]
            vel = -ep['velocity'] if ep['velocity'] is not None else None
            bwd_pf.predict(velocity=vel, dt=ep['dt'])

            sat = ep['sat_ecef'].reshape(-1, 3)
            pr = ep['pseudoranges']
            w = ep['weights']
            dd_ep = ep.get('dd_pseudorange')
            dd_ep_source = ep.get('dd_pseudorange_source')
            skip_dd_ep = (
                bool(skip_widelane_dd_pseudorange)
                and dd_ep_source == 'widelane'
            )
            dd_cp_ep = ep.get('dd_carrier')
            carrier_anchor_ep = ep.get('carrier_anchor_pseudorange')
            carrier_afv_ep = ep.get('carrier_afv')
            # Reverse the complete range-rate observation model, not only the
            # receiver motion used by predict().  Under tau=-t, both satellite
            # velocity and measured range rate change sign; since
            # range_rate=-wavelength*doppler_hz, Doppler changes sign too.
            # This makes the ordinary forward update solve for v_tau=-v_t and
            # preserves the clock-drift nuisance sign consistently.
            doppler_ep = ep.get('doppler_update')
            if dd_ep is not None and not skip_dd_ep:
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
            if doppler_ep is not None:
                bwd_pf.update_doppler(
                    doppler_ep['sat_ecef'],
                    -doppler_ep['sat_vel'],
                    -doppler_ep['doppler_hz'],
                    weights=doppler_ep['weights'],
                    wavelength=float(doppler_ep.get('wavelength_m', 0.19029367279836488)),
                    sigma_mps=(
                        float(ep['doppler_sigma_mps'])
                        if ep.get('doppler_sigma_mps') is not None
                        else 0.5
                    ),
                    velocity_update_gain=(
                        float(ep['doppler_velocity_update_gain'])
                        if ep.get('doppler_velocity_update_gain') is not None
                        else 0.25
                    ),
                    max_velocity_update_mps=(
                        float(ep['doppler_max_velocity_update_mps'])
                        if ep.get('doppler_max_velocity_update_mps') is not None
                        else 10.0
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
