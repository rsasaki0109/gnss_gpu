"""Configuration and native binding setup for ParticleFilterDevice."""

from __future__ import annotations

from types import SimpleNamespace

from gnss_gpu.input_validation import finite_float, positive_float


def _positive_int(name, value):
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def load_pf_device_bindings() -> SimpleNamespace:
    """Import CUDA pf_device entrypoints once per process."""
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
        pf_device_weight_dd_joint,
        pf_device_weight_doppler,
        pf_device_doppler_kf_update,
        pf_device_position_update,
        pf_device_shift_clock_bias,
        pf_device_shift_position,
        pf_device_ess,
        pf_device_position_spread,
        pf_device_resample_systematic,
        pf_device_resample_megopolis,
        pf_device_estimate,
        pf_device_get_particles,
        pf_device_get_particle_states,
        pf_device_set_particle_states,
        pf_device_get_log_weights,
        pf_device_set_log_weights,
        pf_device_get_resample_ancestors,
        pf_device_sync,
    )

    return SimpleNamespace(
        pf_device_create=pf_device_create,
        pf_device_destroy=pf_device_destroy,
        pf_device_initialize=pf_device_initialize,
        pf_device_predict=pf_device_predict,
        pf_device_weight=pf_device_weight,
        pf_device_weight_dd_pseudorange=pf_device_weight_dd_pseudorange,
        pf_device_weight_gmm=pf_device_weight_gmm,
        pf_device_weight_carrier_afv=pf_device_weight_carrier_afv,
        pf_device_weight_dd_carrier_afv=pf_device_weight_dd_carrier_afv,
        pf_device_weight_dd_joint=pf_device_weight_dd_joint,
        pf_device_weight_doppler=pf_device_weight_doppler,
        pf_device_doppler_kf_update=pf_device_doppler_kf_update,
        pf_device_position_update=pf_device_position_update,
        pf_device_shift_clock_bias=pf_device_shift_clock_bias,
        pf_device_shift_position=pf_device_shift_position,
        pf_device_ess=pf_device_ess,
        pf_device_position_spread=pf_device_position_spread,
        pf_device_resample_systematic=pf_device_resample_systematic,
        pf_device_resample_megopolis=pf_device_resample_megopolis,
        pf_device_estimate=pf_device_estimate,
        pf_device_get_particles=pf_device_get_particles,
        pf_device_get_particle_states=pf_device_get_particle_states,
        pf_device_set_particle_states=pf_device_set_particle_states,
        pf_device_get_log_weights=pf_device_get_log_weights,
        pf_device_set_log_weights=pf_device_set_log_weights,
        pf_device_get_resample_ancestors=pf_device_get_resample_ancestors,
        pf_device_sync=pf_device_sync,
    )


_BINDING_ATTRS = (
    "pf_device_create",
    "pf_device_destroy",
    "pf_device_initialize",
    "pf_device_predict",
    "pf_device_weight",
    "pf_device_weight_dd_pseudorange",
    "pf_device_weight_gmm",
    "pf_device_weight_carrier_afv",
    "pf_device_weight_dd_carrier_afv",
    "pf_device_weight_dd_joint",
    "pf_device_weight_doppler",
    "pf_device_doppler_kf_update",
    "pf_device_position_update",
    "pf_device_shift_clock_bias",
    "pf_device_shift_position",
    "pf_device_ess",
    "pf_device_position_spread",
    "pf_device_resample_systematic",
    "pf_device_resample_megopolis",
    "pf_device_estimate",
    "pf_device_get_particles",
    "pf_device_get_particle_states",
    "pf_device_set_particle_states",
    "pf_device_get_log_weights",
    "pf_device_set_log_weights",
    "pf_device_get_resample_ancestors",
    "pf_device_sync",
)


def attach_pf_device_bindings(pf, bindings: SimpleNamespace) -> None:
    for name in _BINDING_ATTRS:
        setattr(pf, f"_{name}", getattr(bindings, name))


def init_pf_device_config(
    pf,
    *,
    n_particles=1_000_000,
    sigma_pos=1.0,
    sigma_cb=300.0,
    sigma_pr=5.0,
    nu=0.0,
    resampling="megopolis",
    ess_threshold=0.5,
    seed=42,
    per_particle_nlos_gate=False,
    per_particle_nlos_dd_pr_threshold_m=10.0,
    per_particle_nlos_dd_carrier_threshold_cycles=0.5,
    per_particle_nlos_undiff_pr_threshold_m=30.0,
    per_particle_huber=False,
    per_particle_huber_dd_pr_k=1.5,
    per_particle_huber_dd_carrier_k=1.5,
    per_particle_huber_undiff_pr_k=1.5,
    sigma_vel=0.0,
    velocity_guide_alpha=1.0,
    rbpf_velocity_kf=False,
    velocity_process_noise=0.0,
    bindings: SimpleNamespace | None = None,
) -> None:
    """Validate constructor args, attach native hooks, and allocate GPU state."""
    if bindings is None:
        bindings = load_pf_device_bindings()
    attach_pf_device_bindings(pf, bindings)

    pf.n_particles = _positive_int("n_particles", n_particles)
    pf.sigma_pos = positive_float("sigma_pos", sigma_pos)
    pf.sigma_cb = positive_float("sigma_cb", sigma_cb)
    pf.sigma_pr = positive_float("sigma_pr", sigma_pr)
    pf.nu = finite_float("nu", nu)
    pf.resampling = resampling
    pf.ess_threshold = ess_threshold
    pf.seed = seed
    pf.per_particle_nlos_gate = bool(per_particle_nlos_gate)
    pf.per_particle_nlos_dd_pr_threshold_m = float(per_particle_nlos_dd_pr_threshold_m)
    pf.per_particle_nlos_dd_carrier_threshold_cycles = float(
        per_particle_nlos_dd_carrier_threshold_cycles
    )
    pf.per_particle_nlos_undiff_pr_threshold_m = float(
        per_particle_nlos_undiff_pr_threshold_m
    )
    pf.per_particle_huber = bool(per_particle_huber)
    pf.per_particle_huber_dd_pr_k = float(per_particle_huber_dd_pr_k)
    pf.per_particle_huber_dd_carrier_k = float(per_particle_huber_dd_carrier_k)
    pf.per_particle_huber_undiff_pr_k = float(per_particle_huber_undiff_pr_k)
    pf.sigma_vel = float(sigma_vel)
    pf.velocity_guide_alpha = float(velocity_guide_alpha)
    pf.rbpf_velocity_kf = bool(rbpf_velocity_kf)
    pf.velocity_process_noise = float(velocity_process_noise)

    pf._state = pf._pf_device_create(pf.n_particles)
    pf._initialized = False
    pf._step = 0


def clone_pf_device_init_kwargs(source) -> dict:
    """Return constructor kwargs for a sibling PF instance."""
    return {
        "n_particles": source.n_particles,
        "sigma_pos": source.sigma_pos,
        "sigma_cb": source.sigma_cb,
        "sigma_pr": source.sigma_pr,
        "nu": source.nu,
        "resampling": source.resampling,
        "ess_threshold": source.ess_threshold,
        "seed": source.seed + 1,
        "per_particle_nlos_gate": source.per_particle_nlos_gate,
        "per_particle_nlos_dd_pr_threshold_m": source.per_particle_nlos_dd_pr_threshold_m,
        "per_particle_nlos_dd_carrier_threshold_cycles": (
            source.per_particle_nlos_dd_carrier_threshold_cycles
        ),
        "per_particle_nlos_undiff_pr_threshold_m": (
            source.per_particle_nlos_undiff_pr_threshold_m
        ),
        "per_particle_huber": source.per_particle_huber,
        "per_particle_huber_dd_pr_k": source.per_particle_huber_dd_pr_k,
        "per_particle_huber_dd_carrier_k": source.per_particle_huber_dd_carrier_k,
        "per_particle_huber_undiff_pr_k": source.per_particle_huber_undiff_pr_k,
        "sigma_vel": source.sigma_vel,
        "velocity_guide_alpha": source.velocity_guide_alpha,
        "rbpf_velocity_kf": source.rbpf_velocity_kf,
        "velocity_process_noise": source.velocity_process_noise,
    }
