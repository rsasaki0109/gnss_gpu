"""Coarse observation/IMU/motion-factor diagnosis over the frozen D245 probe.

Only call boundaries expand. No Python/C function profiler, numerical
rewrites, cache, model mutation or production dispatch changes.
"""

import hashlib
from pathlib import Path
from types import FunctionType

import rbpf_call_profile as base

BASE_SHA256 = "879fca7ddce87766909a355919223b694b127c7e9da05abb110e3addbf37e8c9"
METHODS = (
    "prepare_double_difference_measurements",
    "single_differences",
    "qcedit",
    "update_ambiguities",
    "_manage_ambiguities",
    "_reset_slipped_ambiguities",
    "_tc_literal_gate_stage",
    "_add_doppler_vel_prior",
    "_add_nhc_factor",
    "_zupt_check_and_add",
    "_globalize_epoch_ddpr_tags",
    "_log_main_ddpr",
    "_compute_main_dd_res",
    "_update_cp_hold",
    "_ddpr_sanity_check",
)
CALLS = {
    "build_pim": "rtk.build_pim",
    "compute_zupt_stats": "rtk.compute_zupt_stats",
    "_tc_doppler_velocity_ls": "rtk.doppler_velocity_ls",
    "satposs": "rtk.satposs",
}


class RtkWorkMonitor(base.ProcessMonitor):
    """Reuse the verified AST forwarding in a separate globals dictionary.

    This does not change the frozen helper's source, globals, method list or
    call map. The original __class__ cell still belongs to each source class.
    """

    def instrument_type(self, original):
        path = Path(base.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
            raise ValueError("Frozen span/AST helper changed")
        scope = dict(vars(base))
        scope.update(
            _METHODS=tuple(dict.fromkeys((*base._METHODS, *METHODS))),
            _C_CALLS={**base._C_CALLS, **CALLS},
            _ADAPTER="_d246_call",
        )
        original_adapter = base._instrument_function
        scope["_instrument_function"] = FunctionType(
            original_adapter.__code__,
            scope,
            original_adapter.__name__,
            original_adapter.__defaults__,
            original_adapter.__closure__,
        )
        original_instrument = base.ProcessMonitor.instrument_type
        instrument = FunctionType(
            original_instrument.__code__,
            scope,
            original_instrument.__name__,
            original_instrument.__defaults__,
            original_instrument.__closure__,
        )
        return instrument(self, original)
