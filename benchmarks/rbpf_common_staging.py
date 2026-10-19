"""Group pure common-suffix memo hits, preserving the literal weight/RNG tail."""

import builtins
from types import SimpleNamespace

from rbpf_epoch_memo import _EpochMemo
from rbpf_fused_staging import _MEMO, _CODE
from rbpf_native_staging import NativeStagingStep

_BINDINGS = dict(type=builtins.type, dict=builtins.dict, id=builtins.id)


class CommonStagingStep(NativeStagingStep):
    """Call-local native suffix replay; every logical move and count remains.

    The initial suffix uses original registration. Later unchanged dictionaries,
    lists, priors, memo identities and builtin keys can reuse those indices.
    Mutable/unusual callbacks deoptimize before replay. Fresh records, literal
    weights, particle order, RNG draws, selected-H copies and feedback remain.
    """

    def __init__(self, *args, common=True, **kwargs):
        if type(common) is not bool:
            raise TypeError("common must be explicit bool")
        super().__init__(*args, **kwargs)
        self.last_common = None
        module = self.staging_engine.module
        if not hasattr(module, "stage_common"):
            raise ValueError("Explicit common-staging extension required")

        def stage(*stage_args):
            scope, base = _MEMO.__globals__, _MEMO.__builtins__
            ordinary = (
                _EpochMemo.canonical is _MEMO
                and _MEMO.__code__ is _CODE
                and type(stage_args[0]) is SimpleNamespace
                and _EpochMemo.__getattribute__ is object.__getattribute__
                and _EpochMemo.__dict__.get("counts") is None
                and _EpochMemo.__dict__.get("identities") is None
                and all(scope.get(n, base.get(n)) is v for n, v in _BINDINGS.items())
            )
            if ordinary:
                call = module.stage_common if common else module.stage_fused
                held, stats = call(*stage_args, _MEMO, _CODE, _EpochMemo)
                self.last_common = dict(enabled=True, common=common, **stats)
                return held
            self.last_common = dict(enabled=False, common=False)
            return module.stage(*stage_args)

        self.staging_engine.module = SimpleNamespace(stage=stage)

    def __call__(self, rtk, ctx):
        self.last_common = None
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not previous and self.last is not None and "batch" in self.last:
                self.last["common_staging"] = self.last_common
