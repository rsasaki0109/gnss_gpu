"""Per-request guarded attribute/identity reuse inside original move staging."""

import builtins
from types import SimpleNamespace

from rbpf_epoch_memo import _EpochMemo
from rbpf_fused_staging import _MEMO, _CODE
from rbpf_native_staging import NativeStagingStep

_BINDINGS = dict(type=builtins.type, dict=builtins.dict, id=builtins.id)


class VersionedStagingStep(NativeStagingStep):
    """D236 with explicit cached/on or literal D244/off in the same extension.

    Every canonical source miss and logical counter update remains. Native
    per-request guards reuse stable instance dictionaries and retained integer
    identities; arbitrary dictionary/binding/layout changes deoptimize. No
    floating math, candidate/probability template, new GPU code or value cache.
    """

    def __init__(self, *args, cached=True, **kwargs):
        if type(cached) is not bool:
            raise TypeError("cached must be an explicit bool")
        super().__init__(*args, **kwargs)
        self.cached = cached
        self.last_versioned = None
        module = self.staging_engine.module
        if not hasattr(module, "stage_versioned"):
            raise ValueError("Explicit versioned-staging extension required")

        def stage(*stage_args):
            function = _EpochMemo.canonical
            scope, base = _MEMO.__globals__, _MEMO.__builtins__
            eligible = (
                function is _MEMO
                and function.__code__ is _CODE
                and type(stage_args[0]) is SimpleNamespace
                and _EpochMemo.__getattribute__ is object.__getattribute__
                and _EpochMemo.__dict__.get("counts") is None
                and _EpochMemo.__dict__.get("identities") is None
                and all(scope.get(n, base.get(n)) is v for n, v in _BINDINGS.items())
            )
            if eligible:
                call = module.stage_versioned if cached else module.stage_fused
                held, stats = call(*stage_args, _MEMO, _CODE, _EpochMemo)
                self.last_versioned = dict(enabled=True, cached=cached, **stats)
                return held
            self.last_versioned = dict(enabled=False, cached=False)
            return module.stage(*stage_args)

        self.staging_engine.module = SimpleNamespace(stage=stage)

    def __call__(self, rtk, ctx):
        self.last_versioned = None
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["versioned_staging"] = self.last_versioned
