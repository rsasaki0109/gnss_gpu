"""Fuse pure canonical-memo hits with score-index reuse inside native staging."""

import builtins
import hashlib
from pathlib import Path
from types import SimpleNamespace

from rbpf_epoch_memo import _EpochMemo
from rbpf_native_canonical import _code_key
from rbpf_native_staging import NativeStagingStep

_MEMO = _EpochMemo.canonical
_CODE = _MEMO.__code__
_BINDINGS = dict(type=builtins.type, dict=builtins.dict, id=builtins.id)
_SOURCE = Path(__file__).with_name("rbpf_epoch_memo.py")
if (
    hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    != "178c4302ee1e6c6dcaea05be4e91161e82ce407168666cd864a1449f81bdedeb"
):
    raise ValueError("Frozen memo source changed")
_MODULE_CODE = compile(_SOURCE.read_bytes(), str(_SOURCE), "exec", dont_inherit=True)
_CLASS_CODE = next(
    c for c in _MODULE_CODE.co_consts if getattr(c, "co_name", None) == "_EpochMemo"
)
_LITERAL_CODE = next(
    c for c in _CLASS_CODE.co_consts if getattr(c, "co_name", None) == "canonical"
)
if _code_key(_CODE) != _code_key(_LITERAL_CODE):
    raise ValueError("Canonical memo code is no longer literal")


class FusedStagingStep(NativeStagingStep):
    """Explicit on/off experiment; original numerical/choice tail unchanged.

    Off executes D241 registration in the same extension. On removes only
    pure memo-hit Python calls and already completed builtin key lookups.
    Source canonical calls on misses and all logical counters remain.
    No probability cache, RNG rewrite or cross-epoch cache is introduced.
    """

    def __init__(self, *args, fusion=True, **kwargs):
        if type(fusion) is not bool:
            raise TypeError("fusion must be an explicit bool")
        super().__init__(*args, **kwargs)
        self.fusion = fusion
        self.last_fusion = None
        module = self.staging_engine.module
        if not hasattr(module, "stage_fused"):
            raise ValueError("Explicit fused-staging extension required")

        def stage(*stage_args):
            function = _EpochMemo.canonical
            scope, base = _MEMO.__globals__, _MEMO.__builtins__
            eligible = (
                fusion
                and function is _MEMO
                and function.__code__ is _CODE
                and _EpochMemo.__getattribute__ is object.__getattribute__
                and _EpochMemo.__dict__.get("counts") is None
                and _EpochMemo.__dict__.get("identities") is None
                and all(scope.get(n, base.get(n)) is v for n, v in _BINDINGS.items())
            )
            if eligible:
                held, stats = module.stage_fused(*stage_args, _MEMO, _CODE, _EpochMemo)
                self.last_fusion = dict(enabled=True, **stats)
                return held
            self.last_fusion = dict(
                enabled=False,
                fused_hits=0,
                original_index_calls=0,
                invalidations=0,
                peak_entries=0,
            )
            return module.stage(*stage_args)

        self.staging_engine.module = SimpleNamespace(stage=stage)

    def __call__(self, rtk, ctx):
        self.last_fusion = None
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not None and "batch" in self.last:
                self.last["fused_staging"] = self.last_fusion
