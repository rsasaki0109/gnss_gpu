"""Explicit experimental adapter for the frozen RBPF numeric model only."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

from rbpf_likelihood_cases import SCALAR_SHA256


def runtime_reference(runtime):
    """Bind original methods without replacing any runtime/graph/particle state.

    A newer model must receive its own audited GPU implementation. Reject it
    instead of silently scoring its hypotheses with the frozen model's formulas.
    This adapter does not stage moves or alter ``step`` and has no auto dispatch.
    """
    bound = {}
    for name in ('_loglik', '_pair_dd', '_cluster_state'):
        method = getattr(runtime, name)
        function = getattr(method, '__func__', method)
        source = Path(function.__code__.co_filename)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != SCALAR_SHA256:
            raise ValueError('This GPU prototype supports only the pinned frozen RBPF model')
        bound[name] = method
    function = getattr(bound['_loglik'], '__func__', bound['_loglik'])
    canonical = function.__globals__['_canon_basin_id']
    return SimpleNamespace(cfg=runtime.cfg, basin_id=canonical, **bound)
