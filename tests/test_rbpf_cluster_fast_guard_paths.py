"""Live globals and path-domain restriction without hot filesystem resolution."""

from pathlib import Path
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import rbpf_cluster_batch as initial  # noqa: E402
from rbpf_cluster_fast_guard import ClusterBatchEngine, _literal, guard_AST_exact  # noqa: E402
from rbpf_irls_replay import oracle_from_snapshot  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402


def setup():
    names = ['GNSS_GPU_RBPF_CLUSTER_' + s for s in ('EXT', 'CANONICAL_EXT', 'PARENT_EXT')]
    if not all(os.environ.get(n) for n in names):
        pytest.skip('Explicit standalone extensions required')
    raw, problem, hs = make_case(8, cluster=True)
    oracle = oracle_from_snapshot(dict(cfg=vars(raw.cfg), cluster_states=raw.cluster_states),
        os.environ[names[1]], os.environ[names[2]])
    return ClusterBatchEngine(os.environ[names[0]]), oracle, problem, hs


def test_same_live_globals_and_literal_eligible_body():
    engine, oracle, problem, _ = setup()
    function = engine._eligible.__func__
    assert guard_AST_exact()
    assert function.__globals__ is initial.ClusterBatchEngine._eligible.__globals__
    assert function.__code__.co_freevars == ('_literal',)
    assert function.__closure__[0].cell_contents is _literal
    assert engine._eligible(problem, oracle) is not None


def test_no_hot_path_resolution_or_file_reads(monkeypatch):
    engine, oracle, problem, hs = setup()
    expected = initial._PLAIN(hs, problem, oracle)
    def forbidden(*args, **kwargs):
        raise AssertionError('Hot guard filesystem access')
    monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    actual = engine.prepare(hs, problem, oracle)
    assert actual.tobytes() == expected.tobytes() and engine.batch_calls == 1


@pytest.mark.parametrize('path', ['unknown.py', '../rbpf_fgo.py', 'C:/unrelated/rbpf_fgo.py'])
def test_unknown_filename_retains_original_block(path):
    engine, oracle, problem, hs = setup()
    function = oracle._cluster_state.__func__
    function.__code__ = function.__code__.replace(co_filename=path)
    assert not _literal(function, '_cluster_state')
    expected = initial._PLAIN(hs, problem, oracle)
    actual = engine.prepare(hs, problem, oracle)
    assert actual.tobytes() == expected.tobytes() and engine.batch_calls == 0 and engine.fallback_calls == 1
