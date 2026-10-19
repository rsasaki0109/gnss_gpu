"""Verify bounded prepared triangle views against the retained D217 code."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_reflection_geometry_compare import load, scene, tree_hash
from gnss_gpu import raytrace
from test_reflection_geometry import outcome


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/reflection_barycentric_baseline.py', 'reflection_barycentric_oracle')


def trace_point_dots(monkeypatch, model, rx, sat):
    original, calls, identities = np.dot, [], {}

    def observed(a, b, *args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code.co_name == '_point_in_triangle':
            calls.append((a.dtype.str, a.tobytes(), b.dtype.str, b.tobytes()))
            vectors = caller.f_locals.get('triangle_vectors')
            if vectors is not None:
                index = caller.f_locals['triangle_id']
                cached = vectors[index]
                identities.setdefault(index, set()).add(id(cached))
                geometry = caller.f_locals['geometry'] if 'geometry' in caller.f_locals else cached[5].base
                assert np.shares_memory(cached[0], model.triangles)
                assert all(np.shares_memory(value, geometry) for value in (*cached[1:4], cached[5]))
                assert all(not value.flags.owndata for value in (*cached[:4], cached[5]))
        return original(a, b, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(np, 'dot', observed)
        result = model.compute_reflection_paths(rx, sat)
    return result, calls, identities


@pytest.mark.parametrize('kind', ['wall', 'blocked', 'grid'])
@pytest.mark.parametrize('count', [12, 24, 120])
@pytest.mark.parametrize('satellites', [2, 12])
def test_same_dot_input_bits_order_and_paths_with_shared_prepared_views(oracle, monkeypatch, kind, count, satellites):
    old, rx, sat = scene(oracle, count, satellites, kind)
    new = raytrace.BuildingModel(old.triangles.copy())
    before, old_dots, _ = trace_point_dots(monkeypatch, old, rx, sat)
    after, new_dots, identities = trace_point_dots(monkeypatch, new, rx, sat)
    assert old_dots == new_dots
    assert tree_hash(before) == tree_hash(after)
    assert identities and all(len(values) == 1 for values in identities.values())
    if kind == 'wall':
        assert sum(map(len, before)) > 0


@pytest.mark.parametrize('count', [7, 8, 32768, 32769])
def test_prepared_view_entry_bounds_before_first_triangle(count, monkeypatch):
    model = raytrace.BuildingModel(np.zeros((count, 3, 3)))
    seen = []

    def first_cross(*args, **kwargs):
        frame = sys._getframe(1)
        while frame.f_code.co_name != 'compute_reflection_paths':
            frame = frame.f_back
        vectors = frame.f_locals['triangle_vectors']
        seen.append(None if vectors is None else len(vectors))
        raise RuntimeError('first cross reached')

    monkeypatch.setattr(np, 'cross', first_cross)
    with pytest.raises(RuntimeError, match='first cross reached'):
        model.compute_reflection_paths([0., 0., 1.], [[1., 2., 3.], [2., 3., 4.]])
    assert seen == ([count] if 8 <= count <= 32768 else [None])


@pytest.mark.parametrize('case', ['single', 'float32', 'strict', 'underflow_warn', 'small'])
def test_scalar_fallback_does_not_create_prepared_views(oracle, monkeypatch, case):
    old, rx, sat = scene(oracle, 12, 2, 'wall')
    policy = 'default'
    if case == 'single':
        sat = sat[:1]
    elif case == 'float32':
        old.triangles = old.triangles.astype(np.float32)
    elif case == 'strict':
        policy = 'raise'
    elif case == 'underflow_warn':
        policy = 'warn'
    else:
        old.triangles = old.triangles[:7]
    new = raytrace.BuildingModel(np.asarray(old.triangles, dtype=np.float64))
    new.triangles = old.triangles.copy()
    original, seen = np.cross, []

    def observe(*args, **kwargs):
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_name != 'compute_reflection_paths':
            frame = frame.f_back
        if frame is not None and frame.f_globals['__name__'] == raytrace.__name__:
            seen.append(frame.f_locals['triangle_vectors'])
        return original(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(np, 'cross', observe)
        assert outcome(old, rx, sat, policy) == outcome(new, rx, sat, policy)
    assert seen and all(value is None for value in seen)


@pytest.mark.parametrize('layout', ['fortran', 'strided', 'reversed'])
def test_mesh_views_keep_original_dot_and_output_bits(oracle, monkeypatch, layout):
    old, rx, sat = scene(oracle, 24, 4, 'wall')
    if layout == 'fortran':
        mesh = np.asfortranarray(old.triangles)
    elif layout == 'strided':
        mesh = np.repeat(old.triangles, 2, axis=0)[::2]
    else:
        mesh = old.triangles[::-1, ::-1]
    old.triangles = mesh
    new = raytrace.BuildingModel(mesh.copy())
    new.triangles = mesh.copy(order='K')
    a, before, _ = trace_point_dots(monkeypatch, old, rx, sat)
    b, after, _ = trace_point_dots(monkeypatch, new, rx, sat)
    assert before == after and tree_hash(a) == tree_hash(b)


def test_views_do_not_escape_results_or_survive_input_changes(oracle):
    old, rx, sat = scene(oracle, 24, 4, 'wall')
    new = raytrace.BuildingModel(old.triangles.copy())
    before = new.compute_reflection_paths(rx, sat)
    assert sum(map(len, before)) > 0
    original = tree_hash(before)
    for paths in before:
        for path in paths:
            assert path.normal.flags.owndata and path.reflection_point.flags.owndata
            assert not np.shares_memory(path.normal, new.triangles)
            assert not np.shares_memory(path.reflection_point, new.triangles)
    for shift in [np.array([.25, .5, .75]), np.array([-.5, .25, -.25])]:
        old.triangles += shift
        new.triangles += shift
        rx += shift*.5
        assert outcome(old, rx, sat) == outcome(new, rx, sat)
        assert tree_hash(before) == original
