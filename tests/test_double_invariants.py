"""Compare bounded double-image reuse with the independent original scalar code."""

import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_double_reflection_invariants_compare import make_call, scene
from bench_reflection_geometry_compare import load, tree_hash
from gnss_gpu import double_reflection as current


@pytest.fixture(scope='module')
def oracle():
    return load(Path(__file__).parent/'oracles/double_invariants_scalar.py', 'double_invariants_oracle')


def outcome(module, mesh, rx, sat, policy='default', **kwargs):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        settings = np.geterr() if policy == 'default' else dict.fromkeys(np.geterr(), policy)
        with np.errstate(**settings):
            try:
                result = module.compute_double_reflection_paths(mesh, rx, sat, **kwargs)
                value = ('return', tree_hash(result))
            except (ValueError, RuntimeError, FloatingPointError, TypeError, OverflowError) as exc:
                value = ('error', type(exc).__name__, str(exc))
    return value, [(type(row.message).__name__, str(row.message)) for row in recorded]


def trace(module, mesh, rx, sat):
    pairs, cached, mirrors, predicates = [], [], hashlib.sha256(), hashlib.sha256()

    def profile(frame, event, arg):
        if frame.f_globals.get('__name__') != module.__name__:
            return
        name, values = frame.f_code.co_name, frame.f_locals
        if name == '_compute_single_path':
            if event == 'call':
                pairs.append((values['tri_a_id'], values['tri_b_id']))
                cached.append(values.get('invariants') is not None)
            elif event == 'return':
                mirrors.update(tree_hash(tuple(values.get(key) for key in
                    ('rx_plane_distance', 'sat_plane_distance', 'i1', 'i12'))).encode())
        elif event == 'call' and name in ('_point_in_triangle', '_segment_intersects_triangle'):
            predicates.update(name.encode())
            keys = ('point', 'triangle') if name == '_point_in_triangle' else ('start', 'end', 'triangle')
            predicates.update(tree_hash(tuple(values[key] for key in keys)).encode())

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = module.compute_double_reflection_paths(mesh, rx, sat)
    finally:
        sys.setprofile(previous)
    return result, pairs, mirrors.hexdigest(), predicates.hexdigest(), cached


@pytest.mark.parametrize('kind', ['corner', 'blocked'])
@pytest.mark.parametrize('count', [14, 26])
@pytest.mark.parametrize('satellites', [2, 8, 12])
def test_every_ordered_pair_mirror_bits_and_predicate_sequence_match(oracle, kind, count, satellites):
    model, rx, sat = scene(count, satellites, kind)
    before = trace(oracle, model.triangles, rx, sat)
    after = trace(current, model.triangles, rx, sat)
    assert tree_hash(before[0]) == tree_hash(after[0])
    assert before[1:4] == after[1:4]
    assert len(after[1]) == satellites*count*(count-1)
    assert all(after[4]) == (satellites*count*(count-1) >= 1024)
    if kind == 'corner':
        assert sum(map(len, before[0])) > 0


@pytest.mark.parametrize('policy', ['default', 'ignore', 'warn', 'raise'])
@pytest.mark.parametrize('scale', [1e-160, 1e-16, 1., 1e25, 1e50, np.nextafter(1e50, np.inf), 1e100])
def test_extremes_policies_degenerate_normals_warnings_and_exceptions(oracle, policy, scale):
    rng = np.random.default_rng(219)
    mesh = rng.normal(size=(9, 3, 3))*scale
    mesh[0] = 0.
    mesh[1, 2] = mesh[1, 0]
    rx, sat = np.array([3., 4., 5.])*scale, rng.normal(size=(15, 3))*scale*20
    assert outcome(oracle, mesh, rx, sat, policy) == outcome(current, mesh, rx, sat, policy)


@pytest.mark.parametrize('where', ['mesh', 'receiver', 'satellite'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_preserve_original_warning_order(oracle, where, value):
    model, rx, sat = scene(14, 12)
    mesh = model.triangles.copy()
    if where == 'mesh':
        mesh[0, 0, 0] = value
    elif where == 'receiver':
        rx[0] = value
    else:
        sat[0, 0] = value
    assert outcome(oracle, mesh, rx, sat) == outcome(current, mesh, rx, sat)


@pytest.mark.parametrize('count,satellites,limit,expected', [
    (7, 32, None, False), (8, 32, None, False),
    (9, 14, None, False), (9, 15, None, True),
    (14, 1, None, False), (14, 5, None, False), (14, 6, None, True),
    (14, 32, 63, False), (14, 16, 64, True),
    (182, 2, 32768, True), (182, 2, 32769, False),
    (257, 16, 64, False),
])
def test_cache_work_and_size_bounds_without_skipping_any_pair(count, satellites, limit, expected):
    mesh = np.zeros((count, 3, 3))
    seen = []

    def first_pair(frame, event, arg):
        if event == 'call' and frame.f_code is current._compute_single_path.__code__:
            cached = frame.f_locals['invariants']
            seen.append(cached is not None)
            if cached is not None:
                assert len(cached.pair_images) == (count*(count-1) if limit is None else limit)
            raise RuntimeError('first ordered pair reached')

    previous = sys.getprofile()
    try:
        sys.setprofile(first_pair)
        with pytest.raises(RuntimeError, match='first ordered pair reached'):
            current.compute_double_reflection_paths(mesh, [0., 0., 1.], np.ones((satellites, 3))*100,
                max_pairs=limit)
    finally:
        sys.setprofile(previous)
    assert seen == [expected]


@pytest.mark.parametrize('paths', [None, 0, 1, 4])
@pytest.mark.parametrize('pairs', [None, 0, 1, 63, 128])
def test_existing_path_pair_limits_and_stable_order_are_preserved(oracle, paths, pairs):
    model, rx, sat = scene(14, 12)
    assert outcome(oracle, model.triangles, rx, sat, max_paths=paths, max_pairs=pairs) == outcome(
        current, model.triangles, rx, sat, max_paths=paths, max_pairs=pairs)


@pytest.mark.parametrize('name', ['_compute_single_path', '_triangle_normal', '_norm', '_ordered_pairs'])
def test_replaced_private_hooks_keep_original_signature_and_call_sequence(oracle, monkeypatch, name):
    model, rx, sat = scene(14, 12)
    results, counts = [], []
    for module in (oracle, current):
        original, calls = getattr(module, name), []

        def wrapped(*args):
            calls.append(len(args))
            if name == '_compute_single_path':
                assert len(args) == 7
            return original(*args)

        with monkeypatch.context() as context:
            context.setattr(module, name, wrapped)
            results.append(outcome(module, model.triangles, rx, sat))
        counts.append(calls)
    assert results[0] == results[1] and counts[0] == counts[1]


@pytest.mark.parametrize('name', ['area_eps', 'plane_eps'])
def test_changed_module_thresholds_keep_original_results(oracle, monkeypatch, name):
    model, rx, sat = scene(14, 12)
    with monkeypatch.context() as context:
        context.setattr(oracle, name, 1e-8)
        context.setattr(current, name, 1e-8)
        assert outcome(oracle, model.triangles, rx, sat) == outcome(current, model.triangles, rx, sat)


@pytest.mark.parametrize('layout', ['fortran', 'strided', 'reversed'])
def test_array_layout_and_vector_views_preserve_all_output_bits(oracle, layout):
    model, rx, sat = scene(14, 12)
    mesh = model.triangles
    if layout == 'fortran':
        mesh, sat = np.asfortranarray(mesh), np.asfortranarray(sat)
    elif layout == 'strided':
        mesh, sat = np.repeat(mesh, 2, axis=0)[::2], np.repeat(sat, 2, axis=0)[::2]
    else:
        mesh, rx, sat = mesh[::-1, ::-1], rx[::-1], sat[::-1, ::-1]
    assert outcome(oracle, mesh, rx, sat) == outcome(current, mesh, rx, sat)


def test_cache_is_call_local_and_output_arrays_are_independent(oracle):
    model, rx, sat = scene(14, 12)
    saved = current.compute_double_reflection_paths(model.triangles, rx, sat)
    initial = tree_hash(saved)
    assert sum(map(len, saved)) > 0
    for paths in saved:
        for path in paths:
            for value in [*path.points, *path.normals]:
                assert value.flags.owndata and not np.shares_memory(value, model.triangles)
    for shift in [np.array([.25, .5, .75]), np.array([-.5, .25, -.25])]:
        model.triangles += shift
        rx += shift*.5
        assert outcome(oracle, model.triangles, rx, sat) == outcome(current, model.triangles, rx, sat)
        assert tree_hash(saved) == initial


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize('satellites', [2, 12])
@pytest.mark.parametrize('samples', [31, 2600, 260000])
def test_real_single_double_reflection_CUDA_IQ_and_metadata_match(oracle, satellites, samples):
    calls, owners = [], []
    try:
        for module in (oracle, current):
            call, owner, _ = make_call(module, 14, satellites, samples)
            calls.append(call)
            owners.append(owner)
        a, b = calls[0](), calls[1]()
        assert a['n_double_reflection_paths'] > 0
        assert tree_hash(a) == tree_hash(b)
    finally:
        for owner in owners:
            owner.sim.release_gpu()
