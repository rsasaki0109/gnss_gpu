"""Local ID reuse preserves representative identity, ordering and tracked means."""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_basin_id import canonical  # noqa: E402
from benchmark_posterior_ids import evaluator, input_digest, invoke, result_digest  # noqa: E402

canon = canonical(HERE/"rbpf_fgo.py")
G, OTHER = (0, 0), (1, 1)
A = {G: {2: 5, 1: 0, 3: -7}}
GAUGE = {G: {3: 13, 1: 20, 2: 25}}
B = {G: {1: 0, 2: 5}}
HYPOTHESES = [[{}, {G: {}}, {}, {OTHER: {}}],
              [A, GAUGE, B, {}], [B, {}, A, GAUGE],
              [{OTHER: {9: -(2**60), 8: 2**60}, **A}, A, B, {}]]


def original(particles, w, cfg):
    """Original grouping and tracked-output statements, excluding top solves."""
    out, masses = {}, OrderedDict()
    for i, p in enumerate(particles):
        bid = canon(p.H)
        masses[bid] = masses.get(bid, 0.0)+float(w[i])
    ranked = sorted(masses.items(), key=lambda kv: -kv[1])
    out['n_basins'] = len(ranked)
    rep = {}
    for p in particles:
        rep.setdefault(canon(p.H), p.H)
    gamma = ranked[0][1] if ranked else 0.0
    out['gamma'] = float(gamma)
    out['second_gamma'] = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    map_H = rep[ranked[0][0]] if ranked else {}
    if cfg.track == 2 and len(cfg.track_variants) > 1:
        var_out = []
        for v in range(len(cfg.track_variants)):
            xs, ws = [], []
            for i_, p_ in enumerate(particles):
                st_ = p_.state if v == 0 else (p_.vstates[v-1] if len(p_.vstates) >= v else None)
                if st_ is not None and ranked and canon(p_.H) == ranked[0][0]:
                    xs.append(st_.x)
                    ws.append(float(w[i_]))
            if xs and sum(ws) > 0:
                var_out.append((np.average(np.asarray(xs), axis=0, weights=np.asarray(ws)), float(sum(ws))))
            else:
                var_out.append((None, float('nan')))
        out['trk_var'] = var_out
    if cfg.track:
        for key, sel in (('trk_map', lambda p_: canon(p_.H) == ranked[0][0]),
                         ('trk_mix', lambda p_: bool(p_.H))):
            xs, ws = [], []
            for i_, p_ in enumerate(particles):
                if p_.state is not None and ranked and sel(p_):
                    xs.append(p_.state.x)
                    ws.append(float(w[i_]))
            if xs and sum(ws) > 0:
                out[key] = np.average(np.asarray(xs), axis=0, weights=np.asarray(ws))
                out[key+'_mass'] = float(sum(ws))
    return locals()


def record(hypotheses, track=2, pattern="all", variants=1, reverse=False):
    def state(i, v):
        if pattern == "none" or (pattern == "mixed" and (i+v) % 3 == 0):
            return None
        return SimpleNamespace(x=np.array([1e6+i*.1+v, 2e6-i*2, 3e6+i*.03], dtype=float))
    particles = [SimpleNamespace(H=deepcopy(H), state=state(i, 0),
                                vstates=[state(i, v+1) for v in range(i % variants)])
                 for i, H in enumerate(hypotheses)]
    if reverse:
        particles.reverse()
    weights = np.array([.25]*len(particles))
    return particles, weights, SimpleNamespace(track=track, track_variants=[None]*variants), None


@pytest.mark.parametrize("hypotheses", HYPOTHESES)
@pytest.mark.parametrize("track", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["all", "mixed", "none"])
@pytest.mark.parametrize("variants", [1, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_order_mass_representatives_and_tracked_outputs(hypotheses, track, pattern, variants, reverse):
    args = record(hypotheses, track, pattern, variants, reverse)
    before = input_digest(args)
    expected = original(*args[:3])
    for stage in ["basins", "aggregation"]:
        observed = invoke(evaluator(HERE/"rbpf_fgo.py", stage)[0], args)
        assert result_digest(observed, stage) == result_digest(expected, stage)
        for bid in expected['rep']:
            assert observed['rep'][bid] is expected['rep'][bid]
        assert input_digest(args) == before


def test_each_particle_id_is_computed_once_even_with_multiple_variants():
    args = record([A, GAUGE, B, {}], variants=4)
    owner, namespace = evaluator(HERE/"rbpf_fgo.py")
    calls = []
    def counted(H):
        calls.append(H)
        return canon(H)
    namespace['_canon_basin_id'] = counted
    result = invoke(owner, args)
    assert len(calls) == len(args[0])
    assert result['rep'][canon(A)] is args[0][0].H


@pytest.mark.parametrize("weights", [[0., 0., 0., 0.], [.5, 0., .5, 0.],
                                     [.99, .005, .005, 0.], [1e-20, .5, .5, 0.]])
def test_zero_mass_ties_and_scalar_addition_order(weights):
    args = record([A, GAUGE, B, {G: {}}], variants=3)
    args[1][:] = weights
    expected = original(*args[:3])
    observed = invoke(evaluator(HERE/"rbpf_fgo.py")[0], args)
    assert result_digest(observed) == result_digest(expected)


def test_ids_are_fresh_after_same_object_hypothesis_changes_and_reordering():
    args = record([A, GAUGE, B, {}])
    owner, namespace = evaluator(HERE/"rbpf_fgo.py")
    original_canon = namespace['_canon_basin_id']
    first = invoke(owner, args)
    first_id = first['basin_ids'][0]
    args[0][0].H[G][2] = 17
    second = invoke(owner, args)
    assert second['basin_ids'][0] != first_id
    assert second['basin_ids'][0] == original_canon(args[0][0].H)
    args[0].reverse()
    third = invoke(owner, args)
    assert third['basin_ids'] == [original_canon(p.H) for p in args[0]]
    assert result_digest(third) == result_digest(original(*args[:3]))


def test_empty_population_and_nonfinite_tracked_state():
    for hypotheses in [[], [A, GAUGE, B, {}]]:
        args = record(hypotheses, variants=3)
        if args[0]:
            args[0][0].state.x[1] = np.nan
        expected = original(*args[:3])
        observed = invoke(evaluator(HERE/"rbpf_fgo.py")[0], args)
        assert result_digest(observed) == result_digest(expected)
