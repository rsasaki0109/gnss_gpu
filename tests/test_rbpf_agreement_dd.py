"""Phase-local DD reuse preserves decisions, scalar sums and fresh inputs."""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_agreement_dd import evaluator, invoke, result_digest  # noqa: E402
from benchmark_agreement_dd_materialization import dd_digest, materializer  # noqa: E402
from benchmark_dd_layout import epoch_type  # noqa: E402

THRESHOLDS = (.5, .8, .9, .95, .98, .99)
G = (0, 0)
A, B = {G: {1: 0, 2: -2, 3: 7}}, {G: {1: 10, 2: 8}}
PAIRS = [(G, 1, 2), (G, 1, 3), (G, 2, 1), ((1, 0), 8, 9)]


def scalar_dd(H, prob):
    out = np.full(len(prob.pairs), np.nan)
    for pi, (g, ref, sat) in enumerate(prob.pairs):
        zmap = H.get(g)
        if zmap is not None and zmap.get(ref) is not None and zmap.get(sat) is not None:
            out[pi] = zmap[ref]-zmap[sat]
    return out


def oracle(owner, prob, w, ranked, rep, map_H):
    """Original agreement sums and independent scalar DD generation."""
    dd_map = scalar_dd(map_H, prob) if map_H else None
    pair_mass, used_pairs, gamma_min_used = {}, [], 0.0
    if dd_map is not None:
        dd_all = [scalar_dd(p.H, prob) for p in owner.particles]
        for pi_ in np.where(np.isfinite(dd_map))[0]:
            am = 0.0
            for i_, p in enumerate(owner.particles):
                di = dd_all[i_][pi_]
                if np.isfinite(di) and int(di) == int(dd_map[pi_]):
                    am += float(w[i_])
            pair_mass[int(pi_)] = am
        used_pairs = [pi_ for pi_, am in pair_mass.items() if am > owner.cfg.gamma_fix]
        if used_pairs:
            gamma_min_used = min(pair_mass[pi_] for pi_ in used_pairs)
    out = dict(nb=int(len(used_pairs)), gamma_pair=float(gamma_min_used))
    out['nb_at'] = ([int(sum(1 for am in pair_mass.values() if am > t)) for t in THRESHOLDS]
                    if dd_map is not None else [0]*len(THRESHOLDS))
    ne_idx = [i_ for i_, p_ in enumerate(owner.particles) if p_.H]
    ne_mass = float(sum(w[i_] for i_ in ne_idx))
    out['ne_mass'], out['nb_at_ne'] = ne_mass, [0]*len(THRESHOLDS)
    ne_ranked = [(bid, m_) for bid, m_ in ranked if len(bid) != 0]
    if ne_ranked and ne_mass > 0:
        dd_ne = scalar_dd(rep[ne_ranked[0][0]], prob)
        dd_l = {i_: scalar_dd(owner.particles[i_].H, prob) for i_ in ne_idx}
        masses_ne = []
        for pi_ in np.where(np.isfinite(dd_ne))[0]:
            am = sum(float(w[i_]) for i_ in ne_idx
                     if np.isfinite(dd_l[i_][pi_]) and int(dd_l[i_][pi_]) == int(dd_ne[pi_]))
            masses_ne.append(am/ne_mass)
        out['nb_at_ne'] = [int(sum(1 for am in masses_ne if am > t)) for t in THRESHOLDS]
    return locals()


def record(hypotheses, weights, pairs, prepared, gamma):
    particles = [SimpleNamespace(H=deepcopy(H)) for H in hypotheses]
    weights = np.asarray(weights, dtype=float)
    masses, rep = OrderedDict(), {}
    for p, weight in zip(particles, weights):
        bid = tuple((g, tuple((s, z-zmap[min(zmap)]) for s, z in sorted(zmap.items())))
                    for g, zmap in sorted(p.H.items()) if zmap)
        masses[bid] = masses.get(bid, 0.0)+float(weight)
        rep.setdefault(bid, p.H)
    ranked = sorted(masses.items(), key=lambda kv: -kv[1])
    prob = SimpleNamespace(pairs=deepcopy(pairs))
    if prepared:
        epoch_type(HERE/"rbpf_fgo.py").set_pairs(prob, prob.pairs)
    return (prob, weights, particles, SimpleNamespace(gamma_fix=gamma), ranked, rep,
            rep[ranked[0][0]] if ranked else {}, None)


CASES = [([A, B, {}, A], [.4, .2, .1, .3]),
         ([{}, A, B, {}], [.7, .1, .1, .1]),
         ([{G: {}}, A, B, {}], [.7, .1, .1, .1]),
         ([{}, {G: {}}], [.5, .5]),
         ([{}, A], [1., 0.]),
         ([A, B, A, {}], [.25]*4),
         ([A, {G: {1: 1, 2: -1, 3: 8}}, B], [.5, .4, .1])]


@pytest.mark.parametrize("hypotheses,weights", CASES)
@pytest.mark.parametrize("pairs", [PAIRS, PAIRS[::-1], []])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("gamma", [.5, .8])
def test_exact_decisions_and_ordered_sums(hypotheses, weights, pairs, prepared, gamma):
    args = record(hypotheses, weights, pairs, prepared, gamma)
    owner = evaluator(HERE/"rbpf_fgo.py")
    owner.particles, owner.cfg = args[2:4]
    before = deepcopy([p.H for p in owner.particles]), args[1].tobytes()
    expected = oracle(owner, args[0], args[1], *args[4:7])
    observed = invoke(owner, args)
    assert result_digest(observed) == result_digest(expected)
    assert dd_digest(invoke(materializer(HERE/"rbpf_fgo.py"), args)) == dd_digest(expected)
    assert before == ([p.H for p in owner.particles], args[1].tobytes())


@pytest.mark.parametrize("empty_map", [False, True])
def test_calls_are_reused_only_when_available(empty_map):
    weights = [.1, .2, .7] if empty_map else [.5, .4, .1]
    args = record([A, B, {}], weights, PAIRS, True, .5)
    owner = evaluator(HERE/"rbpf_fgo.py")
    calls, original = [], owner._pair_dd
    def counted(H, prob):
        calls.append(H)
        return original(H, prob)
    owner._pair_dd = counted
    result = invoke(owner, args)
    assert len(calls) == (3 if empty_map else 4)
    if not empty_map:
        assert result['dd_ne'] is result['dd_map']
        assert all(result['dd_l'][i] is result['dd_all'][i] for i in result['ne_idx'])
    else:
        assert result['dd_all'] is None and result['dd_map'] is None


def test_same_hypotheses_and_new_pair_layout_are_fresh_on_each_invocation():
    args = record([A, B, {}], [.5, .4, .1], PAIRS, True, .5)
    owner = evaluator(HERE/"rbpf_fgo.py")
    first = invoke(owner, args)
    bits = first['dd_all'][0].tobytes()
    args[2][0].H[G][2] = -17
    second = invoke(owner, args)
    assert first['dd_all'][0].tobytes() == bits
    assert second['dd_all'][0][0] == 17 and not np.shares_memory(first['dd_all'][0], second['dd_all'][0])
    epoch_type(HERE/"rbpf_fgo.py").set_pairs(args[0], PAIRS[::-1])
    third = invoke(owner, args)
    assert third['dd_all'][0].tobytes() == second['dd_all'][0][::-1].tobytes()


def test_nonfinite_large_integers_and_strict_thresholds_preserve_int_comparison():
    huge = 2**60
    hypotheses = [{G: {1: 0, 2: -huge, 3: 3}}, {G: {1: 0, 2: -huge-1, 3: 4}}, {}]
    args = record(hypotheses, [.5, .3, .2], PAIRS, True, .8)
    owner = evaluator(HERE/"rbpf_fgo.py")
    owner.particles, owner.cfg = args[2:4]
    expected = oracle(owner, args[0], args[1], *args[4:7])
    observed = invoke(owner, args)
    assert result_digest(observed) == result_digest(expected)
    assert observed['pair_mass'][0] == .8 and 0 not in observed['used_pairs']
    assert np.isnan(observed['dd_all'][0][-1])
