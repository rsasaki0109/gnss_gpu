"""Feedback reuse preserves release/commit policy, counters and fresh state."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_feedback_dd import (GraphReplay, evaluator, feedback_fields, graph_state,  # noqa: E402
                                   outcome, pack, unpack)

SOURCE = HERE/"rbpf_fgo.py"
FIELDS, _ = feedback_fields(SOURCE)
G = (0, 0)
PAIRS = [(G, 1, 2), (G, 1, 3)]


def scalar_dd(H, prob):
    result = np.full(len(prob.pairs), np.nan)
    for i, (g, ref, sat) in enumerate(prob.pairs):
        z = H.get(g, {})
        if z.get(ref) is not None and z.get(sat) is not None:
            result[i] = z[ref]-z[sat]
    return result


class Graph:
    def __init__(self):
        self._committed = {(1, 0): 0., (2, 0): -2., (3, 0): -4.}
        self.amb_keys = dict.fromkeys(self._committed)
        self.nav = SimpleNamespace(x=np.array([.3, -2., -4.]), na=3)
        self.events = []

    def IB(self, sat, frequency, na):
        assert frequency == 0 and na == 3
        return sat-1

    def _release_ambiguity(self, sat, frequency, reason=""):
        key = (sat, frequency)
        self.events.append(("release", key, reason))
        self._committed.pop(key, None)
        self.amb_keys.pop(key, None)
        self.nav.x[sat-1] = 0.

    def _inject_hold(self, xa, newly):
        self.events.append(("inject", xa.copy(), list(newly)))
        self.nav.x[:] = xa


def setup(kind, prepared, legacy):
    owner, epoch, accepts = evaluator(SOURCE, legacy)
    owner.cfg = SimpleNamespace(fb_support_min=0, fb_support_m=1, spawn_enable=1, spawn_epochs=25,
        fb_ddpr_rel_d=0., fb_ddpr_rel_m=1, ddpr_qual_max=1., fb_rel_mass=.5, fb_rel_m=3,
        fb_dual_enable=0, fb_dual_min=2, fb_min_nb=1, fb_m=3, fb_witness_pos=0,
        fb_witness_chi=3., fb_commit_max_dpr=0.)
    owner.particles = [SimpleNamespace(H=H) for H in [{G: {1: 0, 2: -2, 3: -4}},
                                                     {G: {1: 0, 2: -3, 3: -4}}, {}]]
    for name in FIELDS:
        setattr(owner, name, {} if name in {'cluster_states', 'fb_bad', 'fb_streak'} else
                set() if name in {'carry_protect', 'fb_unseen_pending'} else 0)
    owner.cluster_states = {'prior': dict(pos=np.array([1., 2., 3.]), P=np.eye(3))}
    prob = SimpleNamespace(pairs=deepcopy(PAIRS), cands=np.array([[2., 9.], [4., 4.]]), witness_cands=None)
    if prepared:
        epoch.set_pairs(prob, prob.pairs)
    rtk, weights = Graph(), np.array([.5, .25, .25])
    dd_map = scalar_dd(owner.particles[0].H, prob)
    kwargs = dict(rep_dpr=2., ddpr_qual=.1, witness_chi=0.)
    if kind in {'mass', 'carry'}:
        owner.cfg.fb_rel_mass, owner.cfg.fb_rel_m = .6, 1
    if kind == 'carry':
        owner.carry_protect = {(2, 0)}
    if kind == 'unseen':
        owner.fb_unseen_pending = {(2, 0)}
    if kind == 'support':
        owner.cfg.fb_support_min = 3
    if kind == 'ddpr':
        owner.cfg.fb_ddpr_rel_d = 1.
    if kind == 'dual':
        owner.cfg.fb_dual_enable = 1
    if kind == 'witness':
        owner.cfg.fb_witness_pos = 1
        kwargs['witness_chi'] = float('nan')
    if kind == 'commit_gate':
        owner.cfg.fb_commit_max_dpr = 1.
    if kind in {'new_commit', 'nonfinite_seed', 'no_committed'}:
        rtk._committed.clear()
        owner.fb_support_bad_streak = 5
    if kind in {'new_commit', 'nonfinite_seed'}:
        owner.cfg.fb_m = 1
    if kind == 'nonfinite_seed':
        rtk.nav.x[0] = np.nan
    calls = []
    def counted(H, prob):
        calls.append(H)
        return scalar_dd(H, prob)
    owner._pair_dd = counted
    return (owner, epoch, accepts), rtk, prob, weights, dd_map, kwargs, calls


@pytest.mark.parametrize('kind', ['normal', 'mass', 'carry', 'unseen', 'support', 'ddpr',
                                  'dual', 'witness', 'commit_gate', 'new_commit', 'nonfinite_seed', 'no_committed'])
@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('prepared', [False, True])
def test_exact_policy_and_graph_actions(kind, reuse, prepared):
    legacy, a, prob, w, dd_map, kwargs, calls_a = setup(kind, prepared, True)
    candidate, b, new_prob, new_w, new_map, new_kwargs, calls_b = setup(kind, prepared, False)
    supplied = [scalar_dd(p.H, new_prob) for p in candidate[0].particles]
    before = pack(supplied), pack([p.H for p in candidate[0].particles])
    expected = legacy[0]._feedback(a, prob, w, dd_map, [0, 1], **kwargs)
    actual = candidate[0]._feedback(b, new_prob, new_w, new_map, [0, 1],
                                     **new_kwargs, dd_all=supplied if reuse else None)
    assert outcome(legacy[0], a, FIELDS, expected) == outcome(candidate[0], b, FIELDS, actual)
    assert pack(a.events) == pack(b.events)
    assert len(calls_a) == 3 and len(calls_b) == (0 if reuse else 3)
    assert before == (pack(supplied), pack([p.H for p in candidate[0].particles]))
    if kind == 'support':
        assert candidate[0].fb_release_support == 1 and not b._committed
    if kind == 'new_commit':
        assert candidate[0].fb_commit_count == 3 and b.events[0][0] == 'inject'
    if kind == 'carry':
        assert (2, 0) in b._committed


@pytest.mark.parametrize('mode', ['no_map', 'empty', 'missing', 'large'])
@pytest.mark.parametrize('reuse', [False, True])
def test_empty_map_population_support_and_large_values(mode, reuse):
    outcomes = []
    for legacy in [True, False]:
        evaluation, rtk, prob, w, dd_map, kwargs, _ = setup('normal', True, legacy)
        owner = evaluation[0]
        if mode == 'no_map':
            dd_map = None
        elif mode == 'empty':
            owner.particles, w = [], np.array([])
        elif mode == 'missing':
            owner.particles[0].H = {G: {2: 4}}
        elif mode == 'large':
            owner.particles[0].H[G][2] = -(2**60+1)
        dd_all = [scalar_dd(p.H, prob) for p in owner.particles]
        if not legacy:
            kwargs['dd_all'] = dd_all if reuse else None
        value = owner._feedback(rtk, prob, w, dd_map, [], **kwargs)
        outcomes.append(outcome(owner, rtk, FIELDS, value))
    assert outcomes[0] == outcomes[1]


def test_graph_transition_playback_preserves_local_committed_alias():
    rtk = Graph()
    before = pack(graph_state(rtk))
    rtk._release_ambiguity(2, 0, reason='rbpf_mass')
    event = dict(name='release', args=pack((2, 0, 'rbpf_mass')), before=before,
                 after=graph_state(rtk), **{'return': None})
    replay = GraphReplay(unpack(before), [event], {(1, 0): 0}, True)
    alias = replay._committed
    replay._release_ambiguity(2, 0, reason='rbpf_mass')
    assert alias is replay._committed and (2, 0) not in alias
    assert pack(graph_state(replay)) == pack(graph_state(rtk))


def test_legacy_calls_recompute_after_hypothesis_and_pair_changes():
    evaluation, rtk, prob, weights, dd_map, kwargs, calls = setup('normal', True, False)
    owner = evaluation[0]
    owner._feedback(rtk, prob, weights, dd_map, [], **kwargs)
    owner.particles[0].H[G][2] = -17
    evaluation[1].set_pairs(prob, PAIRS[::-1])
    owner._feedback(rtk, prob, weights, None, [], **kwargs)
    assert len(calls) == 6
    assert scalar_dd(owner.particles[0].H, prob).tolist() == [4., 17.]


def test_snapshot_roundtrip_preserves_dtype_order_nonfinite_and_keys():
    value = {(np.int64(1), 0): [np.array([np.nan, -0., np.inf], dtype=np.float64),
                                np.array([1, -2], dtype=np.int32), {1, 3}, np.float32(.1), float('nan')]}
    assert pack(unpack(pack(value))) == pack(value)
