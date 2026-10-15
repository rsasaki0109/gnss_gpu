"""Exercise the live scalar proposal block without optional GTSAM imports."""

import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def live_moves():
    source = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo/rbpf_fgo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    start = next(i for i, n in enumerate(step.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "new_states")
    end = next(i for i in range(start, len(step.body))
               if isinstance(step.body[i], ast.If) and ast.unparse(step.body[i].test) == "cfg.track == 2")
    wrapper = ast.parse("def run(self, cfg, prob, ctx, ep, H_cons, cand_H, spawn_H, LL, ll_burst, _lp_nb, _lp_b):\n    pass").body[0]
    wrapper.body = step.body[start:end] + [ast.parse("return new_states").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = dict(np=np, math=math, _copy_H=deepcopy)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["run"]


RUN = live_moves()


def scalar_moves(owner, cfg, H_cons, candidates, spawns, score):
    """Original per-particle scoring, with the same arithmetic and RNG calls."""
    if cfg.track == 1:
        states = owner._moves_tracked(None, None, None, None, None, None, None, None)
    else:
        states = []
        for p in owner.particles:
            moves = []
            survive = score(p.H)
            if p.H:
                moves.append((cfg.p_survive, survive, p.H, "survive"))
                for prior, method, tag in [(cfg.p_extend, owner._extend, "extend"),
                                           (cfg.p_shrink, owner._shrink_worst, "shrink")]:
                    H = method(p.H, None)
                    if H is not None:
                        moves.append((prior, score(H), H, tag))
                moves.append((cfg.p_release, score({}), {}, "release"))
            else:
                moves.append((cfg.p_survive + cfg.p_extend + cfg.p_shrink + cfg.p_release,
                              survive, p.H, "stay"))
            if H_cons is not None:
                moves.append((cfg.p_adopt_cons, score(H_cons), H_cons, "adopt_cons"))
            for H in candidates:
                moves.append((cfg.p_adopt / len(candidates), score(H), H, "adopt"))
            for H in spawns:
                moves.append((cfg.p_spawn / len(spawns), score(H), H, "spawn"))
            pri = np.array([m[0] for m in moves], dtype=float)
            pri /= pri.sum()
            values = np.log(pri) + np.array([m[1] for m in moves])
            mx = float(np.max(values))
            weights = np.exp(values - mx)
            total = float(np.sum(weights))
            p.logw += mx + math.log(total)
            pick = int(owner.rng.choice(len(moves), p=weights / total))
            states.append((moves[pick][2], moves[pick][3]))
    for p, (H, tag) in zip(owner.particles, states):
        p.H, p.last_move = deepcopy(H), tag
    return states


def owner(seed, hypotheses):
    def extend(H, prob):
        result = deepcopy(H)
        result[(0, 0)][9] = 4
        return result

    def shrink(H, prob):
        if len(H[(0, 0)]) < 3:
            return None
        result = deepcopy(H)
        result[(0, 0)].pop(9)
        return result

    return SimpleNamespace(
        rng=np.random.default_rng(seed),
        particles=[SimpleNamespace(H=deepcopy(H), logw=-float(i), last_move="old")
                   for i, H in enumerate(hypotheses)],
        _extend=extend, _shrink_worst=shrink,
        _moves_tracked=lambda *args: [(H, "tracked") for H in hypotheses])


@pytest.mark.parametrize("track", [0, 1, 2])
@pytest.mark.parametrize("case", ["shared", "no_consensus", "no_candidates", "empty", "no_particles"])
@pytest.mark.parametrize("seed", [0, 20260710, 8127])
def test_shared_scoring_preserves_weights_moves_and_rng(track, case, seed):
    base = {(0, 0): {1: 0, 3: -2}}
    candidates = [base, {(0, 0): {3: 10, 1: 12}}, {(0, 0): {1: 0, 3: 5}}]
    spawns = [{(0, 0): {1: 0, 3: -8}}]
    consensus = base
    hypotheses = [base, {}, {(0, 0): {1: 0, 3: -2, 9: 5}}, {}, base, base]
    if case == "no_consensus":
        consensus = None
    elif case == "no_candidates":
        consensus, candidates, spawns = None, [], []
    elif case == "empty":
        hypotheses = [{}] * 6
    elif case == "no_particles":
        hypotheses = []
    cfg = SimpleNamespace(track=track, p_survive=.5, p_extend=.1, p_shrink=.1,
                          p_release=.1, p_adopt_cons=.1, p_adopt=.3, p_spawn=.2)
    expected, observed = owner(seed, hypotheses), owner(seed, hypotheses)
    raw_empty, lp_nb, lp_b = -3., math.log(.9), math.log(.1) - 3.
    calls = []

    def score(H):
        raw = raw_empty - sum(abs(z - zmap[min(zmap)]) for zmap in H.values() for z in zmap.values()) / 10
        return float(np.logaddexp(lp_nb + raw, lp_b))

    def counted(H):
        calls.append(H)
        return score(H)

    before = deepcopy((hypotheses, consensus, candidates, spawns))
    scalar_moves(expected, cfg, consensus, candidates, spawns, score)
    RUN(observed, cfg, None, None, 17, consensus, candidates, spawns,
        counted, raw_empty, lp_nb, lp_b)
    assert [vars(p) for p in observed.particles] == [vars(p) for p in expected.particles]
    assert observed.rng.bit_generator.state == expected.rng.bit_generator.state
    assert (hypotheses, consensus, candidates, spawns) == before
    for p in observed.particles:
        for H in candidates + spawns:
            assert p.H is not H
            assert all(p.H.get(g) is not zmap for g, zmap in H.items())
    if track == 1 or not hypotheses:
        assert not calls
    else:
        # Gauge-equivalent and repeated candidate objects retain separate
        # move entries, but each shared entry is scored once for all particles.
        if spawns:
            assert sum(H is spawns[0] for H in calls) == 1
        if candidates:
            assert sum(H is candidates[-1] for H in calls) == 1


def test_shared_scores_do_not_survive_an_epoch():
    H = {(0, 0): {1: 0, 3: 2}}
    cfg = SimpleNamespace(track=0, p_survive=1., p_extend=1., p_shrink=1.,
                          p_release=1., p_adopt_cons=1., p_adopt=1., p_spawn=1.)
    outputs = []
    for shift in [0., 20.]:
        obj = owner(0, [{}] * 4)
        RUN(obj, cfg, None, None, 1, None, [H], [], lambda _: shift,
            0., 0., 0.)
        outputs.append(obj.particles[0].logw)
    assert outputs[1] - outputs[0] == 20.
