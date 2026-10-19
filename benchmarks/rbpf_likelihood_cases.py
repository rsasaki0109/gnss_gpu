"""Literal scalar numeric oracle and seeded RBPF likelihood component inputs."""

import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALAR_SOURCE = ROOT / 'experiments/rbpf_fgo/rbpf_fgo.py'
SCALAR_SHA256 = 'c977508d6a20413f7669efb39dfb1d87d74b254ce89cf1f57ae8681d7d6b8bb6'


def make_oracle(**overrides):
    """Compile unchanged numeric methods without unrelated GTSAM imports."""
    assert hashlib.sha256(SCALAR_SOURCE.read_bytes()).hexdigest() == SCALAR_SHA256
    tree = ast.parse(SCALAR_SOURCE.read_bytes())
    canonical = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_canon_basin_id')
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RbpfFgo')
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in (
        '_pair_dd', '_basin_parent_score', '_cluster_state', '_loglik')]
    assert len(methods) == 4
    klass = ast.ClassDef(name='ScalarOracle', bases=[], keywords=[], body=deepcopy(methods), decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[deepcopy(canonical), klass], type_ignores=[]))
    scope = dict(np=np, math=math, _LOG2PI=math.log(2. * math.pi))
    exec(compile(module, str(SCALAR_SOURCE), 'exec'), scope)
    oracle = scope['ScalarOracle']()
    oracle.basin_id = scope['_canon_basin_id']
    oracle.cfg = SimpleNamespace(beta=.5, marg_uniform=1, huber_c=3., sig_group=.02,
        mix_eps=.1, mix_sigma_cp=.05, mix_sigma_pr=10., cluster_relin=0,
        cluster_parent_min=3, cluster_parent_frac=.8)
    for key, value in overrides.items():
        assert hasattr(oracle.cfg, key), key
        setattr(oracle.cfg, key, value)
    oracle.cluster_states = {}
    return oracle


def capture(oracle, hypothesis, problem):
    """Inspect original returned locals outside timing; do not rewrite math."""
    saved = {}
    previous = sys.getprofile()

    def profile(frame, event, _argument):
        if event == 'return' and frame.f_code is oracle._loglik.__func__.__code__:
            for name in ('sol', 'Lam', 'Pinv', 'A', 'b', 'sig', 'G', 'logdet_lam', 'logdet_p', 'const_ll'):
                if name in frame.f_locals:
                    value = frame.f_locals[name]
                    saved[name] = value.copy() if isinstance(value, np.ndarray) else value

    try:
        sys.setprofile(profile)
        value = oracle._loglik(hypothesis, problem, {})
    finally:
        sys.setprofile(previous)
    return value, saved


def make_case(count, *, rows=64, groups=4, seed=20261016, cluster=False,
              marg_uniform=True, huber=True, duplicates=True):
    assert rows % 2 == 0 and rows >= groups * 2
    rng = np.random.default_rng(seed + rows * 17 + groups)
    nb = rows // 2
    pairs = []
    group_keys = [(1 + g // 2, g % 2) for g in range(groups)]
    for p in range(nb):
        gi = p % groups
        ref = 1000 + gi * 100
        pairs.append((group_keys[gi], ref, ref + 1 + p // groups))
    jac = rng.normal(size=(nb, 3))
    jac /= np.maximum(np.linalg.norm(jac, axis=1)[:, None], 1e-10)
    base = rng.integers(-20, 21, nb)
    wavelength = rng.choice([.190293672798, .244210213425], nb)
    observed = base + rng.normal(0., .04, nb)
    matrix = rng.normal(size=(3, 3))
    prior = matrix @ matrix.T + np.eye(3)
    problem = SimpleNamespace(pairs=pairs, y=base + rng.normal(0., .1, nb),
        Qb=np.diag(rng.uniform(.01, 4., nb)), pos=np.array([-3957199., 3310205., 3737911.]),
        Ppos=prior, rows_J=np.vstack((jac, jac)), rows_v0=np.r_[rng.normal(0., 2., nb), np.zeros(nb)],
        rows_sig=np.r_[np.full(nb, 1.5), np.full(nb, .008)],
        rows_iscp=np.r_[np.zeros(nb, dtype=bool), np.ones(nb, dtype=bool)],
        rows_pair=np.r_[np.full(nb, -1), np.arange(nb)],
        rows_lam=np.r_[np.ones(nb), wavelength], rows_bmeas=np.r_[np.zeros(nb), observed])
    oracle = make_oracle(cluster_relin=int(cluster), marg_uniform=int(marg_uniform), huber_c=3. if huber else 0.)
    hypotheses, known = [], set()
    while len(hypotheses) < count:
        index = len(hypotheses)
        h = {}
        for p, (group, ref, sat) in enumerate(pairs):
            if index == 0 or (index > 1 and rng.random() < .25):
                continue
            zmap = h.setdefault(group, {ref: 0})
            zmap[sat] = -int(base[p] + (0 if index == 1 else rng.integers(-2, 3)))
        key = oracle.basin_id(h)
        if key in known:
            continue
        known.add(key)
        hypotheses.append(h)
        if cluster and h:
            matrix = rng.normal(size=(3, 3))
            oracle.cluster_states[key] = dict(H=deepcopy(h), pos=problem.pos + rng.normal(0., .1, 3),
                P=matrix @ matrix.T + np.eye(3) * .5)
    if duplicates:
        hypotheses = hypotheses + [deepcopy(h) for h in hypotheses[::2]] + [deepcopy(hypotheses[0])]
    return oracle, problem, hypotheses
