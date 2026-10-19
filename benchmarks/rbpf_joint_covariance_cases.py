"""Actual mixed-dimension GTSAM marginals and literal writer fixtures."""

from copy import deepcopy
from types import FunctionType, SimpleNamespace

import gtsam
import numpy as np
from cssrlib.gnss import Nav
from cssrlib.pppssr import pppos

from rbpf_joint_covariance import SOURCE


def make_case(dim=6, count=8, seed=20261017, *, held=False, subset=False):
    rng = np.random.default_rng(seed)
    key_x = gtsam.symbol("a" if seed % 2 else "z", 99)
    keys = [
        gtsam.symbol("n", int(i)) for i in rng.permutation(np.arange(count) * 13 + 7)
    ]
    values = gtsam.Values()
    values.insert(key_x, gtsam.Pose3() if dim == 6 else np.zeros(3))
    for key in keys:
        values.insert(key, 0.0)
    graph = gtsam.NonlinearFactorGraph()
    for key, width in [(key_x, dim)] + [(k, 1) for k in keys]:
        factor = gtsam.JacobianFactor(
            key, np.eye(width), np.zeros(width), gtsam.noiseModel.Unit.Create(width)
        )
        graph.add(gtsam.LinearContainerFactor(factor, values))
    for key in keys:
        factor = gtsam.JacobianFactor(
            key_x,
            rng.normal(size=(2, dim)),
            key,
            rng.normal(size=(2, 1)),
            np.zeros(2),
            gtsam.noiseModel.Unit.Create(2),
        )
        graph.add(gtsam.LinearContainerFactor(factor, values))
    module = compile(SOURCE.read_bytes(), str(SOURCE), "exec", dont_inherit=True)
    scope = dict(np=np, gtsam=gtsam)
    sorted_code = next(
        c
        for c in module.co_consts
        if getattr(c, "co_name", None) == "_sorted_amb_items"
    )
    scope["_sorted_amb_items"] = FunctionType(sorted_code, scope)
    methods = dict(IB=pppos.IB)
    for owner, name in (("GtsamRtk", "_write_back"), ("GtsamRtkTc", "_write_back_tc")):
        cls = next(c for c in module.co_consts if getattr(c, "co_name", None) == owner)
        code = next(c for c in cls.co_consts if getattr(c, "co_name", None) == name)
        methods[name] = FunctionType(code, scope, name, (None,))
    cls = type("LiteralCovarianceFixture", (), methods)
    rtk = cls()
    rtk.nav = Nav()
    rtk.nav.na, rtk.nav.nf = 3, 3
    size = 3 + 3 * int(pppos.IB.__globals__["uGNSS"].MAXSAT)
    rtk.nav.x, rtk.nav.P = np.zeros(size), np.full((size, size), -9.0)
    rtk.nav.vsat = np.zeros(
        (int(pppos.IB.__globals__["uGNSS"].MAXSAT), 3), dtype=np.int64
    )
    rtk.amb_keys = {(i + 1, 0): key for i, key in enumerate(keys)}
    rtk.smoother = None
    rtk.isam = SimpleNamespace(getFactorsUnsafe=lambda: graph)
    rtk.phase, rtk.epoch2 = 2, 20
    rtk.ecef_T_nav = gtsam.Pose3(
        gtsam.Rot3.RzRyRx(0.1, -0.2, 0.3), np.array([1.0, 2.0, 3.0])
    )
    rtk.R_enu2ecef = rtk.ecef_T_nav.rotation().matrix()
    rtk.lever_arm = np.array([0.2, -0.05, 0.01])
    rtk.cond_hold, rtk.ar_wait_new, rtk.ar_persist_bad_enable = 1, 0, 0
    rtk._persist_bad_hold, rtk._amb_init_epoch = set(), {}
    rtk._committed = {(1, 0): 4.0} if held and count else {}
    rtk.VAR_HOLDAMB = 0.001
    rtk.cond_hold_epochs = rtk.cond_hold_key_epochs = 0
    cp_amb = set(list(rtk.amb_keys)[::2]) if subset else None
    jm = gtsam.Marginals(graph, values).jointMarginalCovariance([key_x, *keys])
    active = [(s, f, k) for (s, f), k in sorted(rtk.amb_keys.items())]
    return cls, rtk, values, key_x, cp_amb, jm, active


def result(rtk):
    return dict(
        x=rtk.nav.x.copy(),
        P=rtk.nav.P.copy(),
        vsat=rtk.nav.vsat.copy(),
        last_joint_P=deepcopy(rtk.last_joint_P),
        cond_hold_epochs=rtk.cond_hold_epochs,
        cond_hold_key_epochs=rtk.cond_hold_key_epochs,
    )
