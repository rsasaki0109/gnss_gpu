"""Schedule each original prior array/sum/divide/log during the GPU batch."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType, MemberDescriptorType

import numpy as np

from rbpf_discovery_pipeline import DiscoveryEngine, DiscoveryPipelineStep

_GENERATOR = np.random.Generator
_BITS = tuple(getattr(np.random, n) for n in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"))
_NUMPY = {n: getattr(np, n) for n in ("array", "log", "max", "exp", "sum", "geterr")}
_PRIORS = ("p_survive", "p_extend", "p_shrink", "p_release", "p_adopt_cons", "p_adopt", "p_spawn")


def _compile():
    path = Path(__file__).with_name("rbpf_move_batch.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea"
    cls = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == "MoveBatchStep")
    node = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_moves"))
    original = deepcopy(node)
    index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.FunctionDef) and n.name == "index")
    stage = deepcopy(node.body[index:index + 3])
    node.body[index:index + 3] = ast.parse("held = _rbpf_stage(self,runtime,cfg,prob,H_cons,cand_H,spawn_H,cache,keys,hypotheses,lookup,pending,staged)").body
    score = next(n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.scorer.score")
    score.func = ast.Name("_pipeline_score", ast.Load())
    score.args.insert(0, ast.Name("self", ast.Load()))
    selection_index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.For) and ast.unparse(n.iter) == "zip(runtime.particles, staged)")
    selection = node.body[selection_index]
    assert len(selection.body) == 10
    assert ast.unparse(selection.body[1]) == "pri = np.array([mv[0] for mv in moves], dtype=float)"
    assert ast.unparse(selection.body[2]) == "pri /= pri.sum()"
    lam = selection.body[3]
    assert ast.unparse(lam.value.left) == "np.log(pri)"
    prep = ast.parse("def _prepare(records): moves = records").body[0]
    prep.body += deepcopy(selection.body[1:3]) + [ast.Return(deepcopy(lam.value.left))]
    prepared = deepcopy(selection)
    prepared.target = ast.parse("_particle_index, (particle, records) = None").body[0].targets[0]
    prepared.iter = ast.parse("enumerate(zip(runtime.particles, staged))", mode="eval").body
    prepared.body = deepcopy(selection.body[3:])
    prepared.body[0].value.left = ast.parse("_pipeline_priors[_particle_index]", mode="eval").body
    prepared.body[0].value.right = ast.parse("np.array([likelihood[i] for prior, i, h, tag in records])", mode="eval").body
    prepared.body[-2] = ast.parse("pick = int(runtime.rng.choice(len(records), p=wsum / tot))").body[0]
    prepared.body[-1] = ast.parse("new_states.append((records[pick][2], records[pick][3]))").body[0]
    node.body[selection_index] = ast.If(ast.parse("_pipeline_priors is None", mode="eval").body,
        body=[deepcopy(selection)], orelse=[prepared])
    restored = deepcopy(node)
    restored.body[selection_index] = deepcopy(selection)
    call = next(n for n in ast.walk(restored) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_pipeline_score")
    call.func = ast.parse("self.scorer.score", mode="eval").body
    call.args.pop(0)
    restored.body[index:index + 1] = stage
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    scope = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node, prep], type_ignores=[])), __file__, "exec", dont_inherit=True), scope)
    return scope["_moves"].__code__, scope["_prepare"]


_MOVES, _PREPARE = _compile()
AST_PROOF = dict(original_moves_AST_restored_exact=True, each_original_prior_operation_preserved=True,
    probability_or_normalizer_cache=False, original_exp_sum_logw_choice_order_preserved=True)


class PriorEngine(DiscoveryEngine):
    def __init__(self, *args):
        super().__init__(*args)
        self.prior = dict(prepared_particles=0, prepared_records=0, literal_weight_calls=0)
        cls = self.particle_type
        self.descriptors = {n: cls.__dict__.get(n) for n in ("H", "logw", "last_move")}

    def _ready(self, staged):
        rt, cfg = self.owner.runtime, self.owner.runtime.cfg
        if (type(rt).__dict__.get("rng") is not None or "rng" not in vars(rt)
                or type(vars(rt)["rng"]) is not _GENERATOR
                or type(vars(rt)["rng"].bit_generator) not in _BITS
                or self.particle_type.__setattr__ is not object.__setattr__
                or any(type(v) is not MemberDescriptorType or self.particle_type.__dict__.get(n) is not v
                    for n, v in self.descriptors.items())
                or any(getattr(np, n) is not f for n, f in _NUMPY.items())
                or any(not 1e-100 <= vars(cfg)[n] <= 1e100 for n in _PRIORS)
                or any(not 1 <= len(moves) <= 4096 for moves in staged)):
            return False
        return True

    def stage(self, *args):
        scope = self.owner._staging_clones[0].__globals__
        scope["_pipeline_priors"] = None
        before = self.stats["planned"]
        held = super().stage(*args)
        staged = args[-1]
        if self.stats["planned"] > before and self._ready(staged):
            scope["_pipeline_priors"] = [_PREPARE(records) for records in staged]
            self.prior["prepared_particles"] += len(staged)
            self.prior["prepared_records"] += sum(map(len, staged))
        else:
            self.prior["literal_weight_calls"] += 1
        return held


class PriorPipelineStep(DiscoveryPipelineStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self._staging_clones[0]
        self._staging_clones[0] = FunctionType(_MOVES, old.__globals__, old.__name__, old.__defaults__, old.__closure__)
        self.staging_engine = PriorEngine(self, old.__globals__, True)

    def __call__(self, rtk, ctx):
        engine, previous = self.staging_engine, self.last
        before = dict(engine.prior)
        try:
            return super().__call__(rtk, ctx)
        finally:
            self._staging_clones[0].__globals__.pop("_pipeline_priors", None)
            if self.last is not previous and self.last is not None and "batch" in self.last:
                self.last["prior_pipeline"] = {k: v - before[k] for k, v in engine.prior.items()}
