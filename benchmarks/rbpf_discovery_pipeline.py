"""Discover the complete score batch,begin GPU work,then materialize records."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import FunctionType

from rbpf_async_score import AsyncScorer
from rbpf_move_batch import MoveBatchStep
from rbpf_move_plan_checked import MovePlanEngine, MovePlanStep
from rbpf_native_staging import _RAW_STAGE


def _moves_code():
    path = Path(__file__).with_name("rbpf_move_batch.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea":
        raise ValueError("Frozen move source changed")
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoveBatchStep")
    node = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_moves"))
    original = deepcopy(node)
    index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.FunctionDef) and n.name == "index")
    stage = deepcopy(node.body[index:index + 3])
    assert ast.unparse(stage[1]) == "held = 0" and ast.unparse(stage[2].iter) == "runtime.particles"
    node.body[index:index + 3] = ast.parse("held = _rbpf_stage(self,runtime,cfg,prob,H_cons,cand_H,spawn_H,cache,keys,hypotheses,lookup,pending,staged)").body
    score = next(n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.scorer.score")
    oldargs = deepcopy(score.args)
    score.func = ast.Name("_pipeline_score", ast.Load())
    score.args.insert(0, ast.Name("self", ast.Load()))
    restored = deepcopy(node)
    call = next(n for n in ast.walk(restored) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_pipeline_score")
    call.func = ast.parse("self.scorer.score", mode="eval").body
    call.args = oldargs
    restored.body[index:index + 1] = stage
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), __file__, "exec", dont_inherit=True)
    return next(c for c in code.co_consts if getattr(c, "co_name", None) == "_moves")


_MOVES = _moves_code()
AST_PROOF = dict(original_moves_AST_restored_exact=True, original_weights_choice_and_tail_unchanged=True)


class DiscoveryEngine(MovePlanEngine):
    """No new numerical cache. Plans end before choice/commit/graph callbacks.

    All first-seen indices are discovered in original particle order,with the
    common suffix after the first prefix. Only record materialization moves
    after begin. Original per-particle weight arithmetic/choice is untouched.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.job = self.expected = None
        self.pipeline = dict(discoveries=0, complete_launches=0, finishes=0, drains=0,
            hypotheses_submitted=0, records_after_begin=0)
        self.pure = {n: (f, f.__code__) for n, f in self.owner._source_functions.items()}

    def _early(self, context):
        return (context.mode == "gpu" and type(context.scorer) is AsyncScorer and context.scorer.safe()
            and all(self.owner._source_functions.get(n) is f and f.__code__ is code
                for n, (f, code) in self.pure.items()))

    def stage(self, context, runtime, cfg, prob, H_cons, cand_H, spawn_H,
              cache, keys, hypotheses, lookup, pending, staged):
        self.owner._staging_clones[0].__globals__["_pipeline_score"] = self.score_result
        self.stats["calls"] += 1
        plan = self._eligible(context, runtime, cfg, prob, H_cons, cand_H, spawn_H, cache)
        if plan is None:
            self.stats["fallback"] += 1
            return self.original(context, runtime, cfg, prob, H_cons, cand_H, spawn_H,
                cache, keys, hypotheses, lookup, pending, staged)
        self.stats["planned"] += 1
        prefixes, common = {}, None
        held = executed = prefix_replays = common_replayed = 0

        def index(hypothesis):
            nonlocal executed
            executed += 1
            key = context.canonical(hypothesis)
            if key not in lookup:
                lookup[key] = len(keys)
                keys.append(key)
                hypotheses.append(hypothesis)
                if cache.get(key) is None:
                    pending.append(len(keys) - 1)
            return lookup[key]

        for h, ordered in plan:
            moves = []
            def add(prior, hypothesis, tag):
                moves.append((prior, index(hypothesis), hypothesis, tag))
            held += int(bool(h))
            if ordered not in prefixes:
                if h:
                    add(cfg.p_survive, h, "survive")
                    extended = runtime._extend(h, prob)
                    if extended is not None:
                        add(cfg.p_extend, extended, "extend")
                    shrunk = runtime._shrink_worst(h, prob)
                    if shrunk is not None:
                        add(cfg.p_shrink, shrunk, "shrink")
                    add(cfg.p_release, {}, "release")
                else:
                    add(cfg.p_survive + cfg.p_extend + cfg.p_shrink + cfg.p_release, h, "stay")
                prefixes[ordered] = tuple(moves)
            else:
                prefix_replays += 1
            if common is None:
                start = len(moves)
                if H_cons is not None:
                    add(cfg.p_adopt_cons, H_cons, "adopt_cons")
                for candidate in cand_H:
                    add(cfg.p_adopt / len(cand_H), candidate, "adopt")
                for spawned in spawn_H:
                    add(cfg.p_spawn / len(spawn_H), spawned, "spawn")
                common = tuple(moves[start:])
            else:
                common_replayed += len(common)
        self.pipeline["discoveries"] += 1
        early = self._early(context)
        if early:
            self.expected = tuple(hypotheses[i] for i in pending)
            self.job = context.scorer.begin(list(self.expected), prob, context.oracle, mode="gpu")
            self.pipeline["complete_launches"] += 1
            self.pipeline["hypotheses_submitted"] += len(self.expected)
        first = set()
        for h, ordered in plan:
            moves = []
            for prior, i, original_h, tag in prefixes[ordered]:
                selected = h if tag in ("survive", "stay") else (
                    (original_h if ordered not in first else {}) if tag == "release" else original_h)
                moves.append((prior, i, selected, tag))
            first.add(ordered)
            for prior, i, hypothesis, tag in common:
                moves.append((prior, i, hypothesis, tag))
            staged.append(moves)
        logical = sum(map(len, staged))
        self.pipeline["records_after_begin"] += logical if early else 0
        for k, value in dict(particles=len(plan), logical_registrations=logical,
            executed_registrations=executed, avoided_registrations=logical - executed,
            prefix_plans=len(prefixes), prefix_replays=prefix_replays,
            common_records=len(common) if common else 0, common_replayed_records=common_replayed).items():
            self.stats[k] += value
        return held

    def score_result(self, context, requested, prob, oracle, mode="gpu"):
        if self.job is None:
            return context.scorer.score(requested, prob, oracle, mode=mode)
        assert mode == "gpu" and len(requested) == len(self.expected)
        assert all(a is b for a, b in zip(requested, self.expected, strict=True))
        job, self.job, self.expected = self.job, None, None
        self.pipeline["finishes"] += 1
        return context.scorer.finish(job)

    def drain(self):
        if self.job is not None:
            job, self.job, self.expected = self.job, None, None
            self.pipeline["drains"] += 1
            return job.owner.cancel(job)
        return 0


class DiscoveryPipelineStep(MovePlanStep):
    def __init__(self, *args, **kwargs):
        if "planned" in kwargs:
            raise TypeError("DiscoveryPipelineStep always uses complete discovery")
        super().__init__(*args, **kwargs)
        old = self._staging_clones[0]
        self._staging_clones[0] = FunctionType(_MOVES, old.__globals__, old.__name__, old.__defaults__, old.__closure__)
        self.staging_engine = DiscoveryEngine(self, old.__globals__, True)

    def __call__(self, rtk, ctx):
        engine = self.staging_engine
        before = dict(engine.pipeline)
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            error = sys.exc_info()[0]
            code = engine.drain()
            self._staging_clones[0].__globals__.pop("_pipeline_score", None)
            if self.last is not previous and self.last is not None and "batch" in self.last:
                self.last["discovery_pipeline"] = {k: v - before[k] for k, v in engine.pipeline.items()}
            if code and error is None:
                raise RuntimeError("GPU job drain failed")
