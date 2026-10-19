"""Conservative readiness ceiling for one early GPU batch; no estimator runs.

This is a rejection-only work triage, not a speed/occupancy model. Passing
the ceiling does not prove that all requests are ready or overlap profitable.
"""

import ast
import hashlib
from pathlib import Path

MOVE_SHA256 = "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea"
MINIMUM_POSSIBLE_SINGLE_BATCH_FRACTION = 0.90


def source_bound(path):
    """Bound registrations at the first completed particle, before scoring."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != MOVE_SHA256:
        raise ValueError("Frozen move source changed")
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoveBatchStep")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_moves")
    loop = next(n for n in method.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "runtime.particles")
    specific = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "h")
    held = [n for item in specific.body for n in ast.walk(item)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "add"]
    empty = [n for item in specific.orelse for n in ast.walk(item)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "add"]
    assert [ast.literal_eval(n.args[2]) for n in held] == ["survive", "extend", "shrink", "release"]
    assert [ast.literal_eval(n.args[2]) for n in empty] == ["stay"]
    assert all(not isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(specific))
    assert ast.unparse(loop.body[-1]) == "staged.append(moves)"
    assert [ast.unparse(n).splitlines()[0] for n in loop.body[-4:-1]] == [
        "if H_cons is not None:", "for h in cand_H:", "for h in spawn_H:"]
    index = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "index")
    assert "if cache.get(key) is None:" in ast.unparse(index)
    assert "pending.append(len(keys) - 1)" in ast.unparse(index)
    requested = next(n for n in method.body if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "requested")
    assert method.body.index(requested) > method.body.index(loop)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "score" for n in ast.walk(loop))
    return dict(source_sha256=MOVE_SHA256, held_prefix_maximum=len(held),
        empty_prefix_maximum=len(empty), prefix_maximum=max(len(held), len(empty)),
        common_after_first_prefix=True, score_after_all_particles=True,
        source_AST_sha256=hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest())


def ceiling(*, particles, unique_moves, existing_cache_hits, common_records, prefix_maximum=4):
    """Use an optimistic bound: every first record could be a fresh request.

    Existing cache hits and duplicate canonical keys can only LOWER actual
    first-particle readiness. We deliberately do not subtract them here, so a
    negative result rules out a complete early single batch even optimistically.
    No per-call timing or GPU work is predicted from the counts.
    """
    values = (particles, unique_moves, existing_cache_hits, common_records, prefix_maximum)
    if any(type(v) is not int or v < 0 for v in values) or particles == 0:
        raise ValueError("Expected nonnegative integer work counts and positive particles")
    if existing_cache_hits > unique_moves or prefix_maximum == 0:
        raise ValueError("Inconsistent work counts")
    pending = unique_moves - existing_cache_hits
    ready_ceiling = min(pending, common_records + prefix_maximum)
    return dict(particles=particles, pending=pending, common_records=common_records,
        ready_ceiling=ready_ceiling, definitely_later_minimum=pending - ready_ceiling,
        single_batch_still_possible=pending <= common_records + prefix_maximum,
        ready_fraction_ceiling=ready_ceiling / pending if pending else None)


def summarize(rows):
    if not rows or any(row["pending"] <= 0 for row in rows):
        raise ValueError("Only nonempty actual score batches enter this triage")
    possible = sum(row["single_batch_still_possible"] for row in rows)
    total = sum(row["pending"] for row in rows)
    maximum_ready = sum(row["ready_ceiling"] for row in rows)
    return dict(valid_batches=len(rows), total_pending=total,
        optimistic_ready_total=maximum_ready, definitely_later_minimum=total - maximum_ready,
        optimistic_ready_fraction=maximum_ready / total,
        possible_single_batches=possible, definitely_incomplete_batches=len(rows) - possible,
        possible_single_batch_fraction=possible / len(rows),
        required_possible_single_batch_fraction=MINIMUM_POSSIBLE_SINGLE_BATCH_FRACTION,
        proceed_to_early_single_batch_prototype=possible / len(rows) >= MINIMUM_POSSIBLE_SINGLE_BATCH_FRACTION,
        actual_early_launches_observed=0, new_timings=0,
        scope="Optimistic rejection ceiling only. Possibility is not readiness,measured overlap,speed,or a hard time bound. Existing late batch remains unchanged. A split-batch or complete-discovery design requires its own work/accuracy/performance study.")
