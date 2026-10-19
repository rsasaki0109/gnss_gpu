"""Read only saved nested PF/RTK clocks;never promote diagnostics to speed gates."""

import ast
from copy import deepcopy
import math
from statistics import median

CLOCKS = ("wall_ns", "process_cpu_ns", "thread_cpu_ns")
ARMS = ("reference", "off", "candidate", "AA")
ORDERS = (("reference", "off", "AA", "candidate"), ("off", "candidate", "reference", "AA"),
    ("candidate", "AA", "off", "reference"), ("AA", "reference", "candidate", "off"),
    ("candidate", "AA", "off", "reference"))


def source_boundaries(field_bytes, model_bytes):
    """Bind recorded seconds to native integer timer and PF to phase2 only."""
    field = ast.parse(field_bytes)
    main = next(n for n in field.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    recorded = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "recorded")
    expected = ast.parse("""
start = time.perf_counter_ns()
output = selected(rtk, ctx)
elapsed = (time.perf_counter_ns() - start) / 1e9
calls.append(elapsed)
""").body
    def dump(n):
        return ast.dump(n, include_attributes=False)
    if list(map(dump, recorded.body[:4])) != list(map(dump, expected)):
        raise ValueError("Original nested PF timer boundary changed")
    if not isinstance(recorded.body[4], ast.Assign) or ast.unparse(recorded.body[4].targets[0]) != "boundary":
        raise ValueError("Unexpected diagnostic boundary")
    if dump(recorded.body[-1]) != dump(ast.parse("return output").body[0]):
        raise ValueError("Original selected output changed")
    # The frozen actual-route adapter removes ONLY this diagnostic suffix.
    short = deepcopy(recorded)
    removed = deepcopy(short.body[4:-1])
    del short.body[4:-1]
    restored = deepcopy(short)
    restored.body[-1:-1] = removed
    if dump(restored) != dump(recorded):
        raise ValueError("Nested clock prefix restoration failed")
    cls = next(n for n in ast.parse(model_bytes).body if isinstance(n, ast.ClassDef) and n.name == "RbpfGtsamRtkTc")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_do_ar")
    guard = ast.parse("""
if getattr(self, 'phase', 1) != 2 or self._rbpf_ctx is None:
    return
self._rbpf_last = self.rbpf.step(self, self._rbpf_ctx)
""").body
    if list(map(dump, fn.body[-2:])) != list(map(dump, guard)):
        raise ValueError("PF is no longer guarded to phase2")
    steps = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.rbpf.step"]
    if len(steps) != 1:
        raise ValueError("Unexpected additional PF dispatch")
    return dict(original_recorded_integer_ns_divided_by_1e9=True, original_timer_prefix_AST_exact=True,
        diagnostic_suffix_undo_AST_exact=True, PF_dispatch_guarded_to_phase2=True,
        per_epoch_pairing_inferred=False, phase2_aggregate_nesting_only=True)


def recover_ns(seconds):
    """Recover the original nonnegative integer ns after exact JSON float load.

    Source stores n/1e9. Under3600seconds,the rounding error after multiplying
    back is far below half a ns;round is unique. Require exact division round
    trip. This is clock encoding recovery,never a new timing or model change.
    """
    if (type(seconds) is not float or not math.isfinite(seconds) or not 0 <= seconds <= 3600
            or (seconds == 0 and math.copysign(1, seconds) < 0)):
        raise ValueError("Invalid saved PF seconds")
    n = round(seconds * 1_000_000_000)
    if n / 1_000_000_000 != seconds:
        raise ValueError("PF seconds are not an encoded integer-ns timer")
    return n


def partition(performance, record):
    rows = performance["records"]
    if len(rows) != 400 or [r["epoch"] for r in rows] != list(range(400)):
        raise ValueError("Missing or reordered RTK clocks")
    if [r["phase_before"] for r in rows] != [1] * 200 + [2] * 200:
        raise ValueError("Unexpected phase sequence")
    if any(r["completed"] is not True or any(type(r[k]) is not int or r[k] < 0 for k in CLOCKS) for r in rows):
        raise ValueError("Invalid or incomplete RTK clocks")
    if record["calls"] != performance["PF_calls"] or record["calls"] != 200 or len(record["call_seconds"]) != 200:
        raise ValueError("Missing or additional nested PF clocks")
    pf = [recover_ns(s) for s in record["call_seconds"]]
    totals = {phase: {k: sum(r[k] for r in rows if phase == "all" or r["phase_before"] == int(phase))
        for k in CLOCKS} for phase in ("all", "1", "2")}
    for phase, values in totals.items():
        if any(performance["summary"][phase][k] != v for k, v in values.items()):
            raise ValueError("Saved RTK summary differs from raw integer clocks")
    wall = totals["2"]["wall_ns"]
    if wall <= 0 or sum(pf) > wall:
        raise ValueError("Nested PF timer exceeds enclosing phase2 wall")
    driver = performance["driver_wall_ns"]
    if type(driver) is not int or driver < totals["all"]["wall_ns"]:
        raise ValueError("Driver is shorter than RTK body")
    return dict(RTK=totals, PF_calls=200, PF_call_ns=pf, PF_wall_ns=sum(pf),
        phase2_outside_PF_wall_ns=wall - sum(pf), driver_wall_ns=driver,
        driver_outside_RTK_wall_ns=driver - totals["all"]["wall_ns"],
        PF_fraction_of_phase2_wall=sum(pf) / wall,
        wall_minus_thread_cpu_ns={p: v["wall_ns"] - v["thread_cpu_ns"] for p, v in totals.items()},
        process_minus_thread_cpu_ns={p: v["process_cpu_ns"] - v["thread_cpu_ns"] for p, v in totals.items()})


def paired(reference, candidate):
    if len(reference) != 5 or len(candidate) != 5 or any(type(n) is not int or n <= 0 for n in (*reference, *candidate)):
        raise ValueError("Expected five complete positive integer clock totals")
    values = [100.0 * (a - b) / a for a, b in zip(reference, candidate, strict=True)]
    absolute = list(map(abs, values))
    return dict(paired_percent=values, median_percent=median(values), positive_rounds=sum(v > 0 for v in values),
        absolute_median_percent=median(absolute), absolute_rounds_under5=sum(v < 5 for v in absolute))


def analyze(cases):
    expected = [dict(warm=True, round=-1, variant=v) for v in ARMS]
    expected += [dict(warm=False, round=r, variant=v) for r, order in enumerate(ORDERS) for v in order]
    if len(cases) != 24 or [c["case"] for c in cases] != expected:
        raise ValueError("Missing or reordered scheduled prefixes")
    records = [dict(case=c["case"], partition=partition(c["performance"], c["record"])) for c in cases]
    by_round = {r: {} for r in range(5)}
    for row in records:
        c = row["case"]
        if not c["warm"]:
            by_round[c["round"]][c["variant"]] = row["partition"]
    series = {}
    for phase in ("all", "1", "2"):
        for clock in CLOCKS:
            series[f"RTK/{phase}/{clock}"] = {a: [by_round[r][a]["RTK"][phase][clock] for r in range(5)] for a in ARMS}
    for key in ("PF_wall_ns", "phase2_outside_PF_wall_ns", "driver_wall_ns", "driver_outside_RTK_wall_ns"):
        series[key] = {a: [by_round[r][a][key] for r in range(5)] for a in ARMS}
    contrasts = {name: {label: paired(arms[a], arms[b]) for label, a, b in (
        ("reference", "reference", "candidate"), ("matched_blocked", "off", "candidate"), ("AA", "reference", "AA"))}
        for name, arms in series.items()}
    fractions = {a: [by_round[r][a]["PF_fraction_of_phase2_wall"] for r in range(5)] for a in ARMS}
    return dict(records=records, series=series, contrasts=contrasts,
        PF_fraction_of_phase2_wall=fractions, new_estimator_runs=0, new_GPU_launches=0, new_timings=0,
        diagnostic_only=True, previous_speed_gate_recomputed_or_replaced=False,
        remainder_is_all_phase2_work_outside_nested_PF_not_GTSAM_only=True,
        CPU_clock_is_not_an_acceptance_substitute=True,
        no_clock_difference_attributed_to_scheduler_GPU_GC_or_frequency=True)
