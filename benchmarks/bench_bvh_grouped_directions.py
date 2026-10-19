"""Exact grouped directions: original/off/on complete calls with calibrated controls."""

import argparse
from contextlib import ExitStack
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info

from bench_bvh_multipath_compare import _build, _load_extension, _rays
from bench_raytrace import _make_building_mesh
from bvh_tiled_geometry_metrics import MODES, evaluate

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    dict(triangles=12000, epochs=128, satellites=12, primary=True),
    dict(triangles=120000, epochs=32, satellites=32, primary=True),
    dict(triangles=12000, epochs=1, satellites=12, primary=False),
    dict(triangles=4096, epochs=16, satellites=8, primary=False),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(arrays):
    return [
        dict(dtype=a.dtype.str, shape=list(a.shape), hex=a.tobytes().hex())
        for a in arrays
    ]


def digest(arrays):
    return hashlib.sha256(
        json.dumps(encode(arrays), separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_plan.read_text(encoding="utf8"))
    for section in (
        "source_sha256",
        "external_python_sha256",
        "input_sha256",
        "gold_sha256",
        "saved_pair_sha256",
    ):
        assert all(sha(ROOT / n) == d for n, d in frozen[section].items()), section
    assert all(
        sha(ROOT / b["path"]) == b["sha256"] for b in frozen["binaries"].values()
    )
    assert gc.isenabled()
    process = psutil.Process()
    process.cpu_affinity([max(process.cpu_affinity())])
    assert process.nice() == psutil.NORMAL_PRIORITY_CLASS
    assert all(
        os.environ.get(n) == "1"
        for n in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    old = _load_extension(
        ROOT / frozen["binaries"]["BVH_BASE"]["path"], "layout_screen_old._bvh"
    )
    new = _load_extension(
        ROOT / frozen["binaries"]["BVH_GROUPED"]["path"], "layout_screen_new._bvh"
    )
    inputs = []
    folder = args.output.with_suffix("").with_name(args.output.stem + "_inputs")
    assert not folder.exists() and not args.output.exists()
    folder.mkdir()
    input_proof = {}
    for ci, condition in enumerate(CONDITIONS):
        count = condition["triangles"]
        nodes, mesh = _build(old, _make_building_mesh((count + 11) // 12)[:count])
        rx, sat = _rays(condition["epochs"], condition["satellites"], seed=271 + ci)
        arrays = (nodes, mesh, rx, sat)
        for array in arrays:
            array.flags.writeable = False
        path = folder / f"condition{ci}.npz"
        np.savez(path, nodes=nodes, mesh=mesh, rx=rx, sat=sat)
        input_proof[str(ci)] = dict(
            path=path.relative_to(ROOT).as_posix(),
            sha256=sha(path),
            arrays_sha256=digest(arrays),
        )
        inputs.append(arrays)
    measurement = args.output.with_suffix(".measurement.plan.json")
    assert not measurement.exists()
    measurement.write_text(
        json.dumps(
            dict(
                frozen_plan_sha256=sha(args.frozen_plan),
                conditions=CONDITIONS,
                modes=MODES,
                timed_rounds=5,
                timed_groups=3,
                timed_repetitions=2,
                cold_rounds=5,
                warm_calls_per_mode=3,
                setup_calls_per_condition=3,
                input_proof=input_proof,
                contexts_per_condition=2,
                original_AA_same_scene=True,
                off_on_and_off_AA_same_scene=True,
                environment=dict(
                    pid=os.getpid(),
                    ppid=os.getppid(),
                    python=sys.version,
                    numpy=np.__version__,
                    affinity=process.cpu_affinity(),
                    priority=int(process.nice()),
                    GC_enabled=gc.isenabled(),
                    thread_pools=threadpool_info(),
                ),
                acceptance="Each bulk warm versus original AND samebinaryoff >=10percent median and4of5positive. Both AA absolute median<5percent AND4of5absrounds<5. Both small fallback warm and every cold case versus both controls nonregression<=5percent median and4of5rounds. Cold AA same strict controls. ALL4cases mustpass before public/urban stage. Keep all raw/setup/initial/negative times,no retime.",
                scope="Whole synchronous trace_batch native validation/H2D/LOS/reflection/sync/D2H/returned arrays. Warm includes dispatch/cache checks;selector call before timer. Cold includes constructor/selector/firsttrace/close. CPU mesh/BVH build,hash/serialization excluded. Shared candidate context constructedwithrawv0capture can retainoldmirror+directioncaches;freshonsetup/cold includesv0snapshot,firstGPU-normalD2H/CPUrawkeygroup/mapH2D/cachepreparation. Freshoriginal/off hasno capture. Per-callpreparealwaysusescurrentrays. Reportsactualcache bytes andgroupcount.12common CUDA/JIT setup calls retained but not primary,then60warm+600timed+100cold,all output bits match. CPU1/BLAS1/normalGC/priority,nonexclusive. Native ray API only,no PF/FGO/urban speed or GPUutilization claim.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    rawpath = args.output.with_suffix(".calls.jsonl.gz")
    statespath = args.output.with_suffix(".states.jsonl.gz")
    assert not rawpath.exists() and not statespath.exists()
    references, all_records, results = {}, [], {}
    with (
        gzip.open(rawpath, "wt", encoding="utf8", newline="\n") as raw,
        gzip.open(statespath, "wt", encoding="utf8", newline="\n") as states,
    ):

        def save(row, output, snapshot=False):
            ci = row["condition"]
            hashed = digest(output)
            if ci not in references:
                assert row["phase"] == "setup" and row["mode"] == "original"
                references[ci] = hashed
            assert hashed == references[ci]
            row.update(
                output_sha256=hashed, input_sha256=input_proof[str(ci)]["arrays_sha256"]
            )
            raw.write(json.dumps(row, separators=(",", ":")) + "\n")
            all_records.append(row)
            if snapshot:
                states.write(
                    json.dumps(
                        dict(
                            condition=ci,
                            mode=row["mode"],
                            repetition=row["repetition"],
                            arrays=encode(output),
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        def construct(mode, nodes, mesh):
            scene = (old.DeviceBVH(nodes, mesh) if mode.startswith("original") else
                     new.DeviceBVH(nodes, mesh, grouped_directions=(mode == "on")))
            if not mode.startswith("original"):
                scene.set_direction_cache_enabled(mode == "on")
            return scene

        # Initial CUDA/JIT/setup is explicitly shared and retained before any primary clock.
        for ci, (nodes, mesh, rx, sat) in enumerate(inputs):
            for mode in ("original", "off", "on"):
                start = time.perf_counter_ns()
                scene = construct(mode, nodes, mesh)
                try:
                    output = scene.trace_batch(rx, sat)
                    ns = time.perf_counter_ns() - start
                    peak = scene.allocated_bytes
                    tiled = (
                        scene.direction_cache_active if mode != "original" else False
                    )
                    geometry = scene.direction_cache_bytes if mode != "original" else 0
                    assert tiled == (mode == "on" and CONDITIONS[ci]["primary"])
                    assert (geometry > 0) == tiled
                    save(
                        dict(
                            phase="setup",
                            condition=ci,
                            mode=mode,
                            ns=ns,
                            allocated_bytes=peak,
                            cache_active=tiled,
                            cache_bytes=geometry,
                            group_count=(scene.direction_group_count if mode != "original" else 0),
                            speed_evidence=False,
                        ),
                        output,
                    )
                finally:
                    scene.close()
                assert scene.allocated_bytes == 0
        for ci, condition in enumerate(CONDITIONS):
            print(json.dumps(dict(condition_start=ci)), flush=True)
            nodes, mesh, rx, sat = inputs[ci]
            records = []
            with ExitStack() as owners:
                original = old.DeviceBVH(nodes, mesh)
                owners.callback(original.close)
                candidate = new.DeviceBVH(nodes, mesh, grouped_directions=True)
                owners.callback(candidate.close)

                def call(mode):
                    scene = original if mode.startswith("original") else candidate
                    if scene is candidate:
                        scene.set_direction_cache_enabled(mode == "on")
                    start = time.perf_counter_ns()
                    output = scene.trace_batch(rx, sat)
                    ns = time.perf_counter_ns() - start
                    active = (
                        scene.direction_cache_active if scene is candidate else False
                    )
                    assert active == (mode == "on" and condition["primary"])
                    if not condition["primary"] and scene is candidate:
                        assert scene.direction_cache_bytes == 0
                    return ns, output, scene, active

                for mode in MODES:
                    for repetition in range(3):
                        ns, output, scene, active = call(mode)
                        save(
                            dict(
                                phase="warm",
                                condition=ci,
                                mode=mode,
                                repetition=repetition,
                                ns=ns,
                                allocated_bytes=scene.allocated_bytes,
                                cache_active=active,
                            ),
                            output,
                            snapshot=True,
                        )
                stable = dict(
                    original=original.allocated_bytes,
                    candidate=candidate.allocated_bytes,
                )
                for rd in range(5):
                    for group in range(3):
                        shift = (ci + rd * 3 + group) % len(MODES)
                        order = MODES[shift:] + MODES[:shift]
                        if (ci + rd + group) % 2:
                            order = order[::-1]
                        for mode in order:
                            for repetition in range(2):
                                ns, output, scene, active = call(mode)
                                assert (
                                    scene.allocated_bytes
                                    == stable[
                                        "original" if scene is original else "candidate"
                                    ]
                                )
                                row = dict(
                                    phase="timed",
                                    condition=ci,
                                    round=rd,
                                    group=group,
                                    mode=mode,
                                    repetition=repetition,
                                    order=order,
                                    ns=ns,
                                    cache_active=active,
                                    allocated_bytes=scene.allocated_bytes,
                                    cache_bytes=(
                                        scene.direction_cache_bytes
                                        if scene is candidate
                                        else 0
                                    ),
                                )
                                save(row, output)
                                records.append(row)
            assert original.allocated_bytes == candidate.allocated_bytes == 0
            for rd in range(5):
                shift = (ci + rd) % len(MODES)
                order = MODES[shift:] + MODES[:shift]
                if (ci + rd) % 2:
                    order = order[::-1]
                for mode in order:
                    start = time.perf_counter_ns()
                    scene = construct(mode, nodes, mesh)
                    try:
                        output = scene.trace_batch(rx, sat)
                    finally:
                        scene.close()
                    ns = time.perf_counter_ns() - start
                    assert scene.allocated_bytes == 0
                    row = dict(
                        phase="cold",
                        condition=ci,
                        round=rd,
                        mode=mode,
                        order=order,
                        ns=ns,
                        closed_bytes=0,
                    )
                    save(row, output)
                    records.append(row)
            assert digest(inputs[ci]) == input_proof[str(ci)]["arrays_sha256"]
            results[str(ci)] = evaluate(records, condition["primary"])
            raw.flush()
            states.flush()
            print(
                json.dumps(dict(condition_complete=ci, summary=results[str(ci)])),
                flush=True,
            )
    assert len(all_records) == 772
    assert {
        p: sum(r["phase"] == p for r in all_records)
        for p in ("setup", "warm", "timed", "cold")
    } == dict(setup=12, warm=60, timed=600, cold=100)
    assert all(sha(ROOT / n) == d for n, d in frozen["source_sha256"].items())
    assert all(
        sha(ROOT / b["path"]) == b["sha256"] for b in frozen["binaries"].values()
    )
    args.output.write_text(
        json.dumps(
            dict(
                measurement_plan_sha256=sha(measurement),
                frozen_plan_sha256=sha(args.frozen_plan),
                calls_sha256=sha(rawpath),
                states_sha256=sha(statespath),
                counts=dict(setup=12, warm=60, timed=600, cold=100),
                all_output_dtype_shape_bits_equal=True,
                all_inputs_and_sources_and_binaries_unchanged=True,
                results=results,
                native_screen_pass=all(v["passed"] for v in results.values()),
                public_or_urban_speed_established=False,
                default_changed=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
