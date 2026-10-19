"""One independent original 400-epoch driver with an optional transition wake.

Every RTK body, including transition and score drain, stays inside its clock.
Primary runs have no state encoding between epochs. Independent processes
prevent one arm's extra GPU operation from warming another arm's scorer.
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_inprocess_pair_checked_field import packed_state, serialized_encode, without_rtk
from rbpf_epoch_clocks import EpochClocks, summarize
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_inprocess_pair import Offerer, Owner, paired_driver
from rbpf_move_plan_route import retaining_type
from rbpf_transition_wake import TransitionWakeScorer, transition_type

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    trial = plan["trials"][args.trial]
    assert trial["trial"] == args.trial
    for section in ("source_sha256", "external_python_sha256", "input_sha256", "gold_sha256"):
        for name, digest in plan[section].items():
            assert ".claude" not in [p.casefold() for p in Path(name).parts]
            assert sha(ROOT / name) == digest, name
    assert all(sha(ROOT / item["path"]) == item["sha256"] for item in plan["binaries"].values())
    assert sha(Path(plan["gtsam_pe"])) == plan["external_GTSAM_PE_SHA256"]
    assert sha(Path(plan["CUDA_runtime"]["path"])) == plan["CUDA_runtime"]["sha256"]
    folder = ROOT / trial["output"]
    assert not folder.exists()
    folder.mkdir()
    _pin()
    policy = gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)
    os.environ["RBPF_CLUSTER_RELIN"] = "0"
    os.environ["PPC_DATA_ROOT"] = plan["data_root"]
    sys.path.insert(0, str(ROOT / "experiments/rbpf_fgo"))
    sys.path.append(plan["tc_src"])
    import wp16_run_rbpf as driver

    gold_dir = ROOT / plan["gold_dir"]
    legacy_metadata = read(gold_dir / "record.json")["caller_metadata"]
    legacy_config = read(gold_dir / "implementation_check.json")["configs"][0]
    with gzip.open(ROOT / plan["gold_final"], "rt", encoding="utf8") as stream:
        gold_final = json.load(stream)
    legacy_states = []
    if trial["accuracy"]:
        with gzip.open(gold_dir / "states.jsonl.gz", "rt", encoding="utf8") as stream:
            legacy_states = [json.loads(line) for line in stream]
        assert len(legacy_states) == 200
    source_path = ROOT / "experiments/rbpf_fgo/wp16_run_rbpf.py"
    source = source_path.read_bytes()
    owner, clock = Owner(), EpochClocks()
    offerer = Offerer(owner, "A")
    context = dict(epoch=None, transition_epoch=None, first_score_epoch=None)
    snapshots, bounds, objects = [], [], {}
    bins = {k: ROOT / v["path"] for k, v in plan["binaries"].items()}
    argv = [str(source_path), "3", "--max-ep", "400", "--out-dir", str(folder / "trajectory"),
        "--rb-seed", "20260710", "--checkpoint-every", "0", "--log-every", "100"]
    for value in ("SEED_N_CYCLES=1", "COND_HOLD=1", "RBPF_FB=1", "RBPF_SPAWN=1", "RBPF_CUDA=0",
            "RBPF_FIX_VOTE_DD=6.0", "RBPF_FIX_VOTE_DPR=3.5", "RBPF_FB_COMMIT_MAX_DPR=1.5"):
        argv += ["--env", value]
    with TransitionWakeScorer(bins["DLL"], plan["CUDA_runtime"]["path"], mode=trial["mode"]) as scorer:
        def begin():
            assert clock.active and context["epoch"] is not None
            assert context["transition_epoch"] is None
            context["transition_epoch"] = context["epoch"]
            scorer.transition()

        def factory(original):
            import rbpf_fgo as model
            objects["originals"] = model, original, model.RbpfFgo.__init__, model.RbpfFgo.step
            observed = transition_type(original, begin)

            class Selected(observed):
                def __init__(self, *args_, **kwargs):
                    super().__init__(*args_, **kwargs)
                    selected = retaining_type(NativeFeedbackMassStep)(self.rbpf, scorer, "gpu",
                        extension=bins["CANONICAL_EXT"], parent_extension=bins["PARENT_EXT"],
                        mass_extension=bins["MASS_EXT"])
                    self.rbpf.step = selected
                    objects["rtk"], objects["controller"] = self, selected
                    assert vars(self.rbpf.cfg) == legacy_config
                    assert self.process.__func__ is original.process
            return Selected

        def offer(epoch, rtk, obs, obsb):
            assert rtk is objects["rtk"]
            return offerer.make(epoch, int(rtk.phase), rtk.process, (obs,), {"obsb": obsb}, (rtk, obs, obsb))

        function, proof = paired_driver(source, vars(driver), factory, offer, argv,
            expected_sha256=plan["source_sha256"][source_path.relative_to(ROOT).as_posix()], filename=str(source_path))
        stream = function()
        driver_start = time.perf_counter_ns()
        idle_ns = None
        try:
            for epoch in range(400):
                ticket = next(stream)
                assert ticket.epoch == epoch and ticket.owner is owner and ticket.state == "ready"
                if epoch == 0:
                    # Controlled cold-start scenario, common to all arms. It is
                    # retained in driver elapsed, never called a speed gain.
                    idle_start = time.perf_counter_ns()
                    time.sleep(plan["idle_seconds"])
                    idle_ns = time.perf_counter_ns() - idle_start
                context["epoch"] = epoch
                before = len(scorer.score_records)
                call_start = time.perf_counter_ns()
                ticket.execute(owner, clock)
                call_end = time.perf_counter_ns()
                bounds.append([call_start, call_end])
                if len(scorer.score_records) != before and context["first_score_epoch"] is None:
                    context["first_score_epoch"] = epoch
                if trial["accuracy"] and ticket.phase == 2:
                    i = epoch - 200
                    controller = objects["controller"]
                    snapshot = packed_state(controller)
                    assert without_rtk(snapshot) == without_rtk(legacy_states[i])
                    assert serialized_encode(controller.last) == legacy_metadata[i]
                    snapshots.append(snapshot)
                ticket.release()
            try:
                next(stream)
            except StopIteration as end:
                assert end.value == 0
            else:
                raise AssertionError("Driver continued beyond its complete prefix")
        finally:
            stream.close()
        driver_end = time.perf_counter_ns()
        scorer.wake.finish()
        controller = objects["controller"]
        assert controller.route_calls == 200 and len(scorer.score_records) == 197
        assert context["transition_epoch"] == 199 and context["first_score_epoch"] == 203
        final = packed_state(controller)
        assert final == gold_final and serialized_encode(controller.last) == legacy_metadata[-1]
        with gzip.open(folder / "final_state.json.gz", "wt", encoding="utf8", newline="\n") as stream:
            json.dump(final, stream, separators=(",", ":"))
            stream.write("\n")
        if trial["accuracy"]:
            assert len(snapshots) == 200
            with gzip.open(folder / "states.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
                for snapshot in snapshots:
                    stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        with np.load(gold_dir / "trajectory/run3.npz") as gold, np.load(folder / "trajectory/run3.npz") as actual:
            assert set(gold.files) == set(actual.files) and len(gold.files) == 57
            for key in gold.files:
                a, b = gold[key], actual[key]
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
        model, original, init, step = objects["originals"]
        assert model.RbpfGtsamRtkTc is original and model.RbpfFgo.__init__ is init and model.RbpfFgo.step is step
        assert (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)) == policy
        wake_record, wake_state = dict(scorer.wake.record), scorer.wake.state
        scores = list(scorer.score_records)
    values = summarize(clock.records)
    assert len(clock.records) == 400 and [v["phase_before"] for v in clock.records] == [1]*200 + [2]*200
    assert idle_ns >= plan["idle_seconds"]*10**9
    a, b = context["transition_epoch"], context["first_score_epoch"]
    latency = sum(row["wall_ns"] for row in clock.records[a:b+1])
    result = dict(verified=True, trial=trial, records=clock.records, totals=values,
        transition=context, transition_through_first_score_RTK_ns=latency,
        transition_through_first_score_wall_ns=bounds[b][1]-bounds[a][0], call_bounds_ns=bounds,
        scores=scores, wake=wake_record, wake_state=wake_state, device=scorer.device,
        driver_wall_ns=driver_end-driver_start, controlled_idle_ns=idle_ns,
        original_driver_proof=proof, all_final_and_trajectory_bits_exact=True,
        captured_post_RTK_states=len(snapshots), trajectory_arrays=57,
        numerical_methods_and_normal_GC_unchanged=True,
        environment=dict(pid=os.getpid(), parent_pid=os.getppid(), Python=sys.version,
            executable=sys.executable, NumPy=np.__version__),
        scope="One full independent original driver;all400 RTK clocks retained. Transition hook and original score drain stay inside clocks. Primary runs have no per-epoch snapshots. Controlled15sidle stays in driver elapsed. Extra CUDA operation is one-shot only;no power/keepalive/default/model change. No GPU utilization claim.")
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(verified=True, trial=args.trial, mode=trial["mode"],
        latency_ms=latency/1e6, all_RTK_ms=values["all"]["wall_ns"]/1e6)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
