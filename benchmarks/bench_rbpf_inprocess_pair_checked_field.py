"""First A/A calibration of two independent,interleaved original RTK drivers.

This is a changed measurement protocol,not continuous driver throughput or
a remeasurement of a completed optimization. Only the frozen D236 caller
runs in both streams; primary pairs contain no per-epoch state encoding.
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_move_field import encode as raw_encode
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_inprocess_pair import LABELS, Offerer, Owner, drive, evaluate, paired_driver
from rbpf_move_cases import state
from rbpf_move_plan_route import retaining_type
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]



def serialized_encode(value):
    """Use the same JSON container form as legacy snapshots;retain all tags/bits."""
    return json.loads(json.dumps(raw_encode(value), separators=(",", ":")))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, value):
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def packed_state(controller):
    rtk, output = controller.route_last
    boundary = SimpleNamespace(_committed=rtk._committed,
        _rbpf_released=getattr(rtk, "_rbpf_released", set()),
        nav=SimpleNamespace(x=rtk.nav.x, P=rtk.nav.P))
    return serialized_encode(state(controller.runtime, boundary, output))


def without_rtk(value):
    # Legacy snapshots were taken at PF return. Our snapshots are after the
    # complete RTK body; only that explicitly different root boundary is omitted.
    assert set(value) == {"mapping"}
    assert sum(k == "rtk" for k, _ in value["mapping"]) == 1
    return {"mapping": [[k, v] for k, v in value["mapping"] if k != "rtk"]}


def cache_manifest(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
        if p.is_file() and ".claude" not in [v.casefold() for v in p.parts]}


def independent(tickets):
    left, right = [tickets[label].context for label in LABELS]
    a, b = left[0], right[0]
    for x, y in ((a, b), (a.nav, b.nav), (a.rbpf, b.rbpf),
            (a.rbpf.rng, b.rbpf.rng), (left[1], right[1]), (left[2], right[2])):
        assert x is not y
    for name in ("smoother", "isam"):
        x, y = getattr(a, name, None), getattr(b, name, None)
        if x is not None or y is not None:
            assert x is not None and y is not None and x is not y
    for name in ("x", "P"):
        assert not np.shares_memory(getattr(a.nav, name), getattr(b.nav, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    for section in ("source_sha256", "external_python_sha256", "input_sha256", "gold_sha256"):
        for name, digest in plan[section].items():
            assert ".claude" not in [p.casefold() for p in Path(name).parts]
            assert sha(ROOT / name) == digest, name
    for item in plan["binaries"].values():
        assert sha(ROOT / item["path"]) == item["sha256"]
    assert sha(Path(plan["gtsam_pe"])) == plan["external_GTSAM_PE_SHA256"]
    output = ROOT / plan["output"]
    assert not output.exists()
    output.mkdir()
    _pin()
    policy = gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)
    os.environ["RBPF_CLUSTER_RELIN"] = "0"
    os.environ["PPC_DATA_ROOT"] = plan["data_root"]
    sys.path.insert(0, str(ROOT / "experiments/rbpf_fgo"))
    sys.path.append(plan["tc_src"])
    import wp16_run_rbpf as driver

    source_path = ROOT / "experiments/rbpf_fgo/wp16_run_rbpf.py"
    source = source_path.read_bytes()
    expected = plan["source_sha256"][source_path.relative_to(ROOT).as_posix()]
    gold_dir = ROOT / plan["gold_dir"]
    with gzip.open(gold_dir / "states.jsonl.gz", "rt", encoding="utf8") as stream:
        legacy_states = [json.loads(line) for line in stream]
    legacy_metadata = read(gold_dir / "record.json")["caller_metadata"]
    legacy_config = read(gold_dir / "implementation_check.json")["configs"]
    with gzip.open(ROOT / plan["gold_final"], "rt", encoding="utf8") as stream:
        gold_final = json.load(stream)
    assert len(legacy_states) == len(legacy_metadata) == 200 and len(legacy_config) == 1
    binaries = {k: ROOT / v["path"] for k, v in plan["binaries"].items()}
    env = ["SEED_N_CYCLES=1", "COND_HOLD=1", "RBPF_FB=1", "RBPF_SPAWN=1", "RBPF_CUDA=0",
        "RBPF_FIX_VOTE_DD=6.0", "RBPF_FIX_VOTE_DPR=3.5", "RBPF_FB_COMMIT_MAX_DPR=1.5"]
    original_argv = sys.argv
    sys.argv = [str(source_path), "3"]
    pairs, warm_cache = [], None
    try:
        with RegisterLUScorer(binaries["DLL"]) as scorer:
            for pair_index in range(6):
                folder = output / f"pair{pair_index}"
                folder.mkdir()
                owner = Owner()
                offers = {label: Offerer(owner, label) for label in LABELS}
                instances, controllers, configs, proofs, streams = {}, {}, {}, {}, {}
                original_classes = []
                before = cache_manifest(Path(os.environ["NUMBA_CACHE_DIR"]))
                snapshots = {label: [] for label in LABELS}

                def factory_for(label):
                    def factory(original):
                        import rbpf_fgo as model
                        original_classes.append((model, original, model.RbpfFgo.__init__, model.RbpfFgo.step))

                        class Selected(original):
                            def __init__(self, *args_, **kwargs):
                                super().__init__(*args_, **kwargs)
                                assert label not in instances
                                selected = retaining_type(NativeFeedbackMassStep)(self.rbpf, scorer, "gpu",
                                    extension=binaries["CANONICAL_EXT"], parent_extension=binaries["PARENT_EXT"],
                                    mass_extension=binaries["MASS_EXT"])
                                self.rbpf.step = selected
                                instances[label], controllers[label] = self, selected
                                configs[label] = vars(self.rbpf.cfg).copy()
                                assert self.process.__func__ is original.process
                        return Selected
                    return factory

                def offer_for(label):
                    def offer(epoch, rtk, obs, obsb):
                        assert instances[label] is rtk
                        ticket = offers[label].make(epoch, int(rtk.phase), rtk.process,
                            (obs,), {"obsb": obsb}, (rtk, obs, obsb))
                        other = offers[LABELS[1 - LABELS.index(label)]].last
                        if epoch in (0, 200) and other is not None and other.epoch == epoch:
                            independent({label: ticket, other.label: other})
                        return ticket
                    return offer

                def capture(epoch, tickets, clocks):
                    assert all(not c.active for c in clocks.values()) and not owner.active
                    if tickets["A"].phase != 2:
                        assert all(c.route_calls == 0 for c in controllers.values())
                        return
                    i = epoch - 200
                    current = {}
                    for label in LABELS:
                        controller = controllers[label]
                        assert controller.route_calls == i + 1
                        current[label] = packed_state(controller)
                        assert without_rtk(current[label]) == without_rtk(legacy_states[i])
                        assert serialized_encode(controller.last) == legacy_metadata[i]
                        snapshots[label].append(current[label])
                    assert current["A"] == current["AA"]

                for label in LABELS:
                    argv = [str(source_path), "3", "--max-ep", "400", "--out-dir",
                        str(folder / label / "trajectory"), "--rb-seed", "20260710",
                        "--checkpoint-every", "0", "--log-every", "100"]
                    for value in env:
                        argv += ["--env", value]
                    function, proofs[label] = paired_driver(source, vars(driver), factory_for(label),
                        offer_for(label), argv, expected_sha256=expected, filename=str(source_path))
                    streams[label] = function()
                result = drive(streams, owner, pair_index=pair_index, epochs=400,
                    after=capture if pair_index == 0 else None)
                for label in LABELS:
                    controller = controllers[label]
                    assert controller.route_calls == 200 and configs[label] == legacy_config[0]
                    final = packed_state(controller)
                    assert final == gold_final and serialized_encode(controller.last) == legacy_metadata[-1]
                    with gzip.open(folder / label / "final_state.json.gz", "wt", encoding="utf8", newline="\n") as stream:
                        json.dump(final, stream, separators=(",", ":"))
                        stream.write("\n")
                    if pair_index == 0:
                        assert len(snapshots[label]) == 200
                        with gzip.open(folder / label / "states.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
                            for snapshot in snapshots[label]:
                                stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                    with np.load(gold_dir / "trajectory/run3.npz") as gold, np.load(folder / label / "trajectory/run3.npz") as actual:
                        assert gold.files == actual.files
                        for key in gold.files:
                            a, b = gold[key], actual[key]
                            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
                for model, original, initialize, step in original_classes:
                    assert model.RbpfGtsamRtkTc is original
                    assert model.RbpfFgo.__init__ is initialize and model.RbpfFgo.step is step
                assert policy == (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks))
                after_cache = cache_manifest(Path(os.environ["NUMBA_CACHE_DIR"]))
                if pair_index == 0:
                    assert after_cache
                    warm_cache = after_cache
                result.update(pair=pair_index, warm=pair_index == 0, configs=configs, AST_proofs=proofs,
                    all_trajectory_and_final_state_bits_equal_gold=True,
                    warm_200_states_equal_AA_and_legacy_except_declared_RTK_boundary=pair_index == 0,
                    PF_calls={label: controllers[label].route_calls for label in LABELS},
                    cache_before=before, cache_after=after_cache,
                    cache_stable_after_warm=pair_index == 0 or before == after_cache == warm_cache,
                    GC_policy_unchanged=True, model_classes_and_numerical_methods_unmodified=True)
                write(folder / "pair.json", result)
                pairs.append(result)
                print(json.dumps(dict(pair=pair_index, warm=pair_index == 0, complete=True,
                    cache_stable=result["cache_stable_after_warm"])), flush=True)
                # Release prototype-owned references outside all clocks. Normal GC
                # and shared JIT/module caches remain unchanged; no forced collect.
                instances.clear()
                controllers.clear()
                offers.clear()
                owner.offers.clear()
                streams.clear()
                original_classes.clear()
                snapshots.clear()
                del function, controller, final
    finally:
        sys.argv = original_argv
    summary = evaluate(pairs)
    summary.update(verified=True, prefixes=12, RTK_calls=4800, PF_calls=2400,
        all_accuracy_checks_passed=True, all_cache_checks_passed=all(p["cache_stable_after_warm"] for p in pairs),
        primary_epoch_snapshots=0, warm_postprocess_snapshots=400, no_completed_run_retried=True,
        scope="Single-process serial paired RTK calls;two independent drivers/graphs/RNGs,shared D236 scorer. Both input preparations precede clocks;order alternates by epoch/pair. Normal GC,single allowed CPU,nonexclusive host. No continuous-driver throughput,GPU utilization,new optimization gain,fullroute or multiseed acceptance.")
    summary["calibration_pass"] &= summary["all_cache_checks_passed"]
    for section in ("source_sha256", "external_python_sha256", "input_sha256", "gold_sha256"):
        assert all(sha(ROOT / name) == digest for name, digest in plan[section].items())
    assert all(sha(ROOT / item["path"]) == item["sha256"] for item in plan["binaries"].values())
    assert sha(Path(plan["gtsam_pe"])) == plan["external_GTSAM_PE_SHA256"]
    write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
