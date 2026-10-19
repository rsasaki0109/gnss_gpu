"""Integer conservation,first-call order and literal score observer controls."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_epoch_clocks import EpochClocks  # noqa: E402
from rbpf_pair_residuals import ScoreObserver, analyze, paired_rows  # noqa: E402
from rbpf_score_host_phases_v2 import PHASES  # noqa: E402


def pairs():
    result = []
    for pair in range(6):
        records = {label: [dict(epoch=e, phase_before=1 if e < 200 else 2, wall_ns=100,
            process_cpu_ns=70, thread_cpu_ns=50, completed=True) for e in range(400)] for label in ("A", "AA")}
        first = "A" if (pair + 203) % 2 == 0 else "AA"
        records[first][203]["wall_ns"] += 80
        records[first][203]["thread_cpu_ns"] += 10
        records[first][203]["process_cpu_ns"] += 10
        events = []
        for epoch in range(400):
            order = ("A", "AA") if (epoch + pair) % 2 == 0 else ("AA", "A")
            sequence = [("ready", label) for label in order[::-1]]
            sequence += [(event, label) for label in order for event in ("run", "done")]
            sequence += [("advance", label) for label in order[::-1]]
            events.extend(dict(event=event, label=label, epoch=epoch, phase=1 if epoch < 200 else 2)
                for event, label in sequence)
        result.append(dict(pair=pair, warm=pair == 0, events=events, records=records))
    return result


def test_first_call_orients_both_roles_without_signed_cancellation():
    result = analyze(pairs(), first_valid_epoch=203)
    assert result["first_valid_first_minus_second_wall_ns"] == [80] * 5
    assert result["first_valid_first_minus_second_thread_cpu_ns"] == [10] * 5
    assert result["first_valid_first_minus_second_residual_ns"] == [70] * 5
    assert result["median_first_valid_first_minus_second_wall_ns"] == 80
    assert [r["first_valid"]["first_label"] for r in result["results"]] == ["AA", "A"] * 3
    assert result["RTK_clocks"] == 4800 and result["paired_epoch_differences"] == 2400
    assert not result["counterfactual_filtered_gate_calculated"] and not result["optimization_speedup_established"]


def test_all_epoch_integer_conservation_retains_negative_residuals_and_warm():
    data = pairs()
    data[0]["records"]["A"][390]["thread_cpu_ns"] += 500
    result = analyze(data, first_valid_epoch=203)
    assert len(result["results"]) == 6 and result["results"][0]["warm"] is True
    for pair in result["results"]:
        assert len(pair["all_epoch_differences"]) == 400
        for phase in ("1", "2"):
            delta = pair["phases"][phase]["signed_A_minus_AA_ns"]
            assert delta["wall_ns"] == delta["thread_cpu_ns"] + delta["wall_minus_thread_ns"]
        first, rest = pair["first_valid"]["A_minus_AA"], pair["phase2_remaining_signed_difference_ns"]
        assert all(first[k] + rest[k] == pair["phases"]["2"]["signed_A_minus_AA_ns"][k] for k in first)
    assert result["results"][0]["all_epoch_differences"][390]["A_minus_AA"]["wall_minus_thread_ns"] == -500


@pytest.mark.parametrize("kind", ["missing", "epoch_order", "epoch_bool", "phase", "phase_bool", "incomplete", "negative", "float", "event_order", "label", "pair_bool"])
def test_bad_pair_clock_or_protocol_rejected(kind):
    pair = pairs()[1]
    rows = pair["records"]["A"]
    if kind == "missing":
        rows.pop()
    elif kind == "epoch_order":
        rows[3]["epoch"] = 2
    elif kind == "epoch_bool":
        rows[0]["epoch"] = False
    elif kind in ("phase", "phase_bool"):
        rows[0]["phase_before"] = 2 if kind == "phase" else True
    elif kind == "incomplete":
        rows[0]["completed"] = False
    elif kind in ("negative", "float"):
        rows[0]["wall_ns"] = -1 if kind == "negative" else 1.0
    elif kind == "event_order":
        pair["events"][2], pair["events"][3] = pair["events"][3], pair["events"][2]
    elif kind == "label":
        pair["records"]["other"] = pair["records"].pop("AA")
    else:
        pair["pair"] = True
    with pytest.raises(ValueError):
        paired_rows(pair)


@pytest.mark.parametrize("kind", ["missing", "order", "warm", "warm_type", "epoch_low", "epoch_high", "epoch_bool"])
def test_sealed_schedule_and_first_valid_epoch_required(kind):
    data, first = pairs(), 203
    if kind == "missing":
        data.pop()
    elif kind == "order":
        data.reverse()
    elif kind == "warm":
        data[1]["warm"] = True
    elif kind == "warm_type":
        data[0]["warm"] = 1
    else:
        first = {"epoch_low": 199, "epoch_high": 400, "epoch_bool": True}[kind]
    with pytest.raises(ValueError):
        analyze(data, first_valid_epoch=first)


def monitor():
    def counter(values):
        return iter(values).__next__
    return EpochClocks(counter([0, 1000, 2000, 3000]), counter([10, 90, 110, 190]), counter([0, 50, 100, 150]))


def events():
    return tuple((name, 100 + i) for i, name in enumerate((*PHASES, "end")))


def test_score_args_return_identity_and_trace_ownership():
    observer = ScoreObserver(lambda: ("A", 203), clocks=monitor())
    arg, token, answer, calls = [], object(), object(), []
    scorer = SimpleNamespace(last_profile=None)

    def score(value, *, key):
        assert value is arg and key is token and observer.clocks.active
        calls.append(value)
        scorer.last_profile = events()
        return answer

    scorer.score = score
    recorded = observer.bind(scorer)
    assert recorded(arg, key=token) is answer and calls == [arg]
    row = observer.records[0]
    assert row["label"] == "A" and row["RTK_epoch"] == 203 and row["wall_ns"] == 1000
    assert row["process_cpu_ns"] == 80 and row["thread_cpu_ns"] == 50 and row["completed"] is True
    assert row["events"] is scorer.last_profile and row["partition"]["native"] == 1
    assert row["partition"]["total"] == 9 and row["probe_error"] is None
    assert not observer.clocks.active and scorer.score is score


def test_original_score_exception_preserved_with_partial_state():
    observer = ScoreObserver(lambda: ("AA", 203), clocks=monitor())
    error, changed = LookupError("original"), []
    scorer = SimpleNamespace(last_profile=None)

    def score():
        changed.append(1)
        scorer.last_profile = events()
        raise error

    scorer.score = score
    with pytest.raises(LookupError) as raised:
        observer.bind(scorer)()
    assert raised.value is error and changed == [1] and not observer.clocks.active
    assert observer.records[0]["completed"] is False and observer.records[0]["probe_error"] is None


@pytest.mark.parametrize("kind", ["stale", "topology", "clock_order", "inner_longer"])
def test_probe_errors_retained_without_changing_numeric_return(kind):
    observer = ScoreObserver(lambda: ("A", 203), clocks=monitor())
    scorer = SimpleNamespace(last_profile=None)
    answer = object()

    def score():
        if kind == "topology":
            scorer.last_profile = (("invalid", 100), ("end", 101))
        elif kind == "clock_order":
            scorer.last_profile = (("entry", 101), ("end", 100))
        elif kind == "inner_longer":
            scorer.last_profile = (("entry", 0), ("end", 1001))
        return answer

    scorer.score = score
    assert observer.bind(scorer)() is answer
    assert observer.records[0]["probe_error"] and observer.records[0]["completed"] is True


@pytest.mark.parametrize("label,epoch", [("other", 203), ("A", True), ("A", -1), ("A", 400)])
def test_unowned_context_rejected_before_score(label, epoch):
    observer = ScoreObserver(lambda: (label, epoch), clocks=monitor())
    calls = []
    scorer = SimpleNamespace(last_profile=None, score=lambda: calls.append(1))
    with pytest.raises(ValueError):
        observer.bind(scorer)()
    assert not calls and not observer.records


def test_recursive_score_observer_rejected_and_original_failure_recorded():
    observer = ScoreObserver(lambda: ("A", 203), clocks=monitor())
    scorer = SimpleNamespace(last_profile=None)

    def score():
        scorer.last_profile = events()
        return recorded()

    scorer.score = score
    recorded = observer.bind(scorer)
    with pytest.raises(RuntimeError, match="serial"):
        recorded()
    assert len(observer.records) == 1 and observer.records[0]["completed"] is False
    assert not observer.clocks.active
