"""Read sealed paired clocks and observe literal score boundaries,no speed gate."""

from statistics import median

from rbpf_epoch_clocks import EpochClocks
from rbpf_score_host_phases_v2 import spans

CLOCKS = ("wall_ns", "process_cpu_ns", "thread_cpu_ns")
LABELS = ("A", "AA")


def paired_rows(pair):
    if type(pair["pair"]) is not int or pair["pair"] < 0:
        raise ValueError("Invalid pair identity")
    records = pair["records"]
    if set(records) != set(LABELS):
        raise ValueError("Both complete streams required")
    expected = []
    for epoch in range(400):
        order = LABELS if (epoch + pair["pair"]) % 2 == 0 else LABELS[::-1]
        prepare = order[::-1]
        sequence = [("ready", label) for label in prepare]
        sequence += [(event, label) for label in order for event in ("run", "done")]
        sequence += [("advance", label) for label in prepare]
        expected.extend(dict(event=event, epoch=epoch, label=label, phase=1 if epoch < 200 else 2)
            for event, label in sequence)
    if pair["events"] != expected:
        raise ValueError("Original complete balanced protocol events required")
    for rows in records.values():
        if (len(rows) != 400 or [r["epoch"] for r in rows] != list(range(400))
                or any(type(r["epoch"]) is not int for r in rows)
                or [r["phase_before"] for r in rows] != [1] * 200 + [2] * 200
                or any(type(r["phase_before"]) is not int for r in rows)
                or any(r["completed"] is not True for r in rows)
                or any(type(r[key]) is not int or r[key] < 0 for r in rows for key in CLOCKS)):
            raise ValueError("Complete nonnegative integer clocks in original order required")
    result = []
    for a, b in zip(records["A"], records["AA"], strict=True):
        epoch = a["epoch"]
        delta = {key: a[key] - b[key] for key in CLOCKS}
        delta["wall_minus_thread_ns"] = delta["wall_ns"] - delta["thread_cpu_ns"]
        sign = 1 if (epoch + pair["pair"]) % 2 == 0 else -1
        result.append(dict(epoch=epoch, phase=a["phase_before"],
            first_label="A" if sign == 1 else "AA", A_minus_AA=delta,
            first_minus_second={key: sign * value for key, value in delta.items()}))
    return result


def analyze(pairs, *, first_valid_epoch):
    if (len(pairs) != 6 or [p["pair"] for p in pairs] != list(range(6))
            or [p["warm"] for p in pairs] != [True] + [False] * 5
            or any(type(p["warm"]) is not bool for p in pairs)
            or type(first_valid_epoch) is not int or not 200 <= first_valid_epoch < 400):
        raise ValueError("Sealed original six pairs and declared first valid epoch required")
    results = []
    for pair in pairs:
        rows = paired_rows(pair)
        phases = {}
        for phase in (1, 2):
            selected = [r for r in rows if r["phase"] == phase]
            totals = {label: {key: sum(r[key] for r in pair["records"][label] if r["phase_before"] == phase)
                for key in CLOCKS} for label in LABELS}
            difference = {key: sum(r["A_minus_AA"][key] for r in selected)
                for key in (*CLOCKS, "wall_minus_thread_ns")}
            if difference["wall_ns"] != difference["thread_cpu_ns"] + difference["wall_minus_thread_ns"]:
                raise ValueError("Integer clock conservation failed")
            phases[str(phase)] = dict(totals=totals, signed_A_minus_AA_ns=difference)
        initial = rows[first_valid_epoch]
        rest = {key: phases["2"]["signed_A_minus_AA_ns"][key] - initial["A_minus_AA"][key]
            for key in initial["A_minus_AA"]}
        results.append(dict(pair=pair["pair"], warm=pair["warm"], phases=phases,
            first_valid=initial, phase2_remaining_signed_difference_ns=rest,
            all_epoch_differences=rows))
    initial = [r["first_valid"]["first_minus_second"] for r in results[1:]]
    return dict(results=results, RTK_clocks=4800, paired_epoch_differences=2400,
        first_valid_first_minus_second_wall_ns=[r["wall_ns"] for r in initial],
        first_valid_first_minus_second_thread_cpu_ns=[r["thread_cpu_ns"] for r in initial],
        first_valid_first_minus_second_residual_ns=[r["wall_minus_thread_ns"] for r in initial],
        median_first_valid_first_minus_second_wall_ns=median(r["wall_ns"] for r in initial),
        original_failed_gate_preserved=True, counterfactual_filtered_gate_calculated=False,
        cause_established=False, optimization_speedup_established=False,
        scope="Signed differences conserve wall=thread CPU+residual. Residual includes blocking,preemption and quantization;it does not identify GPU/OS/GC cause. First-call partition is diagnostic;all original epochs and failed gates remain.")


class ScoreObserver:
    """Keep all original calls/results/exceptions;record after each score clock."""

    def __init__(self, context, *, clocks=None):
        self.context = context
        self.clocks = EpochClocks() if clocks is None else clocks
        self.records = []

    def bind(self, scorer):
        function = scorer.score

        def recorded(*args, **kwargs):
            if self.clocks.active:
                raise RuntimeError("Score observations must be serial")
            label, epoch = self.context()
            if label not in LABELS or type(epoch) is not int or not 0 <= epoch < 400:
                raise ValueError("Owned original RTK epoch required")
            before = scorer.last_profile
            count = len(self.clocks.records)
            try:
                return self.clocks.call(2, function, *args, **kwargs)
            finally:
                if len(self.clocks.records) != count + 1:
                    raise RuntimeError("Score clock record missing")
                row = dict(self.clocks.records[-1], label=label, RTK_epoch=epoch,
                    events=scorer.last_profile, partition=None, probe_error=None)
                try:
                    if scorer.last_profile is before:
                        raise ValueError("Stale or missing score trace")
                    row["partition"] = spans(scorer.last_profile)
                    if row["partition"]["total"] > row["wall_ns"]:
                        raise ValueError("Inner score trace exceeds outer span")
                except Exception as error:
                    row["probe_error"] = type(error).__name__ + ":" + str(error)
                self.records.append(row)
        return recorded
