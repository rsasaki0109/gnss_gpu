"""Predeclared serial RTK screen for the checked ordered-H registration plan."""

from statistics import median

from rbpf_epoch_clocks import EpochClocks, summarize

VARIANTS = ("reference", "off", "candidate", "AA")
# A four-row Williams design, followed by its declared fifth row. No data
# dependent ordering, retries, epoch omissions, or signed A/A cancellation.
ORDERS = (
    ("reference", "off", "AA", "candidate"),
    ("off", "candidate", "reference", "AA"),
    ("candidate", "AA", "off", "reference"),
    ("AA", "reference", "candidate", "off"),
    ("candidate", "AA", "off", "reference"),
)
THRESHOLDS = {"all": 3.0, "2": 5.0, "driver": 3.0}


def schedule():
    return [dict(warm=True, round=-1, variant=v) for v in VARIANTS] + [
        dict(warm=False, round=r, variant=v)
        for r, order in enumerate(ORDERS) for v in order
    ]


class RouteClocks(EpochClocks):
    """The same clocks for every arm, with one complete call per epoch.

    No barrier, state snapshot, metadata encoding or GC observer runs between
    epochs. Small clock record allocations remain common instrumentation.
    """

    def totals(self):
        values = summarize(self.records)
        if (len(self.records) != 400 or values["1"]["epochs"] != 200
                or values["2"]["epochs"] != 200
                or [r["phase_before"] for r in self.records] != [1] * 200 + [2] * 200):
            raise ValueError("Expected all 400 epochs in the frozen two phases")
        return values


def retaining_type(original):
    """Retain only the latest return by reference; never copy timed PF state."""

    class Retained(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.route_calls = 0
            self.route_last = None

        def __call__(self, rtk, ctx):
            result = super().__call__(rtk, ctx)
            self.route_last = (rtk, result)
            self.route_calls += 1
            return result

    return Retained


def metric_totals(performance):
    # Strict integer aggregation before any division. Phase 1 is retained
    # separately in the raw/summary evidence but is not an alternate gate.
    rows = performance["records"]
    values = summarize(rows)
    if (len(rows) != 400 or values["1"]["epochs"] != 200
            or values["2"]["epochs"] != 200
            or [r["phase_before"] for r in rows] != [1] * 200 + [2] * 200):
        raise ValueError("Incomplete or reordered RTK prefix")
    driver = performance["driver_wall_ns"]
    if type(driver) is not int or driver <= 0:
        raise ValueError("Invalid driver wall clock")
    result = {k: values[k]["wall_ns"] for k in ("all", "1", "2")}
    if any(type(v) is not int or v <= 0 for v in result.values()):
        raise ValueError("Invalid process wall clocks")
    if driver < result["all"]:
        raise ValueError("Driver cannot be shorter than its RTK bodies")
    return dict(result, driver=driver)


def evaluate(cases):
    """Require stable A/A and useful gains against both baseline and off."""
    expected = schedule()
    if len(cases) != len(expected) or [c["case"] for c in cases] != expected:
        raise ValueError("Missing, duplicated or reordered scheduled prefix")
    totals = [metric_totals(c["performance"]) for c in cases]
    rounds = {r: {} for r in range(5)}
    for item, value in zip(cases, totals, strict=True):
        if not item["case"]["warm"]:
            rounds[item["case"]["round"]][item["case"]["variant"]] = value
    result = {}
    for key, threshold in THRESHOLDS.items():
        contrasts = {}
        for label, left, right in (("reference", "reference", "candidate"),
                ("off", "off", "candidate"), ("AA", "reference", "AA")):
            percentages = [100.0 * (arms[left][key] - arms[right][key]) / arms[left][key]
                for arms in rounds.values()]
            contrasts[label] = dict(paired_percent=percentages,
                median_percent=median(percentages), positive_rounds=sum(v > 0 for v in percentages),
                absolute_median_percent=median(abs(v) for v in percentages),
                absolute_rounds_under5=sum(abs(v) < 5.0 for v in percentages))
        stable = (contrasts["AA"]["absolute_median_percent"] < 5.0
            and contrasts["AA"]["absolute_rounds_under5"] >= 4)
        for label in ("reference", "off"):
            contrasts[label]["gain_gate"] = (contrasts[label]["median_percent"] >= threshold
                and contrasts[label]["positive_rounds"] >= 4)
        result[key] = dict(minimum_percent=threshold, contrasts=contrasts,
            AA_valid=stable, pass_gate=stable and all(contrasts[k]["gain_gate"] for k in ("reference", "off")))
    return dict(metrics=result, speed_gate=all(item["pass_gate"] for item in result.values()),
        all_warm_and_timed_prefixes_retained=True, no_completed_timing_retry=True)
