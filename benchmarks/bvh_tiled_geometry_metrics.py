"""Fixed whole-call gates; every warm/cold primary and both A/A controls."""

from collections import defaultdict
from statistics import median

MODES = ("original", "off", "on", "original_AA", "off_AA")
CONTRASTS = (
    ("original_on", "original", "on"),
    ("off_on", "off", "on"),
    ("original_AA", "original", "original_AA"),
    ("off_AA", "off", "off_AA"),
)


def evaluate(records, primary):
    result = {}
    for phase in ("timed", "cold"):
        totals = defaultdict(int)
        for row in records:
            if row["phase"] == phase:
                totals[row["round"], row.get("group", 0), row["mode"]] += row["ns"]
        stats = {}
        for label, a, b in CONTRASTS:
            rounds = [
                median(
                    100 * (1 - totals[r, g, b] / totals[r, g, a])
                    for g in range(3 if phase == "timed" else 1)
                )
                for r in range(5)
            ]
            stats[label] = dict(
                reference=a,
                candidate=b,
                round_percent=rounds,
                median_percent=median(rounds),
                positive_rounds=sum(v > 0 for v in rounds),
                absolute_median_percent=median(abs(v) for v in rounds),
                absolute_rounds_under5=sum(abs(v) < 5 for v in rounds),
            )
        valid = all(
            stats[n]["absolute_median_percent"] < 5
            and stats[n]["absolute_rounds_under5"] >= 4
            for n in ("original_AA", "off_AA")
        )
        limit = 10 if primary and phase == "timed" else -5
        improvement = all(
            stats[n]["median_percent"] >= limit
            and sum(
                v >= limit if limit < 0 else v > 0 for v in stats[n]["round_percent"]
            )
            >= 4
            for n in ("original_on", "off_on")
        )
        result[phase] = dict(
            contrasts=stats,
            AA_valid=valid,
            numeric_gate=improvement,
            passed=valid and improvement,
            required_median_percent=limit,
        )
    result["passed"] = all(result[p]["passed"] for p in ("timed", "cold"))
    return result
