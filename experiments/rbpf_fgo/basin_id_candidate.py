"""Experimental key-sorting candidate; not used by the live RB-FGO-PF runtime.

Actual-call wall and process-CPU comparisons improved only 6/10 pairs each.
Retain for reproduction, without claiming a stable latency improvement.
"""


def _canon_basin_id(H):
    """Gauge-invariant, hashable basin identity."""
    out = []
    for g in sorted(H.keys()):
        zmap = H[g]
        if not zmap:
            continue
        sats = sorted(zmap)
        base = zmap[sats[0]]
        out.append((g, tuple((s, zmap[s] - base) for s in sats)))
    return tuple(out)
