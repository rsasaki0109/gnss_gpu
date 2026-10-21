// Summaries of completed Float64 results only: no tracing, interpolation or rounding.
export function captureReference(snapshot) {
  const satellites = snapshot.satellites.map(signal => Object.freeze({
    id: signal.id, status: signal.status,
    excess: signal.status === "LOS" ? 0 : signal.status === "NLOS" && Number.isFinite(signal.reflection?.extra) ? signal.reflection.extra : null,
  }));
  return Object.freeze({ epoch: snapshot.epoch, week: snapshot.week, tow: snapshot.tow,
    antenna: snapshot.antenna, mask: snapshot.mask, counts: Object.freeze({ ...snapshot.counts }),
    bias: Number.isFinite(snapshot.spp.all?.horizontal) ? snapshot.spp.all.horizontal : null,
    satellites: Object.freeze(satellites) });
}

export function compareReceiver(reference, snapshot) {
  if (!reference || !snapshot) return { state: "empty" };
  if (reference.epoch !== snapshot.epoch || reference.week !== snapshot.week || reference.tow !== snapshot.tow) return { state: "different-pose" };
  const current = captureReference(snapshot), before = new Map(reference.satellites.map(signal => [signal.id, signal])), after = new Map(current.satellites.map(signal => [signal.id, signal]));
  const signals = [...new Set([...before.keys(), ...after.keys()])].sort().map(id => {
    const a = before.get(id), b = after.get(id);
    return { id, before: a?.status ?? "MASKED", after: b?.status ?? "MASKED", beforeExcess: a?.excess ?? null, afterExcess: b?.excess ?? null };
  });
  const metrics = ["LOS", "NLOS", "BLOCKED", "bias"].map(key => {
    const a = key === "bias" ? reference.bias : reference.counts[key], b = key === "bias" ? current.bias : current.counts[key];
    return { key, before: a, after: b, delta: a === null || b === null ? null : b - a };
  });
  return { state: "ready", reference, current, signals, metrics, changes: signals.filter(signal => signal.before !== signal.after).length };
}
