// Display-only link model: per-satellite 20 Mm reference,
// fixed 0.5 reflected field coefficient, earliest path only. No C/N0 claim.
export function relativePower(signal) {
  if (signal.status === "BLOCKED") return null;
  if (!["LOS", "NLOS"].includes(signal.status)) throw new RangeError("Unknown path class");
  if (signal.status === "NLOS" && (!Number.isFinite(signal.reflection?.extra) || signal.reflection.extra < 0)) throw new RangeError("Invalid reflection excess");
  const distance = signal.range + (signal.status === "NLOS" ? signal.reflection?.extra : 0);
  if (!Number.isFinite(distance) || distance <= 0) throw new RangeError("Invalid signal path length");
  return 20 * Math.log10(20000000 / distance) + (signal.status === "NLOS" ? 20 * Math.log10(0.5) : 0);
}

export class SignalHistory {
  constructor() { this.clear(); }
  clear() { this.samples = []; this.key = null; }
  append(snapshot) {
    const time = snapshot.week * 604800 + snapshot.tow;
    if (!Number.isFinite(time)) throw new RangeError("Invalid signal time");
    const key = JSON.stringify([snapshot.antenna, snapshot.mask, snapshot.systems]);
    const last = this.samples.at(-1);
    if (key !== this.key || (last && time < last.time)) this.clear();
    this.key = key;
    const sample = {time, epoch:snapshot.epoch, signals:snapshot.satellites.map(s => ({id:s.id, status:s.status, power:relativePower(s)}))};
    if (this.samples.at(-1)?.time === time) this.samples.pop();
    this.samples.push(sample);
    this.samples = this.samples.filter(s => time - s.time <= 120).slice(-600);
  }
}

const NS = "http://www.w3.org/2000/svg";
const COLORS = {G:"#85e4bc", E:"#91bfff", C:"#efbc79", J:"#c5a4ff", R:"#ec8f91"};
export class SignalHistoryPlot {
  constructor(root) {
    this.root = root; this.history = new SignalHistory();
    this.svg = root.querySelector("svg"); this.readout = root.querySelector("output");
  }
  clear() { this.history.clear(); this.render(); }
  append(snapshot) { this.history.append(snapshot); }
  render(selected) {
    const samples = this.history.samples, latest = samples.at(-1);
    this.root.dataset.samples = samples.length;
    this.svg.replaceChildren();
    if (!latest) { delete this.root.dataset.epoch; delete this.root.dataset.selected; this.readout.textContent = "Waiting for a completed trace…"; return; }
    this.root.dataset.epoch = latest.epoch;
    this.root.dataset.selected = selected ?? "";
    const node = (tag, attributes, text) => {
      const el = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
      if (text !== undefined) el.textContent = text;
      this.svg.append(el); return el;
    };
    const values = samples.flatMap(s => s.signals.map(v => v.power).filter(Number.isFinite));
    const low = Math.min(-12, Math.floor(Math.min(0,...values) / 5) * 5), high = Math.max(3, Math.ceil(Math.max(0,...values) / 5) * 5);
    const x = t => 45 + (t - latest.time + 120) / 120 * 645;
    const y = power => 14 + (high - power) / (high - low) * 126;
    for (const level of [high,0,low]) {
      node("line", {x1:45,x2:690,y1:y(level),y2:y(level),stroke:"#ffffff20"});
      node("text", {x:39,y:y(level)+4,"text-anchor":"end",fill:"#a9b9b6","font-size":11},`${level}`);
    }
    for (const age of [120,60,0]) node("text", {x:x(latest.time-age),y:159,"text-anchor":age===0?"end":"start",fill:"#a9b9b6","font-size":11},age?`−${age} s`:"Completed GPST");
    const ids = [...new Set(samples.flatMap(s => s.signals.map(v => v.id)))].sort((a,b) => Number(a===selected)-Number(b===selected) || a.localeCompare(b));
    for (const id of ids) {
      let previous, path = "";
      for (const sample of samples) {
        const signal = sample.signals.find(s => s.id===id);
        if (signal?.power == null) { previous = null; continue; }
        const attrs = {stroke:COLORS[id[0]] ?? "white",opacity:selected===id?1:0.22};
        // Missing/blocked samples and gaps >10 simulated seconds break the line.
        path += `${previous && sample.time-previous.time<=10 ? "L" : "M"}${x(sample.time)},${y(signal.power)} `;
        if (id===selected || sample===latest) {
          const point = node("circle", {...attrs,cx:x(sample.time),cy:y(signal.power),r:id===selected?2.5:2,fill:attrs.stroke});
          const title = document.createElementNS(NS,"title"); title.textContent = `${id} · ${signal.status} · ${signal.power.toPrecision(8)} dB · GPST ${sample.time}`; point.append(title);
        }
        previous = {time:sample.time,power:signal.power};
      }
      node("path", {d:path,fill:"none",stroke:COLORS[id[0]] ?? "white",opacity:selected===id?1:0.22,"stroke-width":id===selected?2.5:1,"data-signal":id});
    }
    const signal = latest.signals.find(s=>s.id===selected);
    const clock = new Date(Date.UTC(1980,0,6) + latest.time * 1000).toISOString().slice(11,23);
    this.readout.textContent = `${selected ?? "—"} · ${signal?.power == null ? "No modeled path" : `${signal.power.toFixed(2)} dB`} · ${clock} GPST · ${samples.length} samples`;
  }
}
