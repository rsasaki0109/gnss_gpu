import { signalMetrics, validatePower } from "./power-model.mjs?v=analysis-1";
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
  append(snapshot, scene, options) {
    const model=validatePower(options);
    const time = snapshot.week * 604800 + snapshot.tow;
    if (!Number.isFinite(time)) throw new RangeError("Invalid signal time");
    const key = JSON.stringify([snapshot.antenna, snapshot.mask, snapshot.systems, model]);
    const last = this.samples.at(-1);
    if (key !== this.key || (last && time < last.time)) this.clear();
    this.key = key;
    const sample = {time, week:snapshot.week, tow:snapshot.tow, epoch:snapshot.epoch, antenna:snapshot.antenna, mask:snapshot.mask, systems:snapshot.systems, model, bias:snapshot.spp?.all?.horizontal??null, signals:scene ? signalMetrics(snapshot,scene,model) : snapshot.satellites.map(s => ({id:s.id, status:s.status, power:relativePower(s), elevation:s.elevation, extra:s.status==="LOS"?0:s.status==="NLOS"?s.reflection.extra:null}))};
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
  append(snapshot,scene,options) { this.history.append(snapshot,scene,options); }
  render(selected) {
    const metric=this.root.querySelector("#history-metric")?.value??"power";
    const unit={power:"dB",elevation:"°",extra:"m",bias:"m"}[metric];
    const heading=this.root.querySelector("#history-unit");if(heading)heading.textContent=metric==="power"?"Model · relative dB":metric==="bias"?"Geometric fit² · m":`${metric} · ${unit}`;
    const samples = this.history.samples.map(sample=>({...sample,signals:metric==="bias"?[{id:"FIT",status:"GEOMETRIC",power:sample.bias}]:sample.signals.map(signal=>({...signal,power:signal[metric]}))})), latest = samples.at(-1);
    if(metric==="bias")selected="FIT";
    this.root.dataset.samples = samples.length;
    this.svg.replaceChildren();
    const exportButton=this.root.querySelector("#history-export");if(exportButton)exportButton.disabled=!latest;
    this.svg.setAttribute("aria-label", `Completed ${metric} history, ${unit}`);
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
    const low = metric==="power"?Math.min(-12, Math.floor(Math.min(0,...values)/5)*5):0, high = Math.max(metric==="elevation"?90:metric==="power"?3:10, Math.ceil(Math.max(0,...values)/5)*5);
    const x = t => 45 + (t - latest.time + 120) / 120 * 645;
    const y = power => 14 + (high - power) / (high - low) * 126;
    for (const level of [...new Set([high,0,low])]) {
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
          const title = document.createElementNS(NS,"title"); title.textContent = `${id} · ${signal.status} · ${signal.power.toPrecision(8)} ${unit} · GPST ${sample.time}`; point.append(title);
        }
        previous = {time:sample.time,power:signal.power};
      }
      node("path", {d:path,fill:"none",stroke:COLORS[id[0]] ?? "white",opacity:selected===id?1:0.22,"stroke-width":id===selected?2.5:1,"data-signal":id});
    }
    const signal = latest.signals.find(s=>s.id===selected);
    const clock = new Date(Date.UTC(1980,0,6) + latest.time * 1000).toISOString().slice(11,23);
    this.readout.textContent = `${selected ?? "—"} · ${signal?.power == null ? (metric==="power"&&signal?.powerState==="zero"?"Zero by RF model":"Value unavailable") : `${signal.power.toFixed(2)} ${unit}`} · ${clock} GPST · ${samples.length} samples`;
  }
}

export function historyCSV(samples) {
  const lines=["epoch,gps_week,gps_tow_s,satellite,path,elevation_deg,earliest_excess_m,modeled_relative_power_db,power_state,frequency_hz,horizontal_geometric_bias_m,material_assumption,polarization,antenna_pattern,systems,antenna_offset_m,mask_deg"];
  for(const sample of samples)for(const signal of sample.signals)lines.push([sample.epoch,sample.week,sample.tow,signal.id,signal.status,signal.elevation,signal.extra,signal.power,signal.powerState,signal.frequency,sample.bias,sample.model.material,sample.model.polarization,sample.model.antenna,sample.systems,sample.antenna,sample.mask].map(v=>v==null?"":String(v)).join(","));
  return lines.join("\r\n")+"\r\n";
}
