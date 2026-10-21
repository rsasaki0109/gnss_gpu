// Display projection of completed satellite directions. No propagation or tracing.
import { reflectedArrival } from "./arrival-direction.mjs";

export function skyPosition(azimuth, elevation) {
  if (!Number.isFinite(azimuth) || !Number.isFinite(elevation) || elevation < 0 || elevation > 90) throw new RangeError("Invalid sky direction");
  const angle = (azimuth % 360) * Math.PI / 180, radius = (90 - elevation) / 90;
  return [radius * Math.sin(angle), -radius * Math.cos(angle)];
}

export function skyLayout(signals) {
  const points = signals.map(({id, azimuth, elevation, status}) => {
    const [x,y] = skyPosition(azimuth,elevation);
    return { id, azimuth, elevation, status, x:150+110*x, y:150+110*y };
  }).sort((a,b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  const boxes = [{x:99,y:195},{x:125,y:176}];
  const offsets = [[18,-12],[18,12],[-18,-12],[-18,12],[0,-23],[0,23],[28,0],[-28,0],[32,-16],[-32,16],[32,16],[-32,-16]];
  for (const point of points) {
    let best;
    for (const [dx,dy] of offsets) {
      const x = Math.max(32,Math.min(268,point.x+dx)), y = Math.max(32,Math.min(268,point.y+dy));
      const overlap = boxes.reduce((sum,box) => sum + Math.max(0,28-Math.abs(box.x-x))*Math.max(0,16-Math.abs(box.y-y)),0);
      const markers = points.reduce((sum,p) => sum + Math.max(0,9-Math.hypot(Math.max(0,Math.abs(p.x-x)-12),Math.max(0,Math.abs(p.y-y)-6))),0);
      const score = overlap + markers*100 + Math.hypot(dx,dy)*.01;
      if (!best || score < best.score) best = {x,y,score};
    }
    point.labelX = best.x; point.labelY = best.y; boxes.push(best);
  }
  return points;
}

const SVG = "http://www.w3.org/2000/svg";
function element(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG,tag);
  for (const [key,value] of Object.entries(attributes)) node.setAttribute(key,String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export class SkyPlot {
  constructor(root, onSelect) {
    this.root = root; this.onSelect = onSelect;
    this.svg = element("svg",{viewBox:"0 0 300 300",role:"group","aria-label":"GPS sky directions, north up and east right"});
    this.svg.append(element("desc",{},"Circles show geometric satellite directions above the completed elevation mask. The diamond shows the selected satellite's shortest reflected arrival when above the local horizon. Select with Enter or Space, or use the satellite list."));
    const grid = element("g",{"aria-hidden":"true",class:"sky-grid"});
    grid.append(element("circle",{cx:150,cy:150,r:110,class:"sky-horizon"}));
    this.mask = element("circle",{cx:150,cy:150,r:0,class:"sky-mask",visibility:"hidden"}); grid.append(this.mask);
    for (const elevation of [30,60]) grid.append(element("circle",{cx:150,cy:150,r:110*(90-elevation)/90,class:"sky-ring-grid"}));
    grid.append(element("path",{d:"M150 40V260 M40 150H260",class:"sky-axes"}));
    for (const [label,x,y] of [["N",150,20],["E",285,154],["S",150,288],["W",15,154]]) grid.append(element("text",{x,y,class:"sky-cardinal"},label));
    for (const [label,x,y] of [["30°",99,195],["60°",125,176]]) grid.append(element("text",{x,y,class:"sky-elevation"},label));
    grid.append(element("circle",{cx:150,cy:150,r:2,class:"sky-zenith"}));
    this.labels = element("g",{"aria-hidden":"true",class:"sky-labels"}); this.points = element("g",{class:"sky-points"});
    this.arrivalMarker = element("g",{class:"sky-arrival-marker",role:"img",visibility:"hidden","aria-hidden":"true"});
    this.arrivalLink = element("line",{class:"sky-arrival-link"});
    this.arrivalDiamond = element("path",{d:"M0 -7 L7 0 L0 7 L-7 0 Z",class:"sky-arrival-diamond"});
    this.arrivalMarker.append(this.arrivalLink,this.arrivalDiamond);
    this.svg.append(grid,this.labels,this.points,this.arrivalMarker); root.append(this.svg);
    this.source = document.createElement("p"); this.source.className = "sky-source";
    this.selection = document.createElement("p"); this.selection.className = "sky-selection";
    this.arrivalReadout = document.createElement("p"); this.arrivalReadout.className = "sky-arrival-readout";
    this.arrivalNote = document.createElement("p"); this.arrivalNote.className = "sky-arrival-note";
    root.append(this.source,this.selection,this.arrivalReadout,this.arrivalNote);
  }

  update(snapshot, selected, {scene, ready, error, playing, time}) {
    this.ready = ready;
    const state = error ? "unavailable" : !snapshot ? "loading" : ready ? "ready" : "pending";
    if (this.root.dataset.state !== state) this.root.dataset.state = state;
    const changed = Boolean(snapshot && this.snapshot !== snapshot);
    if (changed) {
      this.snapshot = snapshot; this.points.replaceChildren(); this.labels.replaceChildren();
      this.mask.setAttribute("r",String(110*(90-snapshot.mask)/90)); this.mask.setAttribute("visibility","visible");
      Object.assign(this.root.dataset,{epoch:snapshot.epoch,antenna:snapshot.antenna,mask:snapshot.mask,count:snapshot.satellites.length});
      for (const point of skyLayout(snapshot.satellites)) {
        const button = element("g",{role:"button",class:"sky-point",tabindex:0,"data-sky-satellite":point.id,"data-path":point.status,"aria-label":`${point.id}, azimuth ${point.azimuth.toFixed(1)} degrees, elevation ${point.elevation.toFixed(1)} degrees, ${point.status}`});
        Object.assign(button.dataset,{azimuth:point.azimuth,elevation:point.elevation});
        button.append(element("title",{},`${point.id} · ${point.status} · az ${point.azimuth.toFixed(1)}° / el ${point.elevation.toFixed(1)}°`),element("circle",{cx:point.x,cy:point.y,r:14,class:"sky-hit"}),element("circle",{cx:point.x,cy:point.y,r:10,class:"sky-selection-ring"}),element("circle",{cx:point.x,cy:point.y,r:5,class:"sky-dot"}));
        const select = () => { if (this.ready) this.onSelect(point.id); };
        button.addEventListener("click",select); button.addEventListener("keydown",event=>{ if (["Enter"," "].includes(event.key)) { event.preventDefault(); select(); } });
        this.points.append(button);
        const label = element("g",{"data-sky-label":point.id,"data-path":point.status});
        label.append(element("line",{x1:point.x,y1:point.y,x2:point.labelX,y2:point.labelY,class:"sky-leader"}),element("text",{x:point.labelX,y:point.labelY+3},point.id)); this.labels.append(label);
      }
    }
    const signal = snapshot?.satellites.find(signal=>signal.id===selected);
    if (changed || this.selected !== selected || this.lastReady !== ready || this.lastError !== error) {
      for (const button of this.points.children) {
        button.setAttribute("aria-pressed",String(button.dataset.skySatellite===selected)); button.setAttribute("aria-disabled",String(!ready)); button.setAttribute("tabindex",ready?"0":"-1");
      }
      for (const label of this.labels.children) label.setAttribute("data-selected",String(label.dataset.skyLabel===selected));
      this.updateArrival(scene,snapshot,signal,{ready,error});
      this.selected = selected; this.lastReady = ready; this.lastError = error;
    }
    const source = error ? "Sky unavailable · simulation stopped" : !snapshot ? "Preparing GPS sky…" : `${ready ? "Completed" : playing ? "Live completed trace · updating" : "Previous trace · updating"} · ${snapshot.satellites.length} GPS · ${snapshot.mask}° mask · ${time} GPST`;
    if (this.source.textContent !== source) this.source.textContent = source;
    const description = signal ? `${signal.id} · az ${signal.azimuth.toFixed(1)}° · el ${signal.elevation.toFixed(1)}° · ${signal.status}` : "Select a satellite to inspect its direction.";
    if (this.selection.textContent !== description) this.selection.textContent = description;
  }

  updateArrival(scene, snapshot, signal, {ready,error}) {
    this.arrivalMarker.setAttribute("visibility","hidden"); this.arrivalMarker.setAttribute("aria-hidden","true");
    for (const name of Object.keys(this.arrivalReadout.dataset)) delete this.arrivalReadout.dataset[name];
    let state, text, note = "";
    if (error) { state = "unavailable"; text = "Arrival unavailable · simulation stopped"; }
    else if (!snapshot) { state = "loading"; text = "Preparing arrival direction…"; }
    else if (!ready) { state = "pending"; text = "Arrival updating · waiting for the selected trace"; }
    else if (!signal) { state = "unselected"; text = "Select a satellite to compare arrival directions."; }
    else if (signal.status === "BLOCKED") { state = "blocked"; text = "Arrival unavailable · no direct or single-bounce path"; }
    else if (!signal.reflection && signal.status === "LOS") { state = "direct"; text = "Direct arrival · same direction as satellite"; }
    else {
      const arrival = reflectedArrival(scene?.rotation,snapshot,signal);
      if (!arrival) { state = "unavailable"; text = "Reflected arrival direction unavailable"; }
      else {
        state = "reflection";
        const {azimuth,elevation,separation} = arrival;
        Object.assign(this.arrivalReadout.dataset,{satellite:signal.id,azimuth:azimuth??"",elevation,separation,plotted:String(elevation>=0)});
        text = `◇ Reflected arrival · az ${azimuth===null?"— (vertical)":azimuth.toFixed(1)+"°"} · el ${elevation.toFixed(1)}°`;
        note = `Shortest single bounce · ${separation.toFixed(1)}° from satellite · ${signal.status === "LOS" ? "direct arrives first" : "earliest modeled path"}.`;
        if (elevation < 0) note += " Below local horizon; not plotted.";
        else {
          if (elevation < snapshot.mask) note += " Below satellite mask; retained.";
          const [x,y] = skyPosition(azimuth??0,elevation).map(value=>150+110*value), [sx,sy] = skyPosition(signal.azimuth,signal.elevation).map(value=>150+110*value);
          this.arrivalDiamond.setAttribute("transform",`translate(${x} ${y})`);
          for (const [name,value] of Object.entries({x1:sx,y1:sy,x2:x,y2:y})) this.arrivalLink.setAttribute(name,String(value));
          this.arrivalMarker.setAttribute("aria-label",`${signal.id} reflected arrival, ${azimuth===null?"vertical":"azimuth "+azimuth.toFixed(1)+" degrees"}, elevation ${elevation.toFixed(1)} degrees, ${separation.toFixed(1)} degrees from satellite`);
          this.arrivalMarker.setAttribute("visibility","visible"); this.arrivalMarker.setAttribute("aria-hidden","false");
        }
      }
    }
    this.arrivalReadout.dataset.state = state;
    this.arrivalReadout.textContent = text; this.arrivalNote.textContent = note; this.arrivalNote.hidden = !note;
  }
}
