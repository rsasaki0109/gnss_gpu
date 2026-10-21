// Display projection of completed satellite directions. No propagation or tracing.
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
    this.svg.append(element("desc",{},"Geometric satellite directions above the completed elevation mask. NLOS points show satellite directions, not reflected arrival angles. Select with Enter or Space, or use the satellite list."));
    const grid = element("g",{"aria-hidden":"true",class:"sky-grid"});
    grid.append(element("circle",{cx:150,cy:150,r:110,class:"sky-horizon"}));
    this.mask = element("circle",{cx:150,cy:150,r:0,class:"sky-mask",visibility:"hidden"}); grid.append(this.mask);
    for (const elevation of [30,60]) grid.append(element("circle",{cx:150,cy:150,r:110*(90-elevation)/90,class:"sky-ring-grid"}));
    grid.append(element("path",{d:"M150 40V260 M40 150H260",class:"sky-axes"}));
    for (const [label,x,y] of [["N",150,20],["E",285,154],["S",150,288],["W",15,154]]) grid.append(element("text",{x,y,class:"sky-cardinal"},label));
    for (const [label,x,y] of [["30°",99,195],["60°",125,176]]) grid.append(element("text",{x,y,class:"sky-elevation"},label));
    grid.append(element("circle",{cx:150,cy:150,r:2,class:"sky-zenith"}));
    this.labels = element("g",{"aria-hidden":"true",class:"sky-labels"}); this.points = element("g",{class:"sky-points"});
    this.svg.append(grid,this.labels,this.points); root.append(this.svg);
    this.source = document.createElement("p"); this.source.className = "sky-source";
    this.selection = document.createElement("p"); this.selection.className = "sky-selection";
    root.append(this.source,this.selection);
  }

  update(snapshot, selected, {ready, error, playing, time}) {
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
    if (changed || this.selected !== selected || this.lastReady !== ready) {
      for (const button of this.points.children) {
        button.setAttribute("aria-pressed",String(button.dataset.skySatellite===selected)); button.setAttribute("aria-disabled",String(!ready)); button.setAttribute("tabindex",ready?"0":"-1");
      }
      for (const label of this.labels.children) label.setAttribute("data-selected",String(label.dataset.skyLabel===selected));
      this.selected = selected; this.lastReady = ready;
    }
    const source = error ? "Sky unavailable · simulation stopped" : !snapshot ? "Preparing GPS sky…" : `${ready ? "Completed" : playing ? "Live completed trace · updating" : "Previous trace · updating"} · ${snapshot.satellites.length} GPS · ${snapshot.mask}° mask · ${time} GPST`;
    if (this.source.textContent !== source) this.source.textContent = source;
    const signal = snapshot?.satellites.find(signal=>signal.id===selected);
    const description = signal ? `${signal.id} · az ${signal.azimuth.toFixed(1)}° · el ${signal.elevation.toFixed(1)}° · ${signal.status}` : "Select a satellite to inspect its direction.";
    if (this.selection.textContent !== description) this.selection.textContent = description;
  }
}
