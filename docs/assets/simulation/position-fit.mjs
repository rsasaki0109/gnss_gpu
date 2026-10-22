// Project the stored ideal-range solution; never solve, round or move it.
const vector = value => value?.length === 3 && Array.from(value).every(Number.isFinite);

export function positionFit(snapshot, view, {ready = true, error = false, focused = false} = {}) {
  if (error) return {state:"unavailable"};
  if (!snapshot) return {state:"loading"};
  if (!ready) return {state:"pending"};
  if (focused) return {state:"surface"};
  const fit = snapshot.spp?.all;
  if (!fit) return {state:"no-solution"};
  if (!vector(snapshot.receiver) || !vector(fit.position) || !vector(fit.offset) ||
      !Number.isFinite(fit.horizontal) || fit.horizontal < 0 || !Number.isFinite(fit.spatial) || fit.spatial < 0 ||
      !view || ![view.width,view.height,view.scale].every(value => Number.isFinite(value) && value > 0) ||
      ![view.target,view.right,view.up].every(vector)) return {state:"unavailable"};
  const project = point => {
    const delta = point.map((value,i) => value-view.target[i]);
    const dot = axis => delta.reduce((sum,value,i) => sum+value*axis[i],0);
    return [view.width/2+dot(view.right)*view.scale, view.height*.56-dot(view.up)*view.scale];
  };
  const referencePoint = project(snapshot.receiver), fitPoint = project(fit.position);
  if (![...referencePoint,...fitPoint].every(Number.isFinite)) return {state:"unavailable"};
  const inside = fitPoint[0] >= 0 && fitPoint[0] <= view.width && fitPoint[1] >= 0 && fitPoint[1] <= view.height;
  const coincident = fit.position.every((value,i) => value === snapshot.receiver[i]);
  return {state:!inside ? "offscreen" : coincident ? "coincident" : "ready", referencePoint,fitPoint,
    receiver:Array.from(snapshot.receiver),position:Array.from(fit.position),
    horizontal:fit.horizontal,spatial:fit.spatial,up:fit.offset[2]};
}

export function pinnedPositionFit(reference,snapshot,view,{enabled = false,ready = true,error = false,focused = false} = {}) {
  if (error) return {state:"unavailable"};
  if (!enabled) return {state:"closed"};
  if (!reference) return {state:"empty"};
  if (!snapshot) return {state:"loading"};
  if (!ready) return {state:"pending"};
  if (focused) return {state:"surface"};
  if (![reference.epoch,reference.week,reference.tow,snapshot.epoch,snapshot.week,snapshot.tow].every(Number.isFinite) ||
      reference.epoch !== snapshot.epoch || reference.week !== snapshot.week || reference.tow !== snapshot.tow) return {state:"different-pose"};
  if ((reference.systems ?? "G") !== (snapshot.systems ?? "G")) return {state:"different-systems"};
  if (reference.antenna === snapshot.antenna && reference.mask === snapshot.mask) return {state:"same-settings"};
  if (!snapshot.spp?.all) return {state:"no-current-solution"};
  // The pin is a detached summary in the same fixed scene frame. Its receiver
  // stays at the pinned height; do not rebase its bias onto the current receiver.
  return positionFit({receiver:reference.receiver,spp:{all:reference.fit}},view);
}

export class PositionFit {
  constructor(root,legend,{caption = "Fit²",name = "Stored geometric fit",radius = 7,labelBelow = false} = {}) {
    this.root = root; this.legend = legend; this.caption = caption; this.name = name; this.radius = radius; this.labelBelow = labelBelow;
    this.line = root.querySelector("line"); this.marker = root.querySelector("polygon"); this.label = root.querySelector("text"); this.origin = root.querySelector("circle");
  }

  update(snapshot,view,options) { this.render(positionFit(snapshot,view,options),snapshot,view); }
  updatePinned(reference,snapshot,view,options) { this.render(pinnedPositionFit(reference,snapshot,view,options),reference,view); }

  render(result,snapshot,view) {
    const active = Boolean(result.fitPoint);
    this.root.toggleAttribute("hidden",!active); this.legend.hidden = !active;
    for (const key of Object.keys(this.root.dataset)) delete this.root.dataset[key];
    this.root.dataset.state = result.state;
    if (!active) {
      this.root.removeAttribute("viewBox"); this.root.setAttribute("aria-label",`${this.name} unavailable`);
      for (const name of ["x1","y1","x2","y2"]) this.line.removeAttribute(name);
      this.marker.removeAttribute("points"); this.label.removeAttribute("x"); this.label.removeAttribute("y");
      if (this.origin) { this.origin.removeAttribute("cx"); this.origin.removeAttribute("cy"); }
      this.legend.removeAttribute("title"); return;
    }
    const [x,y] = result.fitPoint, [rx,ry] = result.referencePoint;
    this.root.setAttribute("viewBox",`0 0 ${view.width} ${view.height}`);
    Object.assign(this.root.dataset,{epoch:snapshot.epoch,antenna:snapshot.antenna,mask:snapshot.mask,
      position:result.position.join(","),reference:result.receiver.join(","),horizontal:result.horizontal,spatial:result.spatial,up:result.up,
      viewTarget:view.target.join(","),viewRight:view.right.join(","),viewUp:view.up.join(","),viewScale:view.scale});
    for (const [name,value] of Object.entries({x1:rx,y1:ry,x2:x,y2:y})) this.line.setAttribute(name,String(value));
    this.line.setAttribute("visibility",result.state === "coincident" ? "hidden" : "visible");
    const r = this.radius;
    this.marker.setAttribute("points",`${x},${y-r} ${x+r},${y} ${x},${y+r} ${x-r},${y}`);
    if (this.origin) { this.origin.setAttribute("cx",String(rx)); this.origin.setAttribute("cy",String(ry)); }
    const visible = result.state !== "offscreen";
    this.marker.setAttribute("visibility",visible ? "visible" : "hidden"); this.label.setAttribute("visibility",visible ? "visible" : "hidden");
    // Only the label has a screen margin. The full 3D position is never clamped.
    this.label.setAttribute("x",String(Math.max(8,Math.min(view.width-36,x+12))));
    this.label.setAttribute("y",String(Math.max(16,Math.min(view.height-8,y+(this.labelBelow ? 22 : -10)))));
    const description = `${this.name}: horizontal ${result.horizontal} m, scene up ${result.up} m, spatial displacement ${result.spatial} m${visible ? "" : "; position outside view"}. Diagnostic overlay; dashed line is displacement.`;
    this.root.setAttribute("aria-label",description); this.legend.title = description;
    this.legend.querySelector("span").textContent = this.caption+(visible ? "" : " ↗");
  }
}
