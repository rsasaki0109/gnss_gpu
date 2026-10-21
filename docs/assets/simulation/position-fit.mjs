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

export class PositionFit {
  constructor(root,legend) {
    this.root = root; this.legend = legend;
    this.line = root.querySelector("line"); this.marker = root.querySelector("polygon"); this.label = root.querySelector("text");
  }

  update(snapshot,view,options) {
    const result = positionFit(snapshot,view,options), active = Boolean(result.fitPoint);
    this.root.toggleAttribute("hidden",!active); this.legend.hidden = !active;
    for (const key of Object.keys(this.root.dataset)) delete this.root.dataset[key];
    this.root.dataset.state = result.state;
    if (!active) {
      this.root.removeAttribute("viewBox"); this.root.setAttribute("aria-label","Geometric fit position unavailable");
      for (const name of ["x1","y1","x2","y2"]) this.line.removeAttribute(name);
      this.marker.removeAttribute("points"); this.label.removeAttribute("x"); this.label.removeAttribute("y");
      this.legend.removeAttribute("title"); return;
    }
    const [x,y] = result.fitPoint, [rx,ry] = result.referencePoint;
    this.root.setAttribute("viewBox",`0 0 ${view.width} ${view.height}`);
    Object.assign(this.root.dataset,{epoch:snapshot.epoch,antenna:snapshot.antenna,mask:snapshot.mask,
      position:result.position.join(","),reference:result.receiver.join(","),horizontal:result.horizontal,spatial:result.spatial,up:result.up,
      viewTarget:view.target.join(","),viewRight:view.right.join(","),viewUp:view.up.join(","),viewScale:view.scale});
    for (const [name,value] of Object.entries({x1:rx,y1:ry,x2:x,y2:y})) this.line.setAttribute(name,String(value));
    this.line.setAttribute("visibility",result.state === "coincident" ? "hidden" : "visible");
    this.marker.setAttribute("points",`${x},${y-7} ${x+7},${y} ${x},${y+7} ${x-7},${y}`);
    const visible = result.state !== "offscreen";
    this.marker.setAttribute("visibility",visible ? "visible" : "hidden"); this.label.setAttribute("visibility",visible ? "visible" : "hidden");
    // Only the label has a screen margin. The full 3D position is never clamped.
    this.label.setAttribute("x",String(Math.max(8,Math.min(view.width-36,x+12))));
    this.label.setAttribute("y",String(Math.max(16,Math.min(view.height-8,y-10))));
    const description = `Stored geometric fit: horizontal ${result.horizontal} m, scene up ${result.up} m, spatial displacement ${result.spatial} m${visible ? "" : "; position outside view"}. Diagnostic overlay; dashed line is displacement.`;
    this.root.setAttribute("aria-label",description); this.legend.title = description;
    this.legend.querySelector("span").textContent = visible ? "Fit²" : "Fit² ↗";
  }
}
