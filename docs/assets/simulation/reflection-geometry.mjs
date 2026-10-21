// Inspect one already-traced reflection. This module never creates paths or
// changes the trace's excess range, satellite state or positioning solution.
const vector = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
const sub = (a,b) => a.map((value,i)=>value-b[i]);
const dot = (a,b) => a.reduce((sum,value,i)=>sum+value*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function unit(value) {
  const length = Math.hypot(...value);
  return length > 0 && Number.isFinite(length) ? value.map(v=>v/length) : null;
}
const angle = (a,b) => Math.atan2(Math.hypot(...cross(a,b)),dot(a,b))*180/Math.PI;
const normalAngle = (a,n) => Math.atan2(Math.hypot(...cross(a,n)),Math.abs(dot(a,n)))*180/Math.PI;

export function reflectionGeometry(snapshot, signal) {
  const path = signal?.reflection;
  if (!path || signal.status === "BLOCKED" || !vector(snapshot?.receiver) || !vector(signal.source) || !vector(path.point) || path.vertices?.length !== 3 || !path.vertices.every(vector)) return null;
  const [a,b,c] = path.vertices;
  let normal = unit(cross(sub(b,a),sub(c,a)));
  const sourceLeg = sub(signal.source,path.point), receiverLeg = sub(snapshot.receiver,path.point);
  const fromSurface = unit(sourceLeg), toReceiver = unit(receiverLeg);
  if (!normal || !fromSurface || !toReceiver) return null;
  if (dot(normal,toReceiver) < 0) normal = normal.map(v=>-v);
  // The incoming propagation vector is -fromSurface. Reflect it about the
  // original triangle normal and compare with the actual outgoing vector.
  const predicted = fromSurface.map((value,i)=>2*dot(fromSurface,normal)*normal[i]-value);
  const incidence = normalAngle(fromSurface,normal), reflection = normalAngle(toReceiver,normal), mismatch = angle(predicted,toReceiver);
  let tangent = unit(cross(normal,cross(toReceiver,normal)));
  if (!tangent) {
    const axis = [0,1,2].sort((i,j)=>Math.abs(normal[i])-Math.abs(normal[j]))[0];
    tangent = unit(cross(normal,[0,1,2].map(i=>i===axis?1:0)));
  }
  // Project normalized directions into the normal/receiver plane. Do not
  // impose mirror symmetry: a non-coplanar input must retain its 3D mismatch.
  const project = direction => [dot(direction,tangent),dot(direction,normal)];
  return {sourceLeg:Math.hypot(...sourceLeg),receiverLeg:Math.hypot(...receiverLeg),incidence,reflection,mismatch,incident:project(fromSurface),outgoing:project(toReceiver)};
}

const SVG = "http://www.w3.org/2000/svg";
function element(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG,tag);
  for (const [name,value] of Object.entries(attributes)) node.setAttribute(name,String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export class ReflectionGeometry {
  constructor(root) {
    this.root = root;
    const find = id => root.querySelector(`#${id}`);
    this.label = find("reflection-label"); this.status = find("reflection-status");
    this.results = find("reflection-results"); this.sketch = find("reflection-sketch"); this.caption = find("reflection-caption");
    this.values = Object.fromEntries(["source-leg","receiver-leg","extra","incidence","angle","mismatch"].map(name=>[name,find(`reflection-${name}`)]));
  }

  update(snapshot, selected, {ready,error}) {
    if (this.snapshot === snapshot && this.selected === selected && this.ready === ready && this.error === error) return;
    this.snapshot = snapshot; this.selected = selected; this.ready = ready; this.error = error;
    this.results.hidden = true; this.sketch.replaceChildren();
    for (const key of Object.keys(this.root.dataset)) delete this.root.dataset[key];
    for (const node of Object.values(this.values)) { node.textContent = "—"; node.removeAttribute("data-value"); node.removeAttribute("title"); }
    const signal = snapshot?.satellites.find(item=>item.id===selected);
    const message = (state,text) => { this.root.dataset.state = state; this.status.textContent = text; this.label.textContent = ready && signal ? signal.id : state === "pending" ? "Updating…" : "Selected shortest path"; };
    if (error) { message("unavailable","Reflection geometry unavailable · simulation stopped."); return; }
    if (!snapshot) { message("loading","Preparing reflection geometry…"); return; }
    if (!ready) { message("pending","Waiting for the selected pose and receiver settings to finish tracing…"); return; }
    if (!signal) { message("unselected","Select a GPS satellite to inspect its reflection."); return; }
    if (!signal.reflection || signal.status === "BLOCKED") { message(signal.status === "LOS"?"direct":"blocked",`${signal.id} · ${signal.status === "LOS" ? "direct LOS; no modeled single-bounce path" : "no direct or single-bounce path"}.`); return; }
    const geometry = reflectionGeometry(snapshot,signal), extra = signal.reflection.extra;
    if (!geometry || !Number.isFinite(extra) || extra < 0) { message("unavailable","The traced reflection has no usable geometry details."); return; }
    message("ready",`${signal.id} · shortest single bounce · ${signal.status === "LOS" ? "additional path; direct arrives first" : "earliest modeled path"}.`);
    Object.assign(this.root.dataset,{geometrySatellite:signal.id,epoch:snapshot.epoch,antenna:snapshot.antenna,mask:snapshot.mask});
    const set = (name,value,text,units) => {
      const node = this.values[name]; node.dataset.value = String(value); node.textContent = text; node.title = `${value} ${units}`;
    };
    set("source-leg",geometry.sourceLeg,`${(geometry.sourceLeg/1000).toLocaleString("en-US",{minimumFractionDigits:3,maximumFractionDigits:3})} km`,"m");
    set("receiver-leg",geometry.receiverLeg,`${geometry.receiverLeg.toFixed(3)} m`,"m");
    set("extra",extra,`+${extra.toFixed(3)} m`,"m");
    set("incidence",geometry.incidence,`${geometry.incidence.toFixed(3)}°`,"degrees");
    set("angle",geometry.reflection,`${geometry.reflection.toFixed(3)}°`,"degrees");
    set("mismatch",geometry.mismatch,geometry.mismatch<.001?"<0.001°":`${geometry.mismatch.toFixed(3)}°`,"degrees");
    this.caption.textContent = `Original triangle ${signal.reflection.triangle} · angles from its surface normal. Normal/receiver plane projection; lengths are not drawn to scale. 3D mismatch checks the full reflected direction.`;
    this.draw(signal.id,geometry); this.results.hidden = false;
  }

  draw(id, geometry) {
    this.sketch.setAttribute("aria-label",`${id} shortest single-bounce reflection: incidence ${geometry.incidence.toFixed(3)} degrees, reflection ${geometry.reflection.toFixed(3)} degrees, 3D mismatch ${geometry.mismatch} degrees. Projected directions; distances are not to scale.`);
    const defs = element("defs");
    for (const kind of ["incident","outgoing"]) {
      const marker = element("marker",{id:`geometry-${kind}-arrow`,viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto",markerUnits:"strokeWidth"});
      marker.append(element("path",{d:"M0 0L10 5L0 10Z",class:`geometry-${kind}-head`})); defs.append(marker);
    }
    this.sketch.append(defs,element("path",{d:"M30 140H290 M40 140L30 151 M65 140L55 151 M90 140L80 151 M115 140L105 151 M140 140L130 151 M165 140L155 151 M190 140L180 151 M215 140L205 151 M240 140L230 151 M265 140L255 151 M290 140L280 151",class:"geometry-surface"}),element("path",{d:"M160 140V25",class:"geometry-normal"}),element("text",{x:166,y:29,class:"geometry-normal-label"},"Surface normal"));
    for (const [kind,direction] of [["incident",geometry.incident],["outgoing",geometry.outgoing]]) {
      const x = 160+100*direction[0], y = 140-100*direction[1];
      this.sketch.append(element("line",{x1:kind==="incident"?x:160,y1:kind==="incident"?y:140,x2:kind==="incident"?160:x,y2:kind==="incident"?140:y,class:`geometry-${kind}`,"marker-end":`url(#geometry-${kind}-arrow)`}));
      const theta = Math.atan2(direction[0],direction[1]), radius = kind==="incident"?30:43;
      this.sketch.append(element("path",{d:`M160 ${140-radius}A${radius} ${radius} 0 0 ${theta>=0?1:0} ${160+radius*Math.sin(theta)} ${140-radius*Math.cos(theta)}`,class:`geometry-arc geometry-${kind}`}));
    }
    this.sketch.append(element("circle",{cx:160,cy:140,r:3,class:"geometry-contact"}),element("text",{x:24,y:173,class:"geometry-incident-label"},"Satellite → surface"),element("text",{x:177,y:173,class:"geometry-outgoing-label"},"Surface → receiver"));
  }
}
