import { extinctionLoss, checkAttenuation } from "./attenuation.mjs?v=extinction-reference-1";
import { checkDensityProfile } from "./density.mjs?v=extinction-reference-1";
import { projectForestPoint, projectForestSegment } from "./signal-overlay.mjs?v=live-history-1";

const unionLength=spans=>{let length=0,end=-Infinity;for(const [a,b]of spans.toSorted((a,b)=>a[0]-b[0])){length+=Math.max(0,b-Math.max(a,end));end=Math.max(end,b);}return length;};

// Additive model components, not fractions of received power. Wood has no
// calibrated dB coefficient; its intersections stay separate from crown loss.
export function attributeForestTrees(legs,{specificLoss=.2,foliage=1,densityProfile="uniform",attenuationModel="linear-v1",materialPreset="maple-pine"}={}){
  if(!Number.isFinite(specificLoss)||specificLoss<0||specificLoss>2||!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid attenuation assumptions");
  checkAttenuation(attenuationModel,materialPreset,foliage);checkDensityProfile(densityProfile);
  const reference=attenuationModel==="itu-extinction-v1",trees=new Map();
  const row=id=>{if(!trees.has(id))trees.set(id,{id,crownLossDb:0,crownPathM:0,wood:[],legs:[]});return trees.get(id);};
  for(const {name,trace}of legs){
    const spans=new Map();
    for(const crown of trace.crowns){
      const tree=row(crown.tree);
      const integral=densityProfile==="uniform"?(crown.exit-crown.enter)*crown.density:crown.densityIntegral;
      tree.crownLossDb+=reference?extinctionLoss([crown],densityProfile):specificLoss*foliage*integral;
      if(!spans.has(crown.tree))spans.set(crown.tree,[]);spans.get(crown.tree).push([crown.enter,crown.exit]);
      if(!tree.legs.includes(name))tree.legs.push(name);
    }
    for(const [id,intervals]of spans)row(id).crownPathM+=unionLength(intervals);
    for(const hit of trace.wood){const tree=row(hit.tree);tree.wood.push({leg:name,part:hit.part,enter:hit.enter,exit:hit.exit});if(!tree.legs.includes(name))tree.legs.push(name);}
  }
  const rows=[...trees.values()].sort((a,b)=>b.crownLossDb-a.crownLossDb||b.wood.length-a.wood.length||a.id.localeCompare(b.id));
  const totalCanopyLossDb=rows.reduce((sum,row)=>sum+row.crownLossDb,0),expectedCanopyLossDb=legs.reduce((sum,leg)=>sum+leg.trace.canopyLoss,0);
  if(!Number.isFinite(totalCanopyLossDb)||!Number.isFinite(expectedCanopyLossDb)||Math.abs(totalCanopyLossDb-expectedCanopyLossDb)>1e-8*Math.max(1,Math.abs(expectedCanopyLossDb)))throw new RangeError("Attribution settings differ from the completed trace");
  return {rows,totalCanopyLossDb,expectedCanopyLossDb,woodIntersections:rows.reduce((sum,row)=>sum+row.wood.length,0)};
}

export function selectedTreeCauses(signal,ground,options,scope="direct"){
  if(!["direct","ground"].includes(scope))throw new RangeError("Unknown attribution scope");
  const available=scope==="direct"?Boolean(signal):Boolean(ground);
  const legs=!available?[]:scope==="direct"?[{name:"direct",trace:signal.trace}]:[{name:"receiver",trace:ground.legs.receiver},{name:"satellite",trace:ground.legs.satellite}];
  return {satellite:signal?.id??null,scope,available,directBlocked:Boolean(signal&&signal.trace.directRelativeDb===null),...attributeForestTrees(legs,options)};
}

export function renderTreeCauses(root,attribution,focused){
  const doc=root.ownerDocument,body=root.querySelector("tbody");body.replaceChildren();root.dataset.scope=attribution.scope;root.dataset.rows=String(attribution.rows.length);root.dataset.focused=focused??"";
  root.querySelector("[data-tree-title]").textContent=`${attribution.satellite??"Selected signal"} · loss by tree / 原因の木`;
  root.querySelector("[data-tree-total]").textContent=!attribution.available?"Unavailable":`${attribution.totalCanopyLossDb.toFixed(2)} dB · canopy subtotal`;
  root.querySelector("[data-tree-note]").textContent=!attribution.available?(attribution.scope==="ground"?"No valid ground path at this epoch":"Select a satellite"):
    attribution.scope==="ground"?"Both ground legs · canopy loss only; bounce loss and spreading are separate. Click a tree to inspect it.":
    attribution.directBlocked?"Direct path blocked: this is the canopy component only; total power is unavailable.":"Click a tree to inspect its model crown envelope. Crown path lengths can overlap between trees.";
  if(attribution.available&&!attribution.rows.length){const tr=doc.createElement("tr"),td=doc.createElement("td");td.colSpan=4;td.textContent="No crown or wood intersections on this path";tr.append(td);body.append(tr);}
  for(const tree of attribution.rows){
    const tr=doc.createElement("tr"),name=doc.createElement("td"),button=doc.createElement("button");button.textContent=tree.id;button.dataset.treeSelect=tree.id;button.setAttribute("aria-label",`Inspect tree ${tree.id}`);button.setAttribute("aria-pressed",String(tree.id===focused));name.append(button);tr.append(name);
    for(const value of [`${tree.crownPathM.toFixed(2)} m`,`${tree.crownLossDb.toFixed(3)} dB`,tree.wood.length?`${tree.wood.length} hits`:"—"]){const cell=doc.createElement("td");cell.textContent=value;tr.append(cell);}
    body.append(tr);
  }
}

export function drawTreeCauseOutline(svg,tree,frame,enabled=true){
  svg.dataset.highlightedTree=enabled&&tree?tree.id:"";
  if(!enabled||!tree)return;
  const ns="http://www.w3.org/2000/svg",group=svg.ownerDocument.createElementNS(ns,"g");group.setAttribute("data-tree-outline",tree.id);group.setAttribute("pointer-events","none");
  const path=d=>{const p=svg.ownerDocument.createElementNS(ns,"path");for(const [k,v]of Object.entries({d,fill:"none",stroke:"#8ecfff","stroke-width":1.5,"stroke-dasharray":"5 4",opacity:.85}))p.setAttribute(k,v);group.append(p);};
  for(const crown of tree.canopies??[tree.crown])for(const axes of [[0,1],[0,2],[1,2]]){
    let d="";const point=i=>{const p=crown.center.slice();p[axes[0]]+=Math.cos(i*Math.PI/12)*crown.radii[axes[0]];p[axes[1]]+=Math.sin(i*Math.PI/12)*crown.radii[axes[1]];return p;};
    for(let i=0;i<24;i++){const span=projectForestSegment(point(i),point(i+1),frame);if(span)d+=`M${span[0].join(",")}L${span[1].join(",")}`;}
    if(d)path(d);
  }
  const stem=projectForestSegment(tree.trunk.a,tree.trunk.b,frame);if(stem)path(`M${stem[0].join(",")}L${stem[1].join(",")}`);
  const p=projectForestPoint(tree.crown.center,frame);
  if(p&&p[0]>0&&p[0]<frame.width&&p[1]>0&&p[1]<frame.height){const text=svg.ownerDocument.createElementNS(ns,"text");for(const [k,v]of Object.entries({x:p[0]+8,y:p[1]-8,fill:"#b4dcff",stroke:"#081f2b","stroke-width":3,"paint-order":"stroke","font-size":13,"font-weight":700}))text.setAttribute(k,v);text.textContent=tree.id;group.append(text);}
  svg.prepend(group);
}
