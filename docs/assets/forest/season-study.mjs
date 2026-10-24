import { traceForest, FOREST_SEASONS } from "./model.mjs?v=extinction-reference-1";
import { GroundReflectionModel } from "./ground-reflections.mjs?v=extinction-reference-1";

// Identical placement, wood, crown components and density. Only the existing
// binary phenology rule changes; no synthetic tree regrowth or new noise.
export function forestAtSeason(forest,season){
  if(!FOREST_SEASONS.includes(season))throw new RangeError("Invalid comparison season");
  return {...forest,season,trees:forest.trees.map(tree=>{
    if(!["broadleaf","conifer"].includes(tree.species))throw new RangeError("Unsupported tree type");
    return {...tree,leafRetention:season==="leaf-off"&&tree.species==="broadleaf"?0:1};
  })};
}

export function compareForestSeasons(input){
  const captured=structuredClone(input),{forest,receiver,signals,selected,options}=captured;
  if(new Set(signals.map(signal=>signal.id)).size!==signals.length)throw new RangeError("Duplicate comparison satellites");
  if(selected!=null&&!signals.some(signal=>signal.id===selected))throw new RangeError("Selected satellite was not captured");
  const cases=FOREST_SEASONS.map(season=>{
    const variant=forestAtSeason(forest,season);
    const direct=signals.map(signal=>{
      const delta=signal.sourceENU.map((v,i)=>v-receiver[i]),distance=Math.hypot(...delta);
      if(!Number.isFinite(distance)||distance<=0)throw new RangeError("Invalid comparison source");
      return {id:signal.id,trace:traceForest(variant,receiver,delta.map(v=>v/distance),{...options,maximum:Math.min(400,distance)})};
    });
    const source=signals.find(signal=>signal.id===selected);
    const ground=source?new GroundReflectionModel(variant).paths(receiver,source.sourceENU,options):null;
    return {season,direct,ground};
  });
  const [on,off]=cases,rows=on.direct.map(a=>{
    const b=off.direct.find(b=>b.id===a.id);
    return {id:a.id,leafOnLossDb:a.trace.canopyLoss,leafOffLossDb:b.trace.canopyLoss,
      lossChangeDb:b.trace.canopyLoss-a.trace.canopyLoss,
      leafOnStatus:a.trace.classification,leafOffStatus:b.trace.classification,
      directBlocked:a.trace.directRelativeDb===null||b.trace.directRelativeDb===null};
  }).sort((a,b)=>Math.abs(b.lossChangeDb)-Math.abs(a.lossChangeDb)||a.id.localeCompare(b.id));
  return {schema:"forest-season-comparison-v1",captured,cases,rows,
    assumptions:{pairedGeometry:true,sameSourceEndpoints:true,selectedGroundOnly:true,
      directMetric:"Canopy loss component only, including blocked geometric alternatives; not total received power or C/N0.",
      phenology:"Existing binary synthetic broadleaf leaf retention; evergreen foliage and all wood retained. No weather, growth, receiver tracking or positioning simulation.",
      reference:"In the extinction reference, leaf-off maple retains out-of-leaf extinction; fixed literature anchors are not site or GNSS calibration."}};
}

export function seasonStudyKey({forest,receiver,signals,selected,options,context}){
  // Display season is excluded: the study already contains both seasons.
  return JSON.stringify([forest.version,forest.seed,forest.count,forest.stand,receiver,signals.map(s=>[s.id,s.sourceENU]),selected,options,context]);
}

export function renderSeasonStudy(root,result){
  const doc=root.ownerDocument,node=(tag,text)=>{const e=doc.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
  const {captured,rows,cases}=result,context=captured.context;
  root.dataset.state="ready";root.dataset.selected=captured.selected??"";root.dataset.seconds=String(context.elapsedSeconds);
  root.querySelector("[data-season-context]").textContent=`Captured +${context.elapsedSeconds.toFixed(1)} s · ${captured.forest.count} trees · ${captured.forest.stand} · ${captured.options.attenuationModel} / ${captured.options.densityProfile}`;
  root.querySelector("[data-season-selected]").textContent=`${captured.selected??"No selected satellite"} · canopy component`;
  const body=root.querySelector("tbody");body.replaceChildren();
  for(const row of rows){
    const tr=node("tr");tr.dataset.comparisonSignal=row.id;
    if(row.id===captured.selected)tr.dataset.selected="true";
    for(const value of [row.id,row.leafOnLossDb.toFixed(3),row.leafOffLossDb.toFixed(3),`${row.lossChangeDb>=0?"+":""}${row.lossChangeDb.toFixed(3)}`,`${row.leafOnStatus} → ${row.leafOffStatus}`])tr.append(node("td",value));
    if(row.directBlocked)tr.setAttribute("data-blocked","true");body.append(tr);
  }
  const svg=root.querySelector("svg"),ns="http://www.w3.org/2000/svg";svg.replaceChildren();svg.setAttribute("viewBox","0 0 420 190");svg.setAttribute("aria-label",`${captured.selected??"Selected satellite"} canopy loss comparison, dB`);
  const selected=rows.find(row=>row.id===captured.selected),maximum=Math.max(1,selected?.leafOnLossDb??0,selected?.leafOffLossDb??0)*1.15;
  const add=(tag,attrs,text)=>{const e=doc.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;svg.append(e);};
  add("line",{x1:40,x2:400,y1:145,y2:145,stroke:"#52766b"});
  if(selected)for(const [i,value]of [selected.leafOnLossDb,selected.leafOffLossDb].entries()){
    const x=95+i*170,height=value/maximum*105,color=i?"#ffcf68":"#91f8da";
    add("rect",{x,y:145-height,width:85,height,rx:4,fill:color,"data-season-bar":i?"leaf-off":"leaf-on"});
    add("text",{x:x+42,y:132-height,fill:color,"text-anchor":"middle","font-size":14},`${value.toFixed(3)} dB`);
    add("text",{x:x+42,y:170,fill:"#c9e6da","text-anchor":"middle","font-size":12},i?"Leaf-off / 落葉期":"Leaf-on / 着葉期");
  }
  const powers=cases.map(c=>c.ground?.paths[0]?.relativePowerDb??null),extras=cases.map(c=>c.ground?.paths[0]?.extra??null);
  root.querySelector("[data-season-ground]").textContent=cases.map((c,i)=>`${c.season}: ${powers[i]===null?"No valid ground path":`${powers[i].toFixed(3)} dB · +${extras[i].toFixed(3)} m`}`).join(" / ");
  root.querySelector("[data-season-change]").textContent=!selected?"No selected satellite":`${selected.lossChangeDb>=0?"+":""}${selected.lossChangeDb.toFixed(3)} dB · leaf-off minus leaf-on${selected.directBlocked?" · direct path blocked in both cases":""}`;
  root.querySelector("[data-season-model-note]").textContent=captured.options.attenuationModel==="itu-extinction-v1"?"Leaf-off retains reference bare-crown extinction. Evergreen foliage and all wood are unchanged; published anchors are not site-calibrated GNSS power.":"Linear mode assumes no broadleaf crown loss after leaf-off; wood/terrain remain. The extinction reference retains bare-crown extinction.";
  root.querySelector("[data-season-results]").hidden=false;
}
