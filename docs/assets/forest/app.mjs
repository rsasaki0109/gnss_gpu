import {attenuationDescription} from "./attenuation.mjs?v=extinction-reference-1";
import { generateForest,forestReceiver,traceForest,PHENOLOGY_ASSUMPTION } from "./model.mjs?v=extinction-reference-1";
import { ForestRenderer } from "./renderer.mjs?v=tree-causes-2";
import { GroundReflectionModel } from "./ground-reflections.mjs?v=extinction-reference-1";
import { setupPositioning } from "./positioning-ui.mjs?v=extinction-reference-1";
import {forestSignals} from "./signals.mjs?v=extinction-reference-1";
import { ForestLiveHistory, renderForestHistory } from "./live-history.mjs?v=live-history-1";
import { selectedTreeCauses, renderTreeCauses } from "./tree-causes.mjs?v=tree-causes-2";
import { compareForestSeasons, renderSeasonStudy, seasonStudyKey } from "./season-study.mjs?v=season-study-1";
import { skyLayout } from "../simulation/sky-plot.mjs?v=glonass-1";
const $=id=>document.getElementById(id),root=$("forest-lab"),canvas=$("forest-scene");
let scene,sourceHash,forest,renderer,signals=[],receiver,selected,playing=false,seconds=600,last=0,elapsed=0;
let groundModel,reflectionCache,reflectionPacket,completedSeconds=600;
let stopPositioning,focusedTree=null,treeAttribution,seasonStudy=null;
const liveHistory=new ForestLiveHistory();
const sceneOptions=()=>({seed:Number($("forest-seed").value),count:Number($("forest-count").value),stand:$("forest-stand").value,season:$("forest-season").value});
const options=()=>({specificLoss:Number($("specific-loss").value),foliage:$("attenuation-model").value==="itu-extinction-v1"?1:Number($("foliage").value),attenuationModel:$("attenuation-model").value,materialPreset:$("material-preset").value,densityProfile:$("density-profile").value});
let linearFoliage=1;
function attenuationControls(){
  const reference=$("attenuation-model").value==="itu-extinction-v1";
  for(const id of ["specific-loss","foliage"])$(id).disabled=reference;
  $("material-preset").disabled=!reference;
  $("attenuation-reference").textContent=reference?"Published anchors: maple 1.3 GHz / conifers 1.5 GHz · ballistic extinction only; GNSS transfer unvalidated":"Illustrative linear leaf-density loss · comparison mode";
}
const colors={CLEAR:"#85e4bc",CANOPY:"#efbc79",WOOD:"#ec8f91",TERRAIN:"#b3b7c7"};
function play(value){playing=value;if(!value)seconds=completedSeconds;$("forest-play").textContent=value?"Pause":"Play · 10×";$("forest-play").setAttribute("aria-pressed",String(value));last=0;}
function select(id){if(selected!==id)focusedTree=null;selected=id;render();}
function updateGroundReflection(){
  const signal=signals.find(s=>s.id===selected),assumptions={...options(),bounceLoss:Number($("bounce-loss").value)},key=JSON.stringify(assumptions);
  if(!signal){reflectionPacket=null;}
  else if(reflectionCache?.model!==groundModel||reflectionCache.signal!==signal||reflectionCache.key!==key){
    const started=performance.now(),result=groundModel.paths(receiver,signal.sourceENU,assumptions);
    reflectionPacket={...result,satellite:signal.id,week:scene.week,tow:scene.track[0][0]+completedSeconds,satelliteStateConvention:"direct-flight-time broadcast state held fixed across geometric alternatives; no reflected-flight-time orbit iteration"};
    reflectionCache={model:groundModel,signal,key};$("forest-ground-cost").textContent=`${(performance.now()-started).toFixed(1)} ms · selected-signal reflection query`;
  }
  const first=reflectionPacket?.paths[0],readout=$("forest-ground");
  readout.dataset.state=first?"ready":"unavailable";
  readout.dataset.paths=reflectionPacket?.paths.length??0;readout.dataset.blocked=reflectionPacket?.blocked.length??0;
  for(const name of ["extra","delayNs","powerDb"])delete readout.dataset[name];
  if(first){Object.assign(readout.dataset,{extra:first.extra,delayNs:first.delayNs,powerDb:first.relativePowerDb});readout.textContent=`Ground bounce · +${first.extra.toFixed(3)} m / ${first.delayNs.toFixed(2)} ns · ${first.relativePowerDb.toFixed(2)} dB relative path power`;}
  else readout.textContent="No valid single-bounce ground path for the selected signal";
  $("forest-ground-count").textContent=`${reflectionPacket?.paths.length??0} valid · ${reflectionPacket?.blocked.length??0} occluded candidates · shortest valid path shown`;
  $("forest-focus-reflection").disabled=!first;$("bounce-out").textContent=`${assumptions.bounceLoss.toFixed(1)} dB`;
  return first;
}
function seasonStudyInput(){return {forest,receiver,selected,signals:signals.map(s=>({id:s.id,sourceENU:s.sourceENU.slice()})),options:{...options(),bounceLoss:Number($("bounce-loss").value)},context:{week:scene.week,tow:scene.track[0][0]+completedSeconds,elapsedSeconds:completedSeconds,source:{file:"odaiba-scene.json.gz",sha256:sourceHash}}};}
function seasonStudyFreshness(){if(!seasonStudy)return;const stale=seasonStudyKey(seasonStudy.captured)!==seasonStudyKey(seasonStudyInput());$("forest-season-study").dataset.stale=String(stale);$("season-study-stale").hidden=!stale;}
function render(){
  const ground=updateGroundReflection(),signal=signals.find(s=>s.id===selected);
  liveHistory.append({seconds:completedSeconds,id:selected,modelKey:JSON.stringify({scene:{seed:forest.seed,count:forest.count,stand:forest.stand,season:forest.season},...options(),bounceLoss:Number($("bounce-loss").value)}),trace:signal?.trace,reflection:ground});
  treeAttribution=selectedTreeCauses(signal,ground,options(),$("forest-tree-scope").value);
  if(!treeAttribution.rows.some(row=>row.id===focusedTree))focusedTree=treeAttribution.rows[0]?.id??null;
  renderTreeCauses($("forest-tree-causes"),treeAttribution,focusedTree);
  renderer.setHighlightedTree(forest.trees.find(tree=>tree.id===focusedTree)??null);
  renderer.setReflectionTrail(liveHistory.trail());
  renderForestHistory($("forest-live-history"),liveHistory);
  renderer.draw(receiver,signals,selected,ground);$("forest-signals").replaceChildren();$("forest-sky").replaceChildren();
  const svg=(tag,attrs,text)=>{const el=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text)el.textContent=text;return el;};
  for(const r of [110,110*2/3,110/3])$("forest-sky").append(svg("circle",{cx:150,cy:150,r,fill:"none",stroke:"#365359"}));
  for(const [label,x,y]of [["N",150,23],["E",281,154],["S",150,287],["W",18,154]])$("forest-sky").append(svg("text",{x,y,fill:"#b6c9c5","text-anchor":"middle"},label));
  for(const p of skyLayout(signals.map(s=>({...s,status:s.trace.classification})))){
    const group=svg("g",{role:"button",tabindex:0,"aria-label":`${p.id}, azimuth ${p.azimuth.toFixed(1)}, elevation ${p.elevation.toFixed(1)}`,"aria-pressed":String(p.id===selected),"data-satellite":p.id});
    group.append(svg("circle",{cx:p.x,cy:p.y,r:p.id===selected?8:5,fill:colors[p.status],stroke:p.id===selected?"white":"none"}),svg("text",{x:p.labelX,y:p.labelY,fill:colors[p.status],"text-anchor":"middle","font-size":10},p.id));
    group.addEventListener("click",()=>select(p.id));group.addEventListener("keydown",e=>{if(["Enter"," "].includes(e.key)){e.preventDefault();select(p.id);}});$("forest-sky").append(group);
  }
  for(const s of signals){
    const tr=document.createElement("tr"),name=document.createElement("td"),button=document.createElement("button");
    button.textContent=s.id;button.dataset.channel=s.id;button.setAttribute("aria-pressed",String(s.id===selected));button.addEventListener("click",()=>select(s.id));name.append(button);tr.append(name);
    for(const value of [s.trace.classification,`${s.trace.canopyPath.toFixed(1)} m`,`${s.trace.canopyLoss.toFixed(1)} dB`]){const td=document.createElement("td");td.textContent=value;tr.append(td);}$("forest-signals").append(tr);
  }
  const s=signals.find(s=>s.id===selected);
  $("forest-selected").textContent=s?`${s.id} · az ${s.azimuth.toFixed(1)}° / el ${s.elevation.toFixed(1)}° · ${s.trace.crowns.length} crowns · ${s.trace.wood.length} wood intersections`:"No satellites above 10°";
  $("forest-power").textContent=!s?"—":s.trace.directRelativeDb===null?"Total direct power unavailable · wood/terrain model incomplete":`${s.trace.directRelativeDb.toFixed(2)} dB relative to unattenuated direct signal`;
  const tow=scene.track[0][0]+completedSeconds,time=new Date(Date.UTC(1980,0,6)+(scene.week*604800+tow)*1000).toISOString().slice(11,23);
  $("forest-time").textContent=`${time} GPST · +${completedSeconds.toFixed(1)} s`;$("forest-progress").value=completedSeconds;
  $("forest-stats").textContent=`${forest.count} trees · ${signals.length} satellites · ${signals.filter(s=>s.trace.classification==="CANOPY").length} canopy-only`;
  $("loss-out").textContent=options().attenuationModel==="itu-extinction-v1"?"reference coefficients":`${options().specificLoss.toFixed(2)} dB/m`;$("foliage-out").textContent=`${options().foliage.toFixed(2)}×`;
  seasonStudyFreshness();
  Object.assign(root.dataset,{state:"ready",seconds:completedSeconds,seed:forest.seed,count:forest.count,stand:forest.stand,season:forest.season,satellites:signals.length,selected:selected??""});
}
function compute(){
  const started=performance.now();completedSeconds=seconds;receiver=forestReceiver(completedSeconds);renderer.setFoliage(options().foliage);
  signals=forestSignals(scene,receiver,completedSeconds,$("forest-systems").value).map(({directionENU,...signal})=>({...signal,trace:traceForest(forest,receiver,directionENU,options())}));
  if(!signals.some(s=>s.id===selected)){selected=(signals.find(s=>s.trace.classification==="CANOPY")??signals[0])?.id;focusedTree=null;}
  $("forest-cost").textContent=`${(performance.now()-started).toFixed(1)} ms · CPU propagation + direct intersections`;render();
}
function fail(error){play(false);stopPositioning?.();root.dataset.state="error";$("forest-status").textContent=error.message;for(const input of document.querySelectorAll("[data-control]"))input.disabled=true;}
function download(){
  const value={schema:"synthetic-forest-snapshot-v2",simulationOnly:true,assumptions:{vegetationModel:attenuationDescription(options().attenuationModel,options().materialPreset),phenology:PHENOLOGY_ASSUMPTION,appearance:"deterministic botanical leaf/needle cutouts, fine branches, roots, litter, stones, fallen wood, understory, alpha-tested sunlight shadows and atmospheric horizon are display only, not additional RF geometry",canopy:"uniform or normalized radial/height density in ellipsoid volumes; overlapping densities add; structured chords use an analytic cubic integral",wood:"finite cylinders; no calibrated power model",terrain:"piecewise planar finite patch",attenuation:options().attenuationModel==="itu-extinction-v1"?"Ballistic extinction from published RET coefficients; not a full ITU/GNSS model":"illustrative linear density integral, not a fitted ITU/GNSS model",reflections:"selected satellite, finite terrain single bounce only; constant uncalibrated bounce loss, no coherent combination or polarization model",diffraction:false,receiverTracking:false},
    treeAttribution:structuredClone(treeAttribution),signalHistory:liveHistory.snapshot(),options:{...options(),maximumDirectRayM:400,elevationMaskDeg:10,receiverHeightM:1.6,routeDurationSeconds:1200},scene:forest,view:{...structuredClone(renderer.camera),lighting:renderer.lighting,signalMode:renderer.signalMode,followObserverOffsetM:{behind:6,above:1}},receiverENU:receiver,anchor:{originECEF:scene.origin,rotation:scene.rotation},week:scene.week,startTow:scene.track[0][0],elapsedSeconds:completedSeconds,tow:scene.track[0][0]+completedSeconds,source:{file:"odaiba-scene.json.gz",sha256:sourceHash},signals,selectedReflection:reflectionPacket};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"})),link=document.createElement("a");link.href=url;link.download="gnss-forest-snapshot.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
  const base=new URL("../plateau/",import.meta.url),response=await fetch(new URL("manifest.json",base),{cache:"no-cache"});if(!response.ok)throw new Error("Orbit manifest unavailable");
  const manifest=await response.json(),asset=manifest.scene;if(!/^[a-z0-9._-]+$/.test(asset.file)||asset.bytes>2e6||!Number.isInteger(asset.bytes))throw new Error("Invalid orbit descriptor");
  const response2=await fetch(new URL(`${asset.file}?sha256=${asset.sha256}`,base));if(!response2.ok)throw new Error("Orbit data unavailable");const bytes=await response2.arrayBuffer();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),b=>b.toString(16).padStart(2,"0")).join("");if(bytes.byteLength!==asset.bytes||hash!==asset.sha256)throw new Error("Orbit data checksum mismatch");
  scene=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();sourceHash=hash;
  forest=generateForest(sceneOptions());groundModel=new GroundReflectionModel(forest);renderer=new ForestRenderer(canvas,forest);compute();$("forest-status").textContent="Synthetic forest · real broadcast orbits";
  for(const input of document.querySelectorAll("[data-control]"))input.disabled=false;attenuationControls();
  $("forest-focus-reflection").disabled=!reflectionPacket?.paths.length;
  stopPositioning=setupPositioning(()=>{
    play(false);
    return {forest:structuredClone(forest),orbitScene:{week:scene.week,startTow:scene.track[0][0],origin:scene.origin,rotation:scene.rotation,navigation:scene.navigation},...options(),receiver:receiver.slice(),signals:signals.map(s=>({id:s.id,sourceENU:s.sourceENU.slice()})),context:{week:scene.week,tow:scene.track[0][0]+completedSeconds,elapsedSeconds:completedSeconds,source:{file:"odaiba-scene.json.gz",sha256:sourceHash}}};
  });
  $("season-study-run").addEventListener("click",async()=>{
    if(root.dataset.state!=="ready")return;
    play(false);const input=structuredClone(seasonStudyInput()),button=$("season-study-run"),panel=$("forest-season-study");
    seasonStudy=null;button.disabled=true;$("season-study-export").disabled=true;panel.dataset.state="running";panel.querySelector("[data-season-results]").hidden=true;$("season-study-stale").hidden=true;$("season-study-status").textContent="Comparing captured leaf-on / leaf-off geometry…";
    await new Promise(requestAnimationFrame);
    try{seasonStudy=compareForestSeasons(input);renderSeasonStudy(panel,seasonStudy);seasonStudyFreshness();$("season-study-export").disabled=false;$("season-study-status").textContent="Comparison complete · same receiver, source endpoints, trees and wood geometry";}
    catch(error){panel.dataset.state="error";$("season-study-status").textContent=error.message;}
    finally{button.disabled=root.dataset.state!=="ready";}
  });
  $("season-study-export").addEventListener("click",()=>{if(!seasonStudy)return;const url=URL.createObjectURL(new Blob([JSON.stringify(seasonStudy,null,2)+"\n"],{type:"application/json"})),link=document.createElement("a");link.href=url;link.download="gnss-forest-season-comparison.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  $("forest-play").addEventListener("click",()=>play(!playing));
  $("forest-progress").addEventListener("input",e=>{play(false);liveHistory.clear("Timeline moved; new history segment");seconds=Number(e.target.value);compute();});
  for(const id of ["specific-loss","foliage"])$(id).addEventListener("input",compute);
  $("attenuation-model").addEventListener("change",()=>{
    if($("attenuation-model").value==="itu-extinction-v1"){linearFoliage=Number($("foliage").value);$("foliage").value=1;}else $("foliage").value=linearFoliage;
    attenuationControls();compute();
  });
  $("material-preset").addEventListener("change",compute);
  $("bounce-loss").addEventListener("input",render);
  $("forest-focus-reflection").addEventListener("click",()=>{play(false);if(reflectionPacket?.paths[0])renderer.focusReflection(reflectionPacket.paths[0],receiver);});
  $("forest-systems").addEventListener("change",compute);$("density-profile").addEventListener("change",compute);
  for(const id of ["forest-count","forest-seed","forest-stand","forest-season"])$(id).addEventListener("change",()=>{if(!$("forest-seed").reportValidity())return;try{play(false);forest=generateForest(sceneOptions());groundModel=new GroundReflectionModel(forest);renderer.setForest(forest);compute();}catch(error){fail(error);}});
  $("forest-export").addEventListener("click",download);$("forest-zoom-in").addEventListener("click",()=>renderer.zoom(1.25));$("forest-zoom-out").addEventListener("click",()=>renderer.zoom(.8));
  const chooseOverlaySignal=event=>{if(root.dataset.state!=="ready")return;const id=event.target.closest("[data-signal-select]")?.dataset.signalSelect;if(id){event.preventDefault();select(id);}};
  $("forest-rays").addEventListener("click",chooseOverlaySignal);
  $("forest-rays").addEventListener("keydown",event=>{if(["Enter"," "].includes(event.key))chooseOverlaySignal(event);});
  $("forest-tree-scope").addEventListener("change",()=>{focusedTree=null;render();});
  $("forest-tree-causes").addEventListener("click",event=>{
    if(root.dataset.state!=="ready")return;
    const id=event.target.closest("[data-tree-select]")?.dataset.treeSelect,tree=forest.trees.find(tree=>tree.id===id);
    if(!tree||!treeAttribution.rows.some(row=>row.id===id))return;
    play(false);focusedTree=id;$("forest-path-mode").value="selected";renderer.signalMode="selected";renderer.focusTree(tree);render();
    canvas.scrollIntoView({block:"center",behavior:renderer.reducedMotion?"auto":"smooth"});
  });
  $("forest-path-mode").addEventListener("change",()=>renderer.setSignalMode($("forest-path-mode").value));
  $("forest-track-signal").addEventListener("click",()=>renderer.focusSignal(signals.find(s=>s.id===selected)));
  $("forest-light").addEventListener("change",()=>renderer.setAtmosphere($("forest-light").value));
  $("forest-trail-view").addEventListener("click",()=>renderer.trailView());
  $("forest-reset-view").addEventListener("click",()=>renderer.resetView());
  new ResizeObserver(()=>renderer.redraw()).observe(canvas);
  const frame=time=>{if(playing){if(last){seconds=Math.min(1200,seconds+Math.min(time-last,250)/100);elapsed+=time-last;if(elapsed>=100||seconds===1200){elapsed=0;compute();}}if(seconds===1200)play(false);}last=time;requestAnimationFrame(frame);};requestAnimationFrame(frame);
}
canvas.addEventListener("webglcontextlost",()=>fail(new Error("Forest graphics context lost. Reload to restore the 3D view.")));
init().catch(fail);
