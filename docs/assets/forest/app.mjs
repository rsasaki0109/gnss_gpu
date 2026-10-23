import { generateForest,forestReceiver,traceForest } from "./model.mjs?v=kernels-1";
import { ForestRenderer } from "./renderer.mjs?v=kernels-1";
import { GroundReflectionModel } from "./ground-reflections.mjs?v=kernels-1";
import { setupPositioning } from "./positioning-ui.mjs?v=kernels-1";
import { selectNavigation,receivedSatellite,sub,mul,norm,dot,geodetic,enuRotation } from "../simulation/orbit.mjs?v=glonass-1";
import { skyLayout } from "../simulation/sky-plot.mjs?v=glonass-1";
const $=id=>document.getElementById(id),root=$("forest-lab"),canvas=$("forest-scene");
let scene,sourceHash,forest,renderer,signals=[],receiver,selected,playing=false,seconds=600,last=0,elapsed=0;
let groundModel,reflectionCache,reflectionPacket,completedSeconds=600;
let stopPositioning;
const options=()=>({specificLoss:Number($("specific-loss").value),foliage:Number($("foliage").value)});
const colors={CLEAR:"#85e4bc",CANOPY:"#efbc79",WOOD:"#ec8f91",TERRAIN:"#b3b7c7"};
function play(value){playing=value;if(!value)seconds=completedSeconds;$("forest-play").textContent=value?"Pause":"Play · 10×";$("forest-play").setAttribute("aria-pressed",String(value));last=0;}
function select(id){selected=id;render();}
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
function render(){
  renderer.draw(receiver,signals,selected,updateGroundReflection());$("forest-signals").replaceChildren();$("forest-sky").replaceChildren();
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
  $("loss-out").textContent=`${options().specificLoss.toFixed(2)} dB/m`;$("foliage-out").textContent=`${options().foliage.toFixed(2)}×`;
  Object.assign(root.dataset,{state:"ready",seconds:completedSeconds,seed:forest.seed,count:forest.count,satellites:signals.length,selected:selected??""});
}
function compute(){
  const started=performance.now();completedSeconds=seconds;receiver=forestReceiver(completedSeconds);renderer.setFoliage(options().foliage);
  const ecef=scene.origin.map((v,i)=>v+scene.rotation.reduce((sum,axis,j)=>sum+axis[i]*receiver[j],0)),llh=geodetic(ecef),orientation=enuRotation(llh.lat,llh.lon),tow=scene.track[0][0]+seconds;
  signals=[];
  for(const record of selectNavigation(scene.navigation.filter(r=>$("forest-systems").value.includes(r.id[0])),tow,scene.week)){
    const state=receivedSatellite(record,tow,ecef,scene.week),delta=sub(state.position,ecef),direction=mul(delta,1/norm(delta));
    const elevation=Math.asin(Math.max(-1,Math.min(1,dot(orientation[2],direction))))*180/Math.PI;
    if(elevation<10)continue;
    signals.push({id:record.id,azimuth:(Math.atan2(dot(orientation[0],direction),dot(orientation[1],direction))*180/Math.PI+360)%360,elevation,sourceECEF:state.position,sourceENU:scene.rotation.map(axis=>dot(axis,sub(state.position,scene.origin))),trace:traceForest(forest,receiver,scene.rotation.map(axis=>dot(axis,direction)),options())});
  }
  if(!signals.some(s=>s.id===selected))selected=(signals.find(s=>s.trace.classification==="CANOPY")??signals[0])?.id;
  $("forest-cost").textContent=`${(performance.now()-started).toFixed(1)} ms · CPU propagation + direct intersections`;render();
}
function fail(error){play(false);stopPositioning?.();root.dataset.state="error";$("forest-status").textContent=error.message;for(const input of document.querySelectorAll("[data-control]"))input.disabled=true;}
function download(){
  const value={schema:"synthetic-forest-snapshot-v2",simulationOnly:true,assumptions:{canopy:"ellipsoid density volumes; overlapping densities add",wood:"finite cylinders; no calibrated power model",terrain:"piecewise planar finite patch",attenuation:"illustrative linear density integral, not a fitted ITU/GNSS model",reflections:"selected satellite, finite terrain single bounce only; constant uncalibrated bounce loss, no coherent combination or polarization model",diffraction:false,receiverTracking:false},
    options:{...options(),maximumDirectRayM:400,elevationMaskDeg:10,receiverHeightM:1.6,routeDurationSeconds:1200},scene:forest,receiverENU:receiver,anchor:{originECEF:scene.origin,rotation:scene.rotation},week:scene.week,startTow:scene.track[0][0],elapsedSeconds:completedSeconds,tow:scene.track[0][0]+completedSeconds,source:{file:"odaiba-scene.json.gz",sha256:sourceHash},signals,selectedReflection:reflectionPacket};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"})),link=document.createElement("a");link.href=url;link.download="gnss-forest-snapshot.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
  const base=new URL("../plateau/",import.meta.url),response=await fetch(new URL("manifest.json",base),{cache:"no-cache"});if(!response.ok)throw new Error("Orbit manifest unavailable");
  const manifest=await response.json(),asset=manifest.scene;if(!/^[a-z0-9._-]+$/.test(asset.file)||asset.bytes>2e6||!Number.isInteger(asset.bytes))throw new Error("Invalid orbit descriptor");
  const response2=await fetch(new URL(`${asset.file}?sha256=${asset.sha256}`,base));if(!response2.ok)throw new Error("Orbit data unavailable");const bytes=await response2.arrayBuffer();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),b=>b.toString(16).padStart(2,"0")).join("");if(bytes.byteLength!==asset.bytes||hash!==asset.sha256)throw new Error("Orbit data checksum mismatch");
  scene=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();sourceHash=hash;
  forest=generateForest();groundModel=new GroundReflectionModel(forest);renderer=new ForestRenderer(canvas,forest);compute();$("forest-status").textContent="Synthetic forest · real broadcast orbits";
  for(const input of document.querySelectorAll("[data-control]"))input.disabled=false;
  $("forest-focus-reflection").disabled=!reflectionPacket?.paths.length;
  stopPositioning=setupPositioning(()=>{
    play(false);
    return {forest:structuredClone(forest),receiver:receiver.slice(),signals:signals.map(s=>({id:s.id,sourceENU:s.sourceENU.slice()})),context:{week:scene.week,tow:scene.track[0][0]+completedSeconds,elapsedSeconds:completedSeconds,source:{file:"odaiba-scene.json.gz",sha256:sourceHash}}};
  });
  $("forest-play").addEventListener("click",()=>play(!playing));
  $("forest-progress").addEventListener("input",e=>{play(false);seconds=Number(e.target.value);compute();});
  for(const id of ["specific-loss","foliage"])$(id).addEventListener("input",compute);
  $("bounce-loss").addEventListener("input",render);
  $("forest-focus-reflection").addEventListener("click",()=>{play(false);if(reflectionPacket?.paths[0])renderer.focusReflection(reflectionPacket.paths[0],receiver);});
  $("forest-systems").addEventListener("change",compute);
  for(const id of ["forest-count","forest-seed"])$(id).addEventListener("change",()=>{if(!$("forest-seed").reportValidity())return;try{play(false);forest=generateForest({seed:Number($("forest-seed").value),count:Number($("forest-count").value)});groundModel=new GroundReflectionModel(forest);renderer.setForest(forest);compute();}catch(error){fail(error);}});
  $("forest-export").addEventListener("click",download);$("forest-zoom-in").addEventListener("click",()=>renderer.zoom(1.25));$("forest-zoom-out").addEventListener("click",()=>renderer.zoom(.8));
  $("forest-reset-view").addEventListener("click",()=>{Object.assign(renderer.camera,{yaw:-.7,pitch:.85,zoom:1,target:[0,0,5]});renderer.redraw();});
  new ResizeObserver(()=>renderer.redraw()).observe(canvas);
  const frame=time=>{if(playing){if(last){seconds=Math.min(1200,seconds+Math.min(time-last,250)/100);elapsed+=time-last;if(elapsed>=100||seconds===1200){elapsed=0;compute();}}if(seconds===1200)play(false);}last=time;requestAnimationFrame(frame);};requestAnimationFrame(frame);
}
canvas.addEventListener("webglcontextlost",()=>fail(new Error("Forest graphics context lost. Reload to restore the 3D view.")));
init().catch(fail);
