import {drawRoute} from "./route-plots.mjs?v=species-season-1";
export function setupPositioning(capture){
  const $=id=>document.getElementById(id),panel=$("position-experiment"),status=$("position-status"),run=$("position-run"),routeRun=$("position-route-run"),cancel=$("position-cancel"),save=$("position-export"),table=$("position-results"),routePanel=$("route-results"),satellite=$("route-satellite");
  let worker,result;
  const stop=()=>{worker?.terminate();worker=null;run.disabled=false;routeRun.disabled=false;cancel.disabled=true;};
  cancel.addEventListener("click",()=>{stop();status.textContent="Cancelled. Previous result, if any, remains exportable.";panel.dataset.state="cancelled";});
  const rows=values=>{for(const r of values){const row=document.createElement("tr");row.dataset.method=r.method;for(const value of [(r.seconds===undefined?"":`+${r.seconds}s · `)+r.method,r.horizontalErrorM===null?"unavailable":`${r.horizontalErrorM.toFixed(2)} m`,r.clockErrorM===null?"—":`${r.clockErrorM.toFixed(2)} m`,r.status+(r.boundary?" · boundary":"")]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}table.append(row);}};
  const start=mode=>{
    if(!$("position-seed").reportValidity())return;
    stop();result=null;save.disabled=true;table.replaceChildren();routePanel.hidden=true;satellite.replaceChildren();
    const input=capture();Object.assign(input,{mode,seed:Number($("position-seed").value),sigma:Number($("position-noise").value),mapBias:Number($("position-bias").value),stepSeconds:Number($("position-route-step").value)});
    panel.dataset.mode=mode;panel.dataset.state="running";run.disabled=true;routeRun.disabled=true;cancel.disabled=false;status.textContent=mode==="route"?`Preparing ${1200/input.stepSeconds+1} route epochs…`:"Generating synthetic GPS ranges…";
    try{
      worker=new Worker(new URL("./positioning-worker.mjs?v=species-season-1",import.meta.url),{type:"module"});
      const failure=message=>{stop();panel.dataset.state="error";status.textContent=message;};
      const active=worker;
      worker.onerror=()=>{if(worker===active)failure("Positioning worker failed. Reload or try another epoch.");};
      worker.onmessage=({data})=>{
        if(worker!==active)return;
        if(data.error){failure(data.error);return;}
        if(data.progress){const p=data.progress;status.textContent=p.route?`Epoch ${p.index+1}/${p.total} · +${p.seconds}s · ${p.phase}${p.method?` · ${p.method} / ${p.step} m`:""}`:`${p.method} · ${p.step} m grid · ${p.evaluations} candidates`;return;}
        result=data.result;stop();save.disabled=false;panel.dataset.state="ready";
        if(result.schema==="synthetic-forest-route-v1"){
          rows(result.cases.flatMap(c=>c.positioning.results.map(r=>({...r,seconds:c.seconds}))));
          const ids=[...new Set(result.cases.flatMap(c=>c.channels.map(s=>s.id)))].sort();
          for(const id of ids){const option=document.createElement("option");option.value=id;option.textContent=id;satellite.append(option);}
          // Open on a canopy-crossing channel, keeping every satellite selectable.
          satellite.value=ids.find(id=>result.cases.some(c=>c.channels.some(s=>s.id===id&&!s.directBlocked&&s.canopyPathM>0)))??ids[0]??"";
          routePanel.hidden=false;drawRoute(result,satellite.value);
          const failed=result.cases.filter(c=>!c.positioning.results.every(r=>r.status==="solution")).length;
          status.textContent=`${result.cases.length} epochs · both density profiles / all 5 GNSS · ${failed} epochs without a full solution. GPS positioning uses fixed σ; density is not a receiver-noise model.`;
        }else{
          rows(result.results);status.textContent=`${result.observations.length} GPS ranges · +${result.context.elapsedSeconds.toFixed(1)} s snapshot · map bias ${input.mapBias} m. Errors use held-out generation truth; costs across methods are not comparable.`;
        }
      };
      worker.postMessage(input);
    }catch(error){stop();panel.dataset.state="error";status.textContent=error.message;}
  };
  run.addEventListener("click",()=>start("snapshot"));routeRun.addEventListener("click",()=>start("route"));
  satellite.addEventListener("change",()=>{if(result?.schema==="synthetic-forest-route-v1")drawRoute(result,satellite.value);});
  save.addEventListener("click",()=>{
    if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=result.schema==="synthetic-forest-route-v1"?"gnss-forest-route.json":"gnss-forest-positioning.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  return ()=>{if(worker){stop();panel.dataset.state="cancelled";status.textContent="Stopped because scene became unavailable.";}run.disabled=true;routeRun.disabled=true;};
}
