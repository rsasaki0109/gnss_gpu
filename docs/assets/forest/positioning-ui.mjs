export function setupPositioning(capture){
  const $=id=>document.getElementById(id),panel=$("position-experiment"),status=$("position-status"),run=$("position-run"),cancel=$("position-cancel"),save=$("position-export"),table=$("position-results");
  let worker,result;
  const stop=()=>{worker?.terminate();worker=null;run.disabled=false;cancel.disabled=true;};
  cancel.addEventListener("click",()=>{stop();status.textContent="Cancelled. Previous result, if any, remains exportable.";panel.dataset.state="cancelled";});
  run.addEventListener("click",()=>{
    if(!$("position-seed").reportValidity())return;
    stop();result=null;save.disabled=true;table.replaceChildren();
    const input=capture();input.seed=Number($("position-seed").value);input.sigma=Number($("position-noise").value);input.mapBias=Number($("position-bias").value);
    panel.dataset.state="running";run.disabled=true;cancel.disabled=false;status.textContent="Generating synthetic GPS ranges…";
    try{
      worker=new Worker(new URL("./positioning-worker.mjs?v=position-1",import.meta.url),{type:"module"});
      const failure=message=>{stop();panel.dataset.state="error";status.textContent=message;};
      worker.onerror=()=>failure("Positioning worker failed. Reload or try another epoch.");
      worker.onmessage=({data})=>{
        if(data.error){failure(data.error);return;}
        if(data.progress){status.textContent=`${data.progress.method} · ${data.progress.step} m grid · ${data.progress.evaluations} candidates`;return;}
        result=data.result;stop();save.disabled=false;panel.dataset.state="ready";
        for(const r of result.results){const row=document.createElement("tr");row.dataset.method=r.method;for(const value of [r.method,r.horizontalErrorM===null?"unavailable":`${r.horizontalErrorM.toFixed(2)} m`,r.clockErrorM===null?"—":`${r.clockErrorM.toFixed(2)} m`,r.status+(r.boundary?" · boundary":"")]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}table.append(row);}
        status.textContent=`${result.observations.length} GPS ranges · +${result.context.elapsedSeconds.toFixed(1)} s snapshot · map bias ${input.mapBias} m. Errors use held-out generation truth; costs across methods are not comparable.`;
      };
      worker.postMessage(input);
    }catch(error){stop();panel.dataset.state="error";status.textContent=error.message;}
  });
  save.addEventListener("click",()=>{
    if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+"\n"],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download="gnss-forest-positioning.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  return ()=>{if(worker){stop();panel.dataset.state="cancelled";status.textContent="Stopped because scene became unavailable.";}run.disabled=true;};
}
