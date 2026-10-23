const NS="http://www.w3.org/2000/svg";
export function routeSeries(result,id){
  const losses=["uniform","radial-height-v1"].map(profile=>({label:profile==="uniform"?"Uniform":"Radial + height",color:profile==="uniform"?"#85d9df":"#efbc79",values:result.cases.map(c=>{const r=c.channels.find(r=>r.id===id);return r&&!r.directBlocked?r.profiles[profile].canopyLossDb:null;})}));
  const paths=[{label:"Direct + ground",color:"#a7dfb2",values:result.cases.map(c=>c.channels.find(r=>r.id===id)?.feasiblePaths??null)}];
  const positions=["straight","robust","paths"].map((method,i)=>({label:method,color:["#b3b7c7","#85d9df","#efbc79"][i],values:result.cases.map(c=>c.positioning.results.find(r=>r.method===method)?.horizontalErrorM??null)}));
  return {times:result.cases.map(c=>c.seconds),losses,paths,positions};
}
// Nulls terminate a segment: unavailable observations are never drawn as zero
// or interpolated across a visibility/solver failure gap.
export function seriesPath(times,values,x,y){
  let open=false,d="";
  values.forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return;}d+=`${open?"L":"M"}${x(times[i]).toFixed(2)},${y(v).toFixed(2)} `;open=true;});return d.trim();
}
export function plotRoute(svg,times,series,title,unit){
  const node=(tag,attrs,text)=>{const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;};
  svg.replaceChildren();svg.setAttribute("viewBox","0 0 420 245");svg.setAttribute("role","img");svg.setAttribute("aria-label",title);
  svg.append(node("title",{},title));
  const max=Math.max(1,...series.flatMap(s=>s.values.filter(Number.isFinite))),last=Math.max(1,...times),x=t=>45+350*t/last,y=v=>180-130*v/max;
  svg.append(node("text",{x:15,y:22,fill:"#dcebe5","font-size":12},title));
  for(const fraction of [0,.5,1]){const yy=y(max*fraction);svg.append(node("line",{x1:45,x2:395,y1:yy,y2:yy,stroke:"#29423c"}),node("text",{x:39,y:yy+4,fill:"#91aaa3","text-anchor":"end","font-size":10},(max*fraction).toFixed(max>10?0:1)));}
  for(const t of [0,last/4,last/2,3*last/4,last])svg.append(node("text",{x:x(t),y:198,fill:"#91aaa3","text-anchor":"middle","font-size":10},`${t}s`));
  series.forEach((s,index)=>{
    svg.append(node("path",{d:seriesPath(times,s.values,x,y),fill:"none",stroke:s.color,"stroke-width":2,"data-series":s.label}));
    s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;const circle=node("circle",{cx:x(times[i]),cy:y(v),r:3,fill:s.color});circle.append(node("title",{},`${times[i]} s · ${s.label}: ${v.toFixed(3)} ${unit}`));svg.append(circle);});
    svg.append(node("text",{x:45+index*116,y:220,fill:s.color,"font-size":10},s.label));
  });
  svg.append(node("text",{x:45,y:237,fill:"#91aaa3","font-size":9},`${unit} · gaps = unavailable`));
}
export function drawRoute(result,id){
  const series=routeSeries(result,id);
  plotRoute(document.getElementById("route-loss"),series.times,series.losses,`${id} · direct canopy attenuation`,"dB (not C/N0)");
  plotRoute(document.getElementById("route-paths"),series.times,series.paths,`${id} · feasible geometric paths`,"paths");
  plotRoute(document.getElementById("route-position"),series.times,series.positions,"Fixed-σ GPS positioning baseline","horizontal error / m");
}
