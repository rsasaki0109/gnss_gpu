import { download } from "./analysis-view.mjs?v=route-summary-1";

function lowerBound(samples,time) {
  let lo=0,hi=samples.length;while(lo<hi){const mid=(lo+hi)>>>1;if(samples[mid][0]<time)lo=mid+1;else hi=mid;}return lo;
}
export function validateRecorded(data,receiver) {
  if(data.version!==1||data.receiver!==receiver||!Number.isInteger(data.week)||data.unit!=="dB-Hz"||!["declared","inferred-rtkconv"].includes(data.unitStatus)||!(data.cadence>0&&data.cadence<=1)||!data.signals||!data.codes)throw new Error("Unsupported recorded observations");
  let count=0;
  for(const [id,signal]of Object.entries(data.signals)){
    if(!/^[GECJR]\d{2}$/.test(id)||signal.code!==data.codes[id[0]]||!/^S[12][A-Z]$/.test(signal.code)||!Array.isArray(signal.samples))throw new Error("Invalid recorded signal identity");
    let last=-Infinity;
    for(const sample of signal.samples){if(sample.length!==2||!sample.every(Number.isFinite)||sample[0]<=last||sample[1]===0)throw new Error("Invalid recorded sample");last=sample[0];count++;}
  }
  if(!count||count>1000000)throw new Error("Invalid recorded sample count");
  return data;
}
export function recordedWindow(data,id,snapshot) {
  if(!snapshot||!Number.isInteger(snapshot.week)||!Number.isFinite(snapshot.tow)||snapshot.antenna!==0)return null;
  const signal=data.signals[id];if(!signal)return null;
  const time=(snapshot.week-data.week)*604800+snapshot.tow, start=lowerBound(signal.samples,time-120),end=lowerBound(signal.samples,time+.002000001);
  return {time,code:signal.code,samples:signal.samples.slice(start,end)};
}
export function nearestRecorded(data,id,snapshot,tolerance=.002) {
  if(!Number.isFinite(tolerance)||tolerance<0||tolerance>.002)throw new RangeError("Invalid observation tolerance");
  if(!snapshot||snapshot.antenna!==0)return null;
  const signal=data.signals[id];if(!signal)return null;
  const time=(snapshot.week-data.week)*604800+snapshot.tow,index=lowerBound(signal.samples,time);
  const candidates=[signal.samples[index-1],signal.samples[index]].filter(Boolean).sort((a,b)=>Math.abs(a[0]-time)-Math.abs(b[0]-time)||a[0]-b[0]);
  const sample=candidates[0];return sample&&Math.abs(sample[0]-time)<=tolerance+1e-9?{time:sample[0],value:sample[1],delta:sample[0]-time,code:signal.code}:null;
}
export function recordedCSV(data,id,snapshot) {
  const window=recordedWindow(data,id,snapshot),lines=["gps_week,gps_tow_s,receiver,satellite,rinex_code,recorded_strength_dbhz,unit_status,source_sha256"];
  for(const [time,value]of window?.samples??[]){const weekOffset=Math.floor(time/604800);lines.push([data.week+weekOffset,time-weekOffset*604800,data.receiver,id,window.code,value,data.unitStatus,data.source.sha256].join(","));}
  return lines.join("\r\n")+"\r\n";
}

export class RecordedStrength {
  constructor(root) {
    this.root=root;this.$=id=>root.querySelector(`#${id}`);this.cache=new Map();this.loading=new Map();
    root.addEventListener("toggle",()=>{if(root.open)this.render();});
    this.$("recorded-receiver").onchange=()=>this.render();
    this.$("recorded-export").onclick=()=>{const data=this.cache.get(this.$("recorded-receiver").value);if(data)download("gnss-recorded-strength.csv",recordedCSV(data,this.selected,this.snapshot));};
  }
  update(snapshot,selected,assets) {Object.assign(this,{snapshot,selected,assets});if(this.root.open)this.render();}
  async load(receiver) {
    if(this.cache.has(receiver))return;
    if(this.loading.has(receiver))return this.loading.get(receiver);
    const promise=(async()=>{
      if(!this.manifest){const response=await fetch(new URL("../plateau/strength-manifest.json",import.meta.url),{cache:"no-cache"});if(!response.ok)throw new Error(`Recorded manifest HTTP ${response.status}`);this.manifest=await response.json();}
      const manifest=this.manifest;
      if(manifest.version!==1||manifest.sceneSha256!==this.assets.data)throw new Error("Recorded data does not match this scene");
      const entry=manifest.receivers[receiver];
      if(!entry||entry.file!==`odaiba-strength-${receiver}.json.gz`||!(entry.bytes>0&&entry.bytes<2000000))throw new Error("Invalid recorded data descriptor");
      const url=new URL(`../plateau/${entry.file}`,import.meta.url);url.searchParams.set("sha256",entry.sha256);
      const response=await fetch(url);if(!response.ok)throw new Error(`Recorded data HTTP ${response.status}`);
      const bytes=await response.arrayBuffer(),hash=[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map(v=>v.toString(16).padStart(2,"0")).join("");
      if(bytes.byteLength!==entry.bytes||hash!==entry.sha256)throw new Error("Recorded data integrity check failed");
      const data=validateRecorded(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json(),receiver);
      if(data.unitStatus!==entry.unitStatus||data.source?.sha256!==entry.source.sha256)throw new Error("Recorded provenance mismatch");
      this.cache.set(receiver,data);
    })();this.loading.set(receiver,promise);
    try{await promise;}finally{this.loading.delete(receiver);}
  }
  async render() {
    const receiver=this.$("recorded-receiver").value,svg=this.$("recorded-plot"),output=this.$("recorded-value");
    svg.replaceChildren();this.$("recorded-export").disabled=true;delete this.root.dataset.matched;delete this.root.dataset.epoch;this.root.dataset.points="0";
    if(!this.snapshot){output.textContent="Waiting for a completed model pose…";return;}
    if(this.snapshot.antenna!==0){this.root.dataset.state="virtual-height";output.textContent="Recorded observations belong to the original receiver. Set height offset to 0 m to compare.";return;}
    if(!this.cache.has(receiver)){
      this.root.dataset.state="loading";output.textContent="Loading verified recorded observations…";
      try{await this.load(receiver);}catch(error){if(receiver===this.$("recorded-receiver").value){this.root.dataset.state="error";output.textContent=error.message;}return;}
      if(receiver!==this.$("recorded-receiver").value||!this.root.open)return;
      return this.render();
    }
    const data=this.cache.get(receiver),window=recordedWindow(data,this.selected,this.snapshot);
    this.root.dataset.state="ready";this.root.dataset.receiver=receiver;this.root.dataset.epoch=this.snapshot.epoch;
    if(!window){output.textContent=`${this.selected??"—"} · no exported first-band observation`;return;}
    const NS="http://www.w3.org/2000/svg",node=(tag,attrs,text)=>{const el=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);return el;};
    const values=window.samples.map(s=>s[1]),low=Math.min(10,...values),high=Math.max(60,...values);
    const x=time=>45+(time-window.time+120)/120*645,y=value=>14+(high-value)/(high-low)*110;
    for(const value of [low,(low+high)/2,high]){node("line",{x1:45,x2:690,y1:y(value),y2:y(value),stroke:"#ffffff20"});node("text",{x:39,y:y(value)+4,"text-anchor":"end",fill:"#a9b9b6","font-size":11},`${value}`);}
    for(const age of [120,60,0])node("text",{x:x(window.time-age),y:147,"text-anchor":age===0?"end":"start",fill:"#a9b9b6","font-size":11},age?`−${age} s`:"Completed model GPST");
    let path="",previous;
    for(const [time,value]of window.samples){path+=`${previous!==undefined&&time-previous<=data.cadence*1.5?"L":"M"}${x(time)},${y(value)} `;previous=time;}
    node("path",{d:path,fill:"none",stroke:"#91bfff","stroke-width":2,"data-recorded":this.selected});
    const nearest=nearestRecorded(data,this.selected,this.snapshot);this.root.dataset.matched=String(Boolean(nearest));this.root.dataset.points=window.samples.length;
    if(nearest)node("circle",{cx:x(nearest.time),cy:y(nearest.value),r:4,fill:"#91bfff"});
    output.textContent=`${this.selected} · ${window.code} · ${nearest?`${nearest.value} dB-Hz, Δt ${(nearest.delta*1000).toFixed(3)} ms`:"no observation within ±2 ms"} · ${data.unitStatus==="declared"?"declared units":"units inferred from RTKCONV"}`;
    svg.setAttribute("aria-label",`${receiver} recorded ${this.selected} ${window.code}, inferred dB-Hz, actual GPS timestamps`);
    this.$("recorded-export").disabled=!window.samples.length;
  }
}
