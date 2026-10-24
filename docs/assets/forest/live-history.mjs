import { seriesPath } from "./route-plots.mjs?v=extinction-reference-1";

// History of completed selected-signal computations, not a forward route scan.
export class ForestLiveHistory {
  constructor(){this.clear();}
  clear(reason="Start playback to record changes") {this.samples=[];this.key=null;this.reason=reason;}
  append({seconds,id,modelKey,trace,reflection}){
    if(!Number.isFinite(seconds))throw new RangeError("Invalid history time");
    const key=JSON.stringify([id??null,modelKey]),last=this.samples.at(-1);
    if(key!==this.key&&this.key!==null)this.clear("Satellite or signal settings changed");
    else if(key===this.key&&last&&(seconds<last.seconds||seconds-last.seconds>5))this.clear("Timeline moved; new history segment");
    this.key=key;
    if(this.samples.at(-1)?.seconds===seconds)this.samples.pop();
    const previous=this.samples.at(-1),status=trace?.classification??"UNAVAILABLE";
    const sample={seconds,id:id??null,status,
      directLossDb:trace&&trace.directRelativeDb!==null?trace.canopyLoss:null,
      canopyLossDb:trace?.canopyLoss??null,
      reflectionExtraM:reflection?.extra??null,reflectionPowerDb:reflection?.relativePowerDb??null,
      groundTriangle:reflection?.triangle??null,groundPoint:reflection?.point?.slice()??null,events:[]};
    if(previous){
      if(previous.status!==status)sample.events.push(`${previous.status} → ${status}`);
      if((previous.groundPoint!==null)!==(sample.groundPoint!==null))sample.events.push(sample.groundPoint?"Ground path gained":"Ground path lost");
      else if(sample.groundPoint&&previous.groundTriangle!==sample.groundTriangle)sample.events.push("Ground facet changed");
    }
    this.samples.push(sample);this.samples=this.samples.filter(s=>seconds-s.seconds<=120).slice(-600);
    return sample;
  }
  trail(){return this.samples.slice(-30).map(s=>s.groundPoint?.slice()??null);}
  snapshot(){return {schema:"forest-live-history-v1",windowSeconds:120,completedEpochsOnly:true,reason:this.reason,samples:structuredClone(this.samples)};}
}

const NS="http://www.w3.org/2000/svg";
export function renderForestHistory(root,history){
  const samples=history.samples,last=samples.at(-1),doc=root.ownerDocument;
  const node=(tag,attrs,text)=>{const el=doc.createElementNS(NS,tag);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,String(value));if(text!==undefined)el.textContent=text;return el;};
  root.dataset.samples=String(samples.length);root.dataset.selected=last?.id??"";root.dataset.seconds=String(last?.seconds??"");
  root.querySelector("[data-history-title]").textContent=last?.id?`${last.id} · signal changes / 信号の変化`:"Signal changes / 信号の変化";
  const loss=last?.directLossDb,extra=last?.reflectionExtraM;
  root.querySelector("[data-live-direct]").textContent=Number.isFinite(loss)?`${loss.toFixed(2)} dB`:last?.status==="UNAVAILABLE"?"Unavailable":"Blocked";
  root.querySelector("[data-live-reflection]").textContent=Number.isFinite(extra)?`+${extra.toFixed(3)} m`:"No ground path";
  const prior=samples.at(-2),delta=Number.isFinite(loss)&&Number.isFinite(prior?.directLossDb)?loss-prior.directLossDb:null;
  root.querySelector("[data-live-change]").textContent=delta===null?"—":`${delta>=0?"+":""}${delta.toFixed(2)} dB / last sample`;
  const events=samples.flatMap(s=>s.events.map(text=>({seconds:s.seconds,text}))).slice(-4);
  const list=root.querySelector("[data-history-events]");list.replaceChildren();
  for(const event of events.toReversed()){const item=doc.createElement("li");item.textContent=`+${event.seconds.toFixed(1)} s · ${event.text}`;list.append(item);}
  root.querySelector("[data-history-note]").textContent=samples.length<2?history.reason:"Recent simulated time · gaps = unavailable · dotted amber = past ground points";
  for(const [metric,title,unit,color,minimum]of [["directLossDb","Direct canopy loss","dB","#91f8da",4],["reflectionExtraM","Ground excess range","m","#ffcf68",1]]){
    const svg=root.querySelector(`[data-history-plot="${metric}"]`);svg.replaceChildren();svg.setAttribute("viewBox","0 0 420 170");svg.setAttribute("role","img");svg.setAttribute("aria-label",`${last?.id??"Selected satellite"} · ${title}, ${unit}`);
    const start=Math.max(0,Math.min(samples[0]?.seconds??0,(last?.seconds??0)-1)),end=Math.max(start+1,last?.seconds??1),maximum=Math.max(minimum,...samples.map(s=>s[metric]).filter(Number.isFinite))*1.08;
    const x=t=>43+(t-start)/(end-start)*359,y=value=>123-value/maximum*84;
    svg.append(node("text",{x:12,y:20,fill:"#c9e6da","font-size":11},`${title} · ${unit}`));
    for(const fraction of [0,.5,1]){const yy=y(maximum*fraction);svg.append(node("line",{x1:43,x2:402,y1:yy,y2:yy,stroke:"#29423c"}),node("text",{x:38,y:yy+4,fill:"#91aaa3","text-anchor":"end","font-size":9},(maximum*fraction).toFixed(metric==="reflectionExtraM"?2:1)));}
    for(const t of [start,(start+end)/2,end])svg.append(node("text",{x:x(t),y:144,fill:"#91aaa3","text-anchor":"middle","font-size":9},`+${t.toFixed(end-start<10?1:0)}s`));
    svg.append(node("path",{d:seriesPath(samples.map(s=>s.seconds),samples.map(s=>s[metric]),x,y),fill:"none",stroke:color,"stroke-width":2,"data-live-series":metric}));
    for(const [index,sample]of samples.entries()){if(!Number.isFinite(sample[metric])||(index%Math.max(1,Math.ceil(samples.length/120))!==0&&sample!==last))continue;const dot=node("circle",{cx:x(sample.seconds),cy:y(sample[metric]),r:sample===last?3.5:1.5,fill:color});dot.append(node("title",{},`+${sample.seconds.toFixed(2)} s · ${sample[metric].toFixed(3)} ${unit}`));svg.append(dot);}
    if(last)svg.append(node("line",{x1:x(last.seconds),x2:x(last.seconds),y1:30,y2:126,stroke:"#beded0","stroke-dasharray":"3 4",opacity:.5,"data-history-cursor":"true"}));
    svg.append(node("text",{x:43,y:163,fill:"#91aaa3","font-size":9},"Model values · not C/N0 or measured receiver power"));
  }
}
