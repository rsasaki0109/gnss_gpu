import { summarizeRoute, routeGeoJSON } from "./route-summary.mjs?v=route-summary-1";
import { routeLayout, nearestRecordedEpoch } from "./route-navigation.mjs?v=route-1";
import { localPoint } from "./orbit.mjs?v=glonass-1";
import { analysisCSV } from "./route-analysis.mjs?v=route-summary-1";

export function download(name,text,type="text/csv;charset=utf-8") {
  const url=URL.createObjectURL(new Blob([text],{type})), a=document.createElement("a");
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export class AnalysisView {
  constructor(root,{settings,power,send,pause,seek}) {
    Object.assign(this,{root,settings,power,send,pause,seek});this.rows=[];this.id=0;this.state="idle";
    this.$=id=>root.querySelector(`#${id}`);this.canvas=this.$("analysis-map");this.ctx=this.canvas.getContext("2d");
    this.$("analysis-start").onclick=()=>this.start();
    this.$("analysis-pause").onclick=()=>{if(this.state==="paused"){this.pause();this.send({type:"analysis-resume"});}else this.send({type:"analysis-pause"});};
    this.$("analysis-stop").onclick=()=>this.send({type:"analysis-cancel"});
    this.$("analysis-csv").onclick=()=>download("gnss-route-analysis.csv",analysisCSV(this.rows,this.metadata));
    this.$("analysis-json").onclick=()=>download("gnss-route-analysis.json",JSON.stringify({schema:1,...this.metadata,state:this.state,summary:summarizeRoute(this.rows),completed:this.rows.length,total:this.total,results:this.rows},null,2),"application/json");
    this.$("analysis-geojson").onclick=()=>download("gnss-route-analysis.geojson",JSON.stringify(routeGeoJSON(this.rows,{...this.metadata,state:this.state,completed:this.rows.length,total:this.total})),"application/geo+json");
    this.$("analysis-position").oninput=()=>{this.selected=Number(this.$("analysis-position").value);this.draw();};
    this.$("analysis-jump").onclick=()=>{const row=this.rows[this.selected??0];if(row)this.seek(row.epoch);};
    for(const id of ["analysis-condition","analysis-metric"])this.$(id).onchange=()=>this.draw();
    this.canvas.addEventListener("pointerdown",event=>{this.press={x:event.clientX,y:event.clientY,id:event.pointerId};});
    this.canvas.addEventListener("pointerup",event=>{
      const press=this.press;this.press=null;if(!press||press.id!==event.pointerId||Math.hypot(event.clientX-press.x,event.clientY-press.y)>8||!this.rows.length)return;
      const box=this.canvas.getBoundingClientRect(), points=this.rows.map(row=>this.layout.points[row.epoch]);
      const index=nearestRecordedEpoch(points,event.clientX-box.left,event.clientY-box.top,this.selected??0,24);
      if(index!==null){this.selected=index;this.draw();this.seek(this.rows[index].epoch);}
    });
    new ResizeObserver(()=>this.draw()).observe(this.canvas);
  }
  init(scene,assets) { this.scene=scene;this.assets=assets;this.positions=scene.track.map(row=>localPoint(row.slice(1),scene));this.$("analysis-start").disabled=false;this.updateSummary();this.draw(); }
  start() {
    if(!this.scene)return;this.pause();
    const a=this.settings(), radius=Number(this.$("analysis-range").value), start=radius?Math.max(0,a.epoch-radius):0,end=radius?Math.min(this.scene.track.length-1,a.epoch+radius):this.scene.track.length-1;
    const conditions=[{antenna:a.antenna,mask:a.mask,systems:a.systems},{antenna:Number(this.$("analysis-height").value),mask:Number(this.$("analysis-mask").value),systems:this.$("analysis-systems").value}];
    if(conditions.some(c=>!Number.isFinite(c.antenna)||c.antenna<0||c.antenna>20||!Number.isFinite(c.mask)||c.mask<5||c.mask>30)){this.$("analysis-status").textContent="Height must be 0–20 m and mask 5–30°.";return;}
    this.metadata={assets:this.assets,start,end,step:Number(this.$("analysis-step").value),conditions,power:this.power(),model:"Earliest-path zero-noise geometric fit; RF is display-only",sampling:"Original recorded poses only; unsampled epochs are not evaluated"};
    this.rows=[];this.selected=0;this.total=0;this.state="running";this.root.dataset.completed="0";
    this.$("analysis-settings").textContent=`A: ${a.systems} · +${a.antenna} m · ${a.mask}° / B: ${conditions[1].systems} · +${conditions[1].antenna} m · ${conditions[1].mask}° · RF: ${this.metadata.power.material}, ${this.metadata.power.polarization}, ${this.metadata.power.antenna}`;
    this.send({type:"analysis-start",id:++this.id,...this.metadata});this.controls();this.updateSummary();this.draw();
  }
  receive(data) {
    if(data.id!==this.id)return;
    this.state=data.state??"error";this.total=data.total??this.total;
    if(data.result){this.rows.push(data.result);if(this.rows.length===1)this.selected=0;}
    this.root.dataset.completed=this.rows.length;this.root.dataset.state=this.state;
    const average=this.rows.length?data.milliseconds/this.rows.length:0;
    this.$("analysis-status").textContent=data.error??`${this.state} · ${this.rows.length}/${this.total} sampled poses · ${average.toFixed(1)} ms/pose · ${data.traces??0} geometric traces`;
    this.controls();if(this.state!=="running"||this.rows.length===1||this.rows.length%20===0)this.updateSummary();this.draw();
  }
  updateSummary() {
    const stats=summarizeRoute(this.rows);
    const formatted=v=>v===null?"—":`${v.toFixed(2)} m`;
    this.$("analysis-statistics").replaceChildren(...stats.conditions.map((c,i)=>{
      const row=document.createElement("tr");
      for(const value of [i===0?"A":"B",`${c.available}/${stats.sampledPoses}`,formatted(c.median),formatted(c.p95),formatted(c.maximum)]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}
      return row;
    }));
    this.$("analysis-paired").textContent=`Paired poses: ${stats.paired.count} · median(A − B) ${formatted(stats.paired.median)}`;
  }
  controls() {
    const active=["running","paused"].includes(this.state);
    this.$("analysis-pause").disabled=!active;this.$("analysis-stop").disabled=!active;
    this.$("analysis-pause").textContent=this.state==="paused"?"Resume":"Pause";
    for(const id of ["analysis-csv","analysis-json","analysis-geojson","analysis-jump","analysis-position"])this.$(id).disabled=!this.rows.length;
    this.$("analysis-position").max=Math.max(0,this.rows.length-1);
  }
  draw() {
    if(!this.positions)return;
    const width=this.canvas.clientWidth,height=this.canvas.clientHeight;if(!width||!height)return;
    const dpr=Math.min(devicePixelRatio||1,2);if(this.canvas.width!==Math.round(width*dpr))this.canvas.width=Math.round(width*dpr);if(this.canvas.height!==Math.round(height*dpr))this.canvas.height=Math.round(height*dpr);
    const ctx=this.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    const dimensions=`${width},${height},${dpr}`;
    if(this.dimensions!==dimensions){this.layout=routeLayout(this.positions,width,height,16);this.dimensions=dimensions;this.layer=null;}
    const points=this.layout.points;
    const index=Number(this.$("analysis-condition").value), metric=this.$("analysis-metric").value;
    const key=`${dimensions},${index},${metric},${this.id}`;
    if(!this.layer||this.layerKey!==key){
      this.layer=document.createElement("canvas");this.layer.width=this.canvas.width;this.layer.height=this.canvas.height;
      this.layerContext=this.layer.getContext("2d");this.layerContext.setTransform(dpr,0,0,dpr,0,0);
      const base=this.layerContext;base.strokeStyle="#ffffff30";base.lineWidth=1;base.beginPath();points.forEach(([x,y],i)=>i?base.lineTo(x,y):base.moveTo(x,y));base.stroke();
      this.layerKey=key;this.applied=0;
    }
    const layer=this.layerContext;
    const value=c=>metric==="bias"?c.bias:(c.signals.length?c.counts.BLOCKED/c.signals.length:null);
    for(const row of this.rows.slice(this.applied)){const a=value(row.conditions[0]),b=value(row.conditions[1]);const v=index===2?(a===null||b===null?null:a-b):value(row.conditions[index]);
      layer.fillStyle=v===null?"#c5a4ff":index===2?(v<0?"#85e4bc":v===0?"#afbfca":"#ec8f91"):v<(metric==="bias"?5:.2)?"#85e4bc":v<(metric==="bias"?25:.5)?"#efbc79":"#ec8f91";
      layer.beginPath();layer.arc(...points[row.epoch],3,0,Math.PI*2);layer.fill();
    }
    this.applied=this.rows.length;ctx.drawImage(this.layer,0,0,width,height);
    const row=this.rows[this.selected??0];if(row){ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...points[row.epoch],6,0,Math.PI*2);ctx.stroke();
      this.$("analysis-position").value=this.selected??0;
      const formatted=c=>value(c)===null?"unavailable":metric==="bias"?`${value(c).toFixed(2)} m`:`${(value(c)*100).toFixed(1)}% blocked`;
      this.$("analysis-value").textContent=`Pose ${row.epoch+1} · GPST ${row.tow} s · A ${formatted(row.conditions[0])} / B ${formatted(row.conditions[1])}`;
    }else this.$("analysis-value").textContent="Grey route = not evaluated";
  }
}
