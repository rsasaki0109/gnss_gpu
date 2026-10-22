import { GNSS_SYSTEMS, validateSystems, geometricSPP } from "./orbit.mjs?v=glonass-1";
import { signalMetrics, validatePower } from "./power-model.mjs?v=analysis-1";

export function evaluationEpochs(start,end,step,count) {
  if (![start,end,step,count].every(Number.isInteger) || start<0 || end<start || end>=count || step<1 || count>12410) throw new RangeError("Invalid evaluation interval");
  const epochs=[]; for(let i=start;i<=end;i+=step)epochs.push(i);
  if(epochs.at(-1)!==end)epochs.push(end);
  return epochs;
}
export function validateCondition(condition) {
  const {antenna,mask,systems}=condition;
  validateSystems(systems);
  if(!Number.isFinite(antenna)||antenna<0||antenna>20||!Number.isFinite(mask)||mask<5||mask>30)throw new RangeError("Invalid evaluation receiver");
  return {antenna,mask,systems};
}

// Reuse exact paths at an identical receiver. A wider satellite mask is traced
// once, then filtered in the original satellite order before the unchanged solve.
export function compareEpoch(simulation,epoch,conditions,power) {
  if(!Array.isArray(conditions)||conditions.length!==2)throw new RangeError("Two comparison conditions required");
  const models=conditions.map(validateCondition), groups=new Map(), snapshots=[];
  models.forEach((condition,i)=>{
    const key=condition.antenna;
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push({condition,i});
  });
  let traces=0;
  for(const group of groups.values()) {
    const systems=[...GNSS_SYSTEMS].filter(s=>group.some(({condition})=>condition.systems.includes(s))).join("");
    const mask=Math.min(...group.map(({condition})=>condition.mask));
    const base=simulation.compute({epoch,antenna:group[0].condition.antenna,mask,systems}); traces++;
    for(const {condition,i} of group) {
      const satellites=base.satellites.filter(s=>condition.systems.includes(s.id[0])&&s.elevation>=condition.mask);
      const counts={LOS:0,NLOS:0,BLOCKED:0};satellites.forEach(s=>counts[s.status]++);
      const fit=geometricSPP(base.receiver,satellites);
      const summary={...base,...condition,satellites};
      snapshots[i]={...condition,counts,bias:fit?.horizontal??null,signals:signalMetrics(summary,simulation.scene,power)};
    }
  }
  const row=simulation.scene.track[epoch];
  return {epoch,week:simulation.scene.week,tow:row[0],conditions:snapshots,traces};
}

// Cooperative scheduling keeps the one shared mesh/index resident. At most one
// evaluated pose is in flight; playback can pause this job between exact poses.
export class RouteEvaluation {
  constructor(simulation,emit,schedule=callback=>setTimeout(callback,0),now=()=>performance.now()) {
    Object.assign(this,{simulation,emit,schedule,now});this.job=null;
  }
  start(request) {
    const epochs=evaluationEpochs(request.start,request.end,request.step,this.simulation.scene.track.length);
    if(!Array.isArray(request.conditions)||request.conditions.length!==2)throw new RangeError("Two conditions required");
    const conditions=request.conditions.map(validateCondition),power=validatePower(request.power);
    const job={id:request.id,epochs,conditions,power,index:0,paused:false,cancelled:false,scheduled:false,milliseconds:0,traces:0};
    if(this.job)this.job.cancelled=true;
    this.job=job;this.state("running");this.queue(job);
  }
  state(state,extra={}) {
    const j=this.job;if(j)this.emit({type:"analysis",id:j.id,state,completed:j.index,total:j.epochs.length,milliseconds:j.milliseconds,traces:j.traces,...extra});
  }
  queue(job) { if(job.scheduled)return;job.scheduled=true;this.schedule(()=>{job.scheduled=false;this.step(job);}); }
  step(job) {
    if(this.job!==job||job.cancelled||job.paused)return;
    try {
      const started=this.now(),result=compareEpoch(this.simulation,job.epochs[job.index],job.conditions,job.power);
      job.milliseconds+=this.now()-started;job.traces+=result.traces;job.index++;
      this.state(job.index===job.epochs.length?"complete":"running",{result});
      if(job.index<job.epochs.length)this.queue(job);
    } catch(error) { job.cancelled=true;this.state("error",{error:error.message}); }
  }
  pause() { const j=this.job;if(!j||j.cancelled||j.index===j.epochs.length)return;j.paused=true;this.state("paused"); }
  resume() { const j=this.job;if(!j||j.cancelled||j.index===j.epochs.length)return;j.paused=false;this.state("running");this.queue(j); }
  cancel() { if(!this.job)return;this.job.cancelled=true;this.state("cancelled"); }
}

const cell=value=>value===null||value===undefined?"":String(value);
export function analysisCSV(rows,metadata) {
  const header=["epoch","gps_week","gps_tow_s","condition","systems","antenna_offset_m","mask_deg","horizontal_geometric_bias_m","satellite","path","elevation_deg","earliest_excess_m","modeled_relative_power_db","power_state","frequency_hz","material_assumption","polarization","antenna_pattern"];
  const lines=[header.join(",")];
  for(const row of rows)for(const [i,condition]of row.conditions.entries()) {
    const signals=condition.signals.length?condition.signals:[{}];
    for(const signal of signals)lines.push([row.epoch,row.week,row.tow,i===0?"A":"B",condition.systems,condition.antenna,condition.mask,condition.bias,signal.id,signal.status,signal.elevation,signal.extra,signal.power,signal.powerState,signal.frequency,metadata.power.material,metadata.power.polarization,metadata.power.antenna].map(cell).join(","));
  }
  return lines.join("\r\n")+"\r\n";
}
