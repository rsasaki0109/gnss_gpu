import {generateForest,forestReceiver,terrainHeight,traceForest,randomSequence} from './model.mjs?v=extinction-reference-1';
import {forestSignals} from './signals.mjs?v=extinction-reference-1';
export const CONDITIONS=['open','forest','wet'];
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
export const SYSTEM_OPTIONS=['G','GECJR'];
// One receiver clock per GNSS system (the first letter of the satellite id);
// observations without an id share one clock. Inter-system offsets are
// therefore unknowns, as in a real multi-GNSS receiver: each extra system
// costs one satellite. With a single system the unknowns, arithmetic order
// and results are identical to the original 4-state solver.
const clockKey=o=>o.id?o.id[0]:'';
// Weighted least squares, ENU position plus receiver clock(s) in metres. The
// solver sees measurements only, never the generating receiver or noise draws.
export function solveRanges(observations){
  const systems=[...new Set(observations.map(clockKey))],n=3+Math.max(1,systems.length);
  if(observations.length<n)return {status:'insufficient-satellites',position:null};
  const x=Array(n).fill(0);
  for(let iteration=0;iteration<12;iteration++){
    const a=Array.from({length:n},()=>Array(n+1).fill(0));
    for(const o of observations){
      const c=3+systems.indexOf(clockKey(o)),r=distance(x.slice(0,3),o.sourceENU),h=[...x.slice(0,3).map((v,i)=>(v-o.sourceENU[i])/r),...Array(n-3).fill(0)],residual=o.range-r-x[c],w=1/o.sigma**2;
      h[c]=1;
      for(let j=0;j<n;j++){for(let k=0;k<n;k++)a[j][k]+=w*h[j]*h[k];a[j][n]+=w*h[j]*residual;}
    }
    const scale=Math.max(...a.map(row=>Math.max(...row.slice(0,n).map(Math.abs))));
    for(let j=0;j<n;j++){
      let pivot=j;for(let k=j+1;k<n;k++)if(Math.abs(a[k][j])>Math.abs(a[pivot][j]))pivot=k;
      if(!Number.isFinite(scale)||Math.abs(a[pivot][j])<scale*1e-10)return {status:'singular-geometry',position:null};
      [a[j],a[pivot]]=[a[pivot],a[j]];const d=a[j][j];for(let k=j;k<=n;k++)a[j][k]/=d;
      for(let k=0;k<n;k++)if(k!==j){const v=a[k][j];for(let l=j;l<=n;l++)a[k][l]-=v*a[j][l];}
    }
    const delta=a.map(row=>row[n]);delta.forEach((v,i)=>x[i]+=v);
    if(!x.every(Number.isFinite))return {status:'non-finite',position:null};
    if(Math.hypot(...delta)<1e-5){
      const fit={status:'solution',position:x.slice(0,3),clockM:x[3]};
      if(systems.length>1)fit.clocksM=Object.fromEntries(systems.map((s,i)=>[s,x[3+i]]));
      return fit;
    }
  }
  return {status:'no-convergence',position:null};
}
function normal(seed,seconds,id){
  let key=(seed^Math.imul(seconds,2654435761))>>>0;
  for(const char of id)key=(Math.imul(key,16777619)^char.charCodeAt(0))>>>0;
  const random=randomSequence(key);
  return Math.sqrt(-2*Math.log(Math.max(Number.MIN_VALUE,random())))*Math.cos(2*Math.PI*random());
}
export function prepareEpoch({forest,receiver,signals,seconds,systems='G'}){
  const open={...forest,trees:[],count:0};
  const channels=signals.filter(s=>systems.includes(s.id[0])&&/^[A-Z]\d+$/.test(s.id)).map(s=>{
    const delta=s.sourceENU.map((v,i)=>v-receiver[i]),direction=delta.map(v=>v/Math.hypot(...delta));
    const options={densityProfile:'radial-height-v1',specificLoss:.2};
    const dry=traceForest(forest,receiver,direction,options),clear=traceForest(open,receiver,direction,options);
    return {id:s.id,sourceENU:s.sourceENU,azimuth:s.azimuth,elevation:s.elevation,canopyPathM:dry.canopyPath,
      dryLossDb:dry.canopyLoss,blocked:dry.directRelativeDb===null,terrainBlocked:clear.directRelativeDb===null,
      };
  });
  return {seconds,receiver,channels};
}
export function evaluateEpoch({seconds,receiver,channels}, {seed=7,wetLoss=.3,threshold=30}={}){
  channels=channels.map(c=>({...c,normalDraw:normal(seed,seconds,c.id)}));
  return {seconds,cases:CONDITIONS.map(condition=>{
    const rows=channels.map(c=>{
      const blocked=condition==='open'?c.terrainBlocked:c.blocked;
      const lossDb=condition==='open'?0:c.dryLossDb+(condition==='wet'?wetLoss*c.canopyPathM:0);
      const cn0=blocked?null:45-lossDb,used=cn0!==null&&cn0>=threshold;
      const sigma=used?.3*10**((45-cn0)/20):null;
      return {...c,lossDb,cn0,used,reason:blocked?'blocked':used?'used':'below-threshold',sigma,
        range:used?distance(receiver,c.sourceENU)+23.4+sigma*c.normalDraw:null};
    });
    const fit=solveRanges(rows.filter(r=>r.used));
    return {condition,channels:rows,used:rows.filter(r=>r.used).length,missing:rows.filter(r=>!r.used).length,...fit,
      horizontalErrorM:fit.position?Math.hypot(fit.position[0]-receiver[0],fit.position[1]-receiver[1]):null,
      verticalErrorM:fit.position?fit.position[2]-receiver[2]:null};
  })};
}
const meanUsedBySystem=(cases,systems)=>Object.fromEntries([...systems].map(s=>[s,cases.reduce((n,c)=>n+c.channels.filter(r=>r.used&&r.id[0]===s).length,0)/cases.length]));
export function compareEpoch(input){return evaluateEpoch(prepareEpoch(input),input);}
// Shared catalogue keeps map markers, exports and worker receiver positions identical.
export function comparisonLocations(forest=generateForest()){
  return [{id:'trail',label:'林道の中央',receiver:forestReceiver(600)},...forest.trees.map((tree,i)=>({
    id:i===0?'canopy':tree.id,treeId:tree.id,label:`${tree.id} · 幹の東1 m`,
    receiver:[tree.x+1,tree.y,terrainHeight(tree.x+1,tree.y)+1.6]
  }))];
}
export function runComparison(scene,{seed=7,wetLoss=.3,threshold=30,location='canopy',systems='G'}={},progress=()=>{}){
  if(!SYSTEM_OPTIONS.includes(systems))throw new RangeError('Unknown GNSS systems');
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff||!Number.isFinite(wetLoss)||wetLoss<0||wetLoss>2||!Number.isFinite(threshold)||threshold<20||threshold>45)throw new RangeError('Invalid comparison settings');
  const forest=generateForest(),point=comparisonLocations(forest).find(p=>p.id===location);
  if(!point)throw new RangeError('Unknown comparison location');
  const receiver=point.receiver;
  const epochs=[];
  for(let seconds=0;seconds<=1200;seconds+=30){
    epochs.push(compareEpoch({forest,receiver,signals:forestSignals(scene,receiver,seconds,systems),seconds,systems,seed,wetLoss,threshold}));
    progress({completed:epochs.length,total:41});
  }
  const summary=CONDITIONS.map((condition,i)=>{
    const cases=epochs.map(e=>e.cases[i]),errors=cases.map(c=>c.horizontalErrorM).filter(Number.isFinite).sort((a,b)=>a-b);
    return {condition,available:errors.length,total:epochs.length,availability:errors.length/epochs.length,
      horizontalP95M:errors.length?errors[Math.ceil(errors.length*.95)-1]:null,
      meanUsed:cases.reduce((s,c)=>s+c.used,0)/cases.length,
      ...(systems.length>1?{meanUsedBySystem:meanUsedBySystem(cases,systems)}:{})};
  });
  return {schema:'forest-environment-comparison-v1',simulationOnly:true,settings:{seed,wetLoss,threshold,location,systems},
    scene:forest,locationLabel:point.label,receiverENU:receiver,week:scene.week,startTow:scene.startTow??scene.track[0][0],summary,epochs,
    assumptions:{orbit:`Stored broadcast ${systems==='G'?'GPS':'GPS, Galileo, BeiDou, QZSS and GLONASS'} orbits; 10 degree elevation mask; same stationary receiver and epochs for all conditions.`,
      open:'Trees removed; terrain and receiver unchanged.',dry:'Existing synthetic linear canopy model, 0.2 dB per integrated-density metre; wood/terrain block the direct ray.',
      wet:'Additional wetLoss dB per geometric canopy metre. Hypothetical wet-canopy sensitivity, NOT rain rate or a calibrated water-content model.',
      receiver:'45 dB-Hz open-sky baseline for every system and signal; hard C/N0 selection threshold; sigma = 0.3 * 10^((45-CN0)/20) metres; independent Gaussian draws across satellite/epoch, shared across conditions.',
      positioning:'Direct geometric ranges plus 23.4 m clock and Gaussian noise; weighted least squares for 3D position plus one clock per GNSS system (inter-system offsets unknown, so each extra system costs one satellite), starting at ENU origin; no truth input to solver.',
      limits:'No multipath recovery, code tracking, atmosphere, rain-path attenuation or hardware validation. No real-world accuracy claim. Single seed; p95 is over available epochs only; failures retained.'}};
}
