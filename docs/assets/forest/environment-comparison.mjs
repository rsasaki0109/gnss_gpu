import {generateForest,forestReceiver,terrainHeight,traceForest,randomSequence} from './model.mjs?v=extinction-reference-1';
import {forestSignals} from './signals.mjs?v=extinction-reference-1';
export const CONDITIONS=['open','forest','wet'];
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
// Weighted least squares, ENU position plus receiver clock in metres. The
// solver sees measurements only, never the generating receiver or noise draws.
export function solveRanges(observations){
  if(observations.length<4)return {status:'insufficient-satellites',position:null};
  const x=[0,0,0,0];
  for(let iteration=0;iteration<12;iteration++){
    const a=Array.from({length:4},()=>Array(5).fill(0));
    for(const o of observations){
      const r=distance(x.slice(0,3),o.sourceENU),h=[...x.slice(0,3).map((v,i)=>(v-o.sourceENU[i])/r),1],residual=o.range-r-x[3],w=1/o.sigma**2;
      for(let j=0;j<4;j++){for(let k=0;k<4;k++)a[j][k]+=w*h[j]*h[k];a[j][4]+=w*h[j]*residual;}
    }
    const scale=Math.max(...a.map(row=>Math.max(...row.slice(0,4).map(Math.abs))));
    for(let j=0;j<4;j++){
      let pivot=j;for(let k=j+1;k<4;k++)if(Math.abs(a[k][j])>Math.abs(a[pivot][j]))pivot=k;
      if(!Number.isFinite(scale)||Math.abs(a[pivot][j])<scale*1e-10)return {status:'singular-geometry',position:null};
      [a[j],a[pivot]]=[a[pivot],a[j]];const d=a[j][j];for(let k=j;k<5;k++)a[j][k]/=d;
      for(let k=0;k<4;k++)if(k!==j){const v=a[k][j];for(let l=j;l<5;l++)a[k][l]-=v*a[j][l];}
    }
    const delta=a.map(row=>row[4]);delta.forEach((v,i)=>x[i]+=v);
    if(!x.every(Number.isFinite))return {status:'non-finite',position:null};
    if(Math.hypot(...delta)<1e-5)return {status:'solution',position:x.slice(0,3),clockM:x[3]};
  }
  return {status:'no-convergence',position:null};
}
function normal(seed,seconds,id){
  let key=(seed^Math.imul(seconds,2654435761))>>>0;
  for(const char of id)key=(Math.imul(key,16777619)^char.charCodeAt(0))>>>0;
  const random=randomSequence(key);
  return Math.sqrt(-2*Math.log(Math.max(Number.MIN_VALUE,random())))*Math.cos(2*Math.PI*random());
}
export function prepareEpoch({forest,receiver,signals,seconds}){
  const open={...forest,trees:[],count:0};
  const channels=signals.filter(s=>/^G\d+$/.test(s.id)).map(s=>{
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
export function compareEpoch(input){return evaluateEpoch(prepareEpoch(input),input);}
// Shared catalogue keeps map markers, exports and worker receiver positions identical.
export function comparisonLocations(forest=generateForest()){
  return [{id:'trail',label:'林道の中央',receiver:forestReceiver(600)},...forest.trees.map((tree,i)=>({
    id:i===0?'canopy':tree.id,treeId:tree.id,label:`${tree.id} · 幹の東1 m`,
    receiver:[tree.x+1,tree.y,terrainHeight(tree.x+1,tree.y)+1.6]
  }))];
}
export function runComparison(scene,{seed=7,wetLoss=.3,threshold=30,location='canopy'}={},progress=()=>{}){
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff||!Number.isFinite(wetLoss)||wetLoss<0||wetLoss>2||!Number.isFinite(threshold)||threshold<20||threshold>45)throw new RangeError('Invalid comparison settings');
  const forest=generateForest(),point=comparisonLocations(forest).find(p=>p.id===location);
  if(!point)throw new RangeError('Unknown comparison location');
  const receiver=point.receiver;
  const epochs=[];
  for(let seconds=0;seconds<=1200;seconds+=30){
    epochs.push(compareEpoch({forest,receiver,signals:forestSignals(scene,receiver,seconds,'G'),seconds,seed,wetLoss,threshold}));
    progress({completed:epochs.length,total:41});
  }
  const summary=CONDITIONS.map((condition,i)=>{
    const cases=epochs.map(e=>e.cases[i]),errors=cases.map(c=>c.horizontalErrorM).filter(Number.isFinite).sort((a,b)=>a-b);
    return {condition,available:errors.length,total:epochs.length,availability:errors.length/epochs.length,
      horizontalP95M:errors.length?errors[Math.ceil(errors.length*.95)-1]:null,
      meanUsed:cases.reduce((s,c)=>s+c.used,0)/cases.length};
  });
  return {schema:'forest-environment-comparison-v1',simulationOnly:true,settings:{seed,wetLoss,threshold,location},
    scene:forest,locationLabel:point.label,receiverENU:receiver,week:scene.week,startTow:scene.startTow??scene.track[0][0],summary,epochs,
    assumptions:{orbit:'Stored broadcast GPS orbits; 10 degree elevation mask; same stationary receiver and epochs for all conditions.',
      open:'Trees removed; terrain and receiver unchanged.',dry:'Existing synthetic linear canopy model, 0.2 dB per integrated-density metre; wood/terrain block the direct ray.',
      wet:'Additional wetLoss dB per geometric canopy metre. Hypothetical wet-canopy sensitivity, NOT rain rate or a calibrated water-content model.',
      receiver:'45 dB-Hz open-sky baseline; hard C/N0 selection threshold; sigma = 0.3 * 10^((45-CN0)/20) metres; independent Gaussian draws across satellite/epoch, shared across conditions.',
      positioning:'Direct geometric ranges plus 23.4 m clock and Gaussian noise; weighted 3D+clock least squares, starting at ENU origin; no truth input to solver.',
      limits:'No multipath recovery, code tracking, atmosphere, rain-path attenuation or hardware validation. No real-world accuracy claim. Single seed; p95 is over available epochs only; failures retained.'}};
}
