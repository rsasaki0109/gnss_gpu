// Controlled single-arrival ranging experiment, not a receiver tracking model.
import {randomSequence,traceForest,terrainHeight} from "./model.mjs?v=ground-1";
import {GroundReflectionModel} from "./ground-reflections.mjs?v=ground-1";
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
const vector=a=>Array.isArray(a)&&a.length===3&&a.every(Number.isFinite);
export const METHODS=["straight","robust","paths"];

// Profile the common clock in metres. Each row contains possible observed minus
// predicted ranges; path identity is latent, with normalized equal priors.
export function profileClock(rows,sigmas,method="paths"){
  if(!METHODS.includes(method)||!rows.length||rows.length!==sigmas.length||rows.some(r=>!r.length||r.some(v=>!Number.isFinite(v)))||sigmas.some(s=>!Number.isFinite(s)||s<=0))throw new RangeError("Invalid clock profile");
  const weights=sigmas.map(s=>1/(s*s)),W=weights.reduce((a,b)=>a+b,0);
  const score=b=>rows.reduce((sum,row,i)=>{
    const costs=row.map(d=>.5*((d-b)/sigmas[i])**2);
    if(method==="robust"){const z=Math.abs((row[0]-b)/sigmas[i]);return sum+(z<=1.5?.5*z*z:1.5*(z-.75));}
    if(method==="straight")return sum+costs[0];
    const low=Math.min(...costs);return sum+low-Math.log(costs.reduce((s,c)=>s+Math.exp(low-c),0)/costs.length);
  },0);
  const flat=rows.flat(),lo=Math.min(...flat),hi=Math.max(...flat);
  if(method==="straight"||method==="paths"&&rows.every(r=>r.length===1)){
    const clock=rows.reduce((s,r,i)=>s+r[0]*weights[i],0)/W;return {clock,cost:score(clock),clockCostBound:0};
  }
  if(method==="robust"){
    let a=lo,b=hi;for(let j=0;j<60;j++){const m=(a+b)/2,g=rows.reduce((s,r,i)=>s+Math.max(-1.5,Math.min(1.5,(m-r[0])/sigmas[i]))/sigmas[i],0);if(g>0)b=m;else a=m;}
    const clock=(a+b)/2;return {clock,cost:score(clock),clockCostBound:0};
  }
  // NLL second derivative <= W. Spacing <= 1/(2 sqrt(W)) bounds
  // the grid minimum's additive NLL error by 1/32, including multimodal clocks.
  const steps=Math.max(1,Math.ceil((hi-lo)*2*Math.sqrt(W)));
  if(steps>200000)throw new RangeError("Clock search exceeds experiment budget");
  let best={clock:lo,cost:score(lo),clockCostBound:1/32};
  for(let j=1;j<=steps;j++){const clock=lo+(hi-lo)*j/steps,cost=score(clock);if(cost<best.cost)best={...best,clock,cost};}
  return best;
}

export function pathRanges(forest,model,receiver,source){
  const direct=distance(receiver,source),direction=source.map((v,i)=>(v-receiver[i])/direct);
  const trace=traceForest(forest,receiver,direction,{maximum:direct});
  const ranges=[];
  if(trace.ground===null&&!trace.wood.length)ranges.push(direct);
  for(const p of model.paths(receiver,source).paths)ranges.push(p.totalRange);
  return ranges;
}

export function generateObservations(forest,receiver,signals,{seed=7,sigma=.3,clock=23.4}={}){
  if(!vector(receiver)||!Number.isFinite(sigma)||sigma<.05||sigma>3||!Number.isFinite(clock))throw new RangeError("Invalid synthetic observation assumptions");
  const random=randomSequence(seed),model=new GroundReflectionModel(forest),observations=[],associations=[],missing=[];
  for(const signal of signals.filter(s=>/^G\d+$/.test(s.id)).toSorted((a,b)=>a.id.localeCompare(b.id))){
    if(!vector(signal.sourceENU))throw new RangeError("Invalid source endpoint");
    const ranges=pathRanges(forest,model,receiver,signal.sourceENU);
    if(!ranges.length){missing.push(signal.id);continue;}
    const index=Math.floor(random()*ranges.length),noise=sigma*Math.sqrt(-2*Math.log(Math.max(Number.MIN_VALUE,random())))*Math.cos(2*Math.PI*random());
    observations.push({id:signal.id,sourceENU:signal.sourceENU.slice(),range:ranges[index]+clock+noise,sigma});
    associations.push({id:signal.id,index,available:ranges.length,range:ranges[index],noise});
  }
  return {observations,truth:{receiverENU:receiver.slice(),clock,associations,missing},generation:{seed,sigma,selection:"uniform over feasible direct and single-ground paths; artificial single arrival"}};
}

export function shiftedForest(forest,bias){
  if(!Number.isFinite(bias)||Math.abs(bias)>3)throw new RangeError("Invalid vertical map bias");
  const p=a=>[a[0],a[1],a[2]+bias],c=a=>({...a,a:p(a.a),b:p(a.b)});
  return {...forest,terrain:forest.terrain.map(t=>t.map(p)),trees:forest.trees.map(t=>({...t,trunk:c(t.trunk),branches:t.branches.map(c),crown:{...t.crown,center:p(t.crown.center)}}))};
}

// This interface intentionally accepts no evaluation truth or path labels.
export function solvePosition(observations,prior,forest,{mapBias=0,onProgress=()=>{}}={}){
  if(observations.length<3||new Set(observations.map(o=>o.id)).size!==observations.length||observations.some(o=>!/^G\d+$/.test(o.id)||!vector(o.sourceENU)||!Number.isFinite(o.range)||!Number.isFinite(o.sigma)||o.sigma<.05||o.sigma>3)||!vector(prior))throw new RangeError("Need at least three valid GPS ranges and a finite prior");
  const mapped=shiftedForest(forest,mapBias),model=new GroundReflectionModel(mapped),cache=new Map(),results=[];
  const bounds=[0,1].map(i=>[Math.max(-80,prior[i]-12),Math.min(80,prior[i]+12)]);
  if(bounds.some(([a,b])=>a>b))throw new RangeError("Prior outside patch search domain");
  const prediction=(x,y,needsPaths)=>{
    const key=`${x},${y}`;let value=cache.get(key);
    if(!value){
      const receiver=[x,y,terrainHeight(x,y)+mapBias+1.6];
      const direct=observations.map(o=>[o.range-distance(receiver,o.sourceENU)]);
      value={receiver,direct,paths:null};cache.set(key,value);
    }
    // The straight baselines never consume reflection geometry. Compute it only
    // for positions visited by the path solver, retaining identical binary64 math.
    if(needsPaths&&!value.paths)value.paths=observations.map(o=>pathRanges(mapped,model,value.receiver,o.sourceENU).map(r=>o.range-r));
    return value;
  };
  for(const method of METHODS){
    let best=null,evaluations=0,invalid=0;
    for(const [radius,step] of [[12,3],[3,1],[1,.25],[.25,.05]]){
      const center=best?.receiverENU??prior,n=Math.round(radius/step);
      for(let a=-n;a<=n;a++)for(let b=-n;b<=n;b++){
        const x=center[0]+a*step,y=center[1]+b*step;
        if(x<bounds[0][0]||x>bounds[0][1]||y<bounds[1][0]||y>bounds[1][1])continue;
        const p=prediction(x,y,method==="paths"),rows=method==="paths"?p.paths:p.direct;evaluations++;
        // Never silently discard an observation that has no path at a candidate.
        if(rows.some(r=>!r.length)){invalid++;continue;}
        const fit=profileClock(rows,observations.map(o=>o.sigma),method);
        if(!best||fit.cost<best.cost)best={...fit,receiverENU:p.receiver};
      }
      onProgress({method,step,evaluations,invalid});
    }
    results.push({method,status:best?"solution":"no-feasible-candidate",...best,evaluations,invalid,boundary:best?bounds.some(([a,b],i)=>Math.min(Math.abs(best.receiverENU[i]-a),Math.abs(best.receiverENU[i]-b))<.051):null});
  }
  return {results,search:{bounds,stages:[[12,3],[3,1],[1,.25],[.25,.05]],globalSpatialOptimum:false,uniqueCandidates:cache.size,reflectionCandidates:[...cache.values()].filter(p=>p.paths!==null).length,mapBias,receiverHeightM:1.6,clockUnits:"metres",pathPrior:"equal normalized feasible paths",clockGridAdditiveNllBound:1/32}};
}
