import {generateForest} from './model.mjs?v=extinction-reference-1';
import {forestSignals} from './signals.mjs?v=extinction-reference-1';
import {comparisonLocations,prepareEpoch,evaluateEpoch,CONDITIONS} from './environment-comparison.mjs?v=sensitivity-1';
export const WET_LOSSES=[0,.1,.3,.6],THRESHOLDS=[25,30,35,40];
const median=values=>{const a=values.toSorted((a,b)=>a-b),n=a.length;return n?(a[Math.floor((n-1)/2)]+a[Math.floor(n/2)])/2:null;};
export function sensitivityStudy(scene,{location='canopy',seed=7}={},progress=()=>{}){
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw new RangeError('Invalid sensitivity seed');
  const forest=generateForest(),point=comparisonLocations(forest).find(p=>p.id===location);
  if(!point)throw new RangeError('Unknown sensitivity location');
  const seeds=Array.from({length:8},(_,i)=>(seed+i)>>>0),prepared=[];
  for(let seconds=0;seconds<=1200;seconds+=30){
    prepared.push(prepareEpoch({forest,receiver:point.receiver,signals:forestSignals(scene,point.receiver,seconds,'G'),seconds}));
    progress({phase:'geometry',completed:prepared.length,total:41});
  }
  const cells=[];
  for(const threshold of THRESHOLDS)for(const wetLoss of WET_LOSSES){
    const runs=seeds.map(seed=>{
      const epochs=prepared.map(e=>evaluateEpoch(e,{seed,threshold,wetLoss}));
      return {seed,conditions:CONDITIONS.map((condition,i)=>{
        const errors=epochs.map(e=>e.cases[i].horizontalErrorM).filter(Number.isFinite).sort((a,b)=>a-b);
        return {condition,available:errors.length,total:41,horizontalP95M:errors.length?errors[Math.ceil(errors.length*.95)-1]:null,
          failedSeconds:epochs.filter(e=>e.cases[i].status!=='solution').map(e=>e.seconds)};
      })};
    });
    const summary=CONDITIONS.map((condition,i)=>{
      const values=runs.map(r=>r.conditions[i]),errors=values.map(r=>r.horizontalP95M).filter(Number.isFinite);
      return {condition,minAvailable:Math.min(...values.map(r=>r.available)),maxAvailable:Math.max(...values.map(r=>r.available)),total:41,
        medianSeedP95M:median(errors),minSeedP95M:errors.length?Math.min(...errors):null,maxSeedP95M:errors.length?Math.max(...errors):null,seedsWithSolution:errors.length};
    });
    cells.push({threshold,wetLoss,runs,summary});progress({phase:'grid',completed:cells.length,total:16});
  }
  return {schema:'forest-sensitivity-v1',simulationOnly:true,settings:{location,seed,seeds,wetLosses:WET_LOSSES,thresholds:THRESHOLDS},locationLabel:point.label,
    receiverENU:point.receiver,week:scene.week,startTow:scene.startTow??scene.track[0][0],forest,cells,
    assumptions:{model:'Same paired open/dry/wet model as the epoch comparison. Geometry and broadcast endpoints computed once per epoch and reused for every cell and seed.',
      wet:'Uncalibrated extra canopy loss; not rainfall intensity or field prediction.',
      noise:'Eight consecutive uint32 seeds shared across grid cells and conditions. Gaussian range noise only; no atmospheric, forest or weather uncertainty.',
      availability:'Hard C/N0 selection is independent of the noise seed. Report numerical solver failures too. No confidence interval for availability.',
      error:'p95 over available epochs per seed; median and min/max across seeds with solutions. Missing solutions are never zero errors; min/max are not confidence bounds.'}};
}
